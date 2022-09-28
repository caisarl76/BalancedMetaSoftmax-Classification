import argparse
import builtins
import os

os.environ['OPENBLAS_NUM_THREADS'] = '2'
import time
import json
import logging

import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from data import dataloader
from custum_data.new_dataset import get_dataset, dataset_info

from loss.BalancedSoftmaxLoss import create_loss_w_list

from utils import AverageMeter, ProgressMeter, accuracy
from models.reactnet_imagenet import reactnet
from models.DotProductClassifier import DotProduct_Classifier

Dataloader = None
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='caltech101')
parser.add_argument('--data', metavar='DIR', default='./data/')
parser.add_argument('--root_path', type=str, default='./runs/runs/dive/diveTtobinary')
parser.add_argument('--imb_ratio', type=float, default=0.1)
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--teacher_path', help='choose model to use as a teacher', default='imgnet21k', type=str)
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--scheduler', default='cosann', type=str, choices=['cosann', 'lambda'])
parser.add_argument('--warmup', action='store_true')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--optimizer', help='choose which optimizer to use', default='adam', type=str)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', default=True, type=bool,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--mark', default='tmp', type=str,
                    help='log dir')
parser.add_argument('--num_classes', default=200, type=int, help='num classes in dataset')
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('-T', type=float, default=6.0)
parser.add_argument('--tau', type=float, default=2.0)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--balms_T', type=str, default=None)  # use this as balms teacher path

def main():
    global best_acc
    global best_head
    global best_med
    global best_tail
    args = parser.parse_args()
    teacher_opts = args.teacher_path.split('/')
    args.dataset = teacher_opts[2]
    if '_' in args.dataset:
        args.dataset, args.imb_ratio = args.dataset.split('_')
        args.imb_ratio = (float)(args.imb_ratio)
    elif args.dataset == 'fruits':
        args.imb_ratio = 0.01
    else:
        args.imb_ratio = 0.1

    args.mark = '_'.join([('epochs' + (str)(args.epochs)),
                          ('bs' + (str)(args.batch_size)),
                          ('lr' + (str)(args.lr)),
                          ])
    dive_mark = '_'.join([('Temp' + (str)(args.T)),
                          ('tau' + (str)(args.tau)),
                          ('alpha' + (str)(args.alpha))
                          ])
    args.root_path = os.path.join(os.getcwd(), args.root_path, args.dataset, args.mark, dive_mark)

    if args.seed is not None:
        args.root_path = os.path.join(args.root_path, ('seed_' + (str)(args.seed)))
    # if args.maximize:
    #     args.root_path = os.path.join(args.root_path, 'maximize')

    os.makedirs(args.root_path, exist_ok=True)
    with open(os.path.join(args.root_path, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print('exp save on: ', args.root_path)

    dataset_info(args)
    train_loader = dataloader.load_data(data_root='./dataset/',
                                         dataset = args.dataset,
                                         phase = 'train',
                                         batch_size = args.batch_size,
                                         num_workers = args.workers
                                         )
    val_loader = dataloader.load_data(data_root='./dataset/',
                                        dataset=args.dataset,
                                        phase='val',
                                        batch_size=args.batch_size,
                                        num_workers=args.workers
                                        )

    args.num_classes = len(train_loader.dataset.get_cls_num_list())
    args.cls_num_list = train_loader.dataset.get_cls_num_list()
    if args.dataset == 'inat':
        args.cls_num_list = train_loader.dataset.cls_num_list
    else:
        args.cls_num_list = train_loader.dataset.get_cls_num_list()

    teacher_enc = reactnet()
    teacher_fc = DotProduct_Classifier(num_classes=args.num_classes, feat_dim=1024)
    student_enc = reactnet()
    student_fc = DotProduct_Classifier(num_classes=args.num_classes, feat_dim=1024)

    args.teacher_path = os.path.join(os.getcwd(), args.teacher_path)
    print(args.teacher_path)
    assert os.path.isfile(args.teacher_path)

    loc = 'cuda:{}'.format(args.gpu)
    checkpoint = torch.load(args.teacher_path, map_location=loc)
    enc_state_dict = checkpoint['state_dict_best']['feat_model']
    fc_state_dict = checkpoint['state_dict_best']['classifier']
    new_enc_dict = {}
    new_fc_dict = {}
    for key in enc_state_dict.keys():
        if key.strip('module.') in teacher_enc.state_dict():
            new_enc_dict[key.strip('module.')] = enc_state_dict[key]
    for key in fc_state_dict.keys():
        if key.strip('module.') in teacher_fc.state_dict():
            new_fc_dict[key.strip('module.')] = fc_state_dict[key]

    teacher_enc.load_state_dict(new_enc_dict)
    teacher_fc.load_state_dict(new_fc_dict)
    print('teacher weight load complete')

    if torch.cuda.device_count() > 1:
        teacher_enc = nn.DataParallel(teacher_enc)
        teacher_fc = nn.DataParallel(teacher_fc)
        student_enc = nn.DataParallel(student_enc)
        student_fc = nn.DataParallel(student_fc)

    if args.gpu is not None:
        device = torch.device("cuda:%d" % args.gpu)
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    student_enc.to(device)
    student_fc.to(device)
    teacher_enc.to(device)
    teacher_fc.to(device)
    parameters = [{'params': list(student_enc.parameters())}]
    parameters.append({'params': student_fc.parameters()})
    optimizer = torch.optim.Adam(parameters, args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs)

    for p in teacher_enc.parameters():
        p.requires_grad = False
    for p in teacher_fc.parameters():
        p.requires_grad = False

    teacher_enc.eval()
    teacher_fc.eval()

    cudnn.benchmark = True

    criterion_ce = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion = create_loss_w_list(cls_num_list=args.cls_num_list)

    log_dir = os.path.join(args.root_path, 'logs')
    writer = SummaryWriter(log_dir)
    log_file = os.path.join(log_dir, 'log_train.txt')
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    # logger.info('\n' + pprint.pformat(args))
    # logger.info('\n' + str(args))
    best_acc=0
    if args.evaluate:
        print(" start evaualteion **** ")
        validate(val_loader, student_enc, student_fc, criterion_ce, logger, args)
        return

    # mixed precision

    for epoch in range(args.start_epoch, args.epochs):
        # print('5')
        # train for one epoch
        train(train_loader, teacher_enc, teacher_fc, student_enc, student_fc, optimizer, criterion, epoch, args)

        acc, loss, head_acc, med_acc, tail_acc = validate(val_loader, student_enc, student_fc, criterion_ce, logger, args)
        scheduler.step()

        print("Epoch: %d, %.2f %.2f %.2f %.2f" % (epoch, acc, head_acc, med_acc, tail_acc))
        if acc > best_acc:
            best_acc = acc
            best_head = head_acc
            best_med = med_acc
            best_tail = tail_acc
        writer.add_scalar('val loss', loss, epoch)
        writer.add_scalar('val acc', acc, epoch)
        logger.info('Epoch: %d | Best Prec@1: %.3f%% | Prec@1: %.3f%% loss: %.3f\n' % (epoch, best_acc, acc, loss))
    logger.info('Best Prec@1: %.2f %.2f %.2f %.2f ' % (best_acc, best_head, best_med, best_tail))
    open(args.root_path + "/" + "log.log", "a+").write(
        'Best Prec@1: %.2f %.2f %.2f %.2f ' % (best_acc, best_head, best_med, best_tail))


def train(train_loader, teacher_enc, teacher_fc, student_enc, student_fc, optimizer, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    teacher_enc.eval()
    student_enc.train()

    end = time.time()
    for i, (images, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)

        with torch.no_grad():
            t_feat = teacher_enc(images)
            t_out, _ = teacher_fc(t_feat)

        s_feat = student_enc(images)
        s_out, _ = student_fc(s_feat)

        bsce_loss = criterion(s_out, labels)
        s_out = F.log_softmax(s_out / args.T)
        t_out = torch.sqrt(F.log_softmax(t_out / args.T))
        t_out /= t_out.sum(dim=1, keepdim=True)
        kl_loss = F.kl_div(F.log_softmax(s_out / args.T), t_out) * (args.T ** 2)
        total_loss = 0.5 * kl_loss + 0.5 * bsce_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        acc1, acc5 = accuracy(s_out, labels, topk=(1, 5))
        losses.update(total_loss.item())

        top1.update(acc1[0], s_out.size(0))
        top5.update(acc5[0], s_out.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, args)
        # break


def validate(val_loader, encoder, classifier, criterion, logger, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    total_s_out = torch.empty((0, args.num_classes)).cuda()
    total_labels = torch.empty(0, dtype=torch.long).cuda()
    class_num = torch.zeros(args.num_classes).cuda()
    correct = torch.zeros(args.num_classes).cuda()

    encoder.eval()
    classifier.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, labels, _) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                labels = labels.cuda(args.gpu, non_blocking=True)

            # compute output
            output, _ = classifier(encoder(images))
            loss = criterion(output, labels)

            total_s_out = torch.cat((total_s_out, output))
            total_labels = torch.cat((total_labels, labels))

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure head, tail classwise accuracy
            _, predicted = output.max(1)
            label_one_hot = F.one_hot(labels, args.num_classes)
            predict_one_hot = F.one_hot(predicted, args.num_classes)
            class_num = class_num + label_one_hot.sum(dim=0).to(torch.float)
            correct = correct + (label_one_hot + predict_one_hot == 2).sum(dim=0).to(torch.float)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i, args)
            # break
        acc_classes = correct / class_num
        head_acc = acc_classes[args.head_class_idx[0]:args.head_class_idx[1]].mean() * 100

        med_acc = acc_classes[args.med_class_idx[0]:args.med_class_idx[1]].mean() * 100
        tail_acc = acc_classes[args.tail_class_idx[0]:args.tail_class_idx[1]].mean() * 100
        logger.info(
            '* Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}% HAcc {head_acc:.3f}% MAcc {med_acc:.3f}% TAcc {tail_acc:.3f}%.'.format(
                top1=top1, top5=top5, head_acc=head_acc, med_acc=med_acc, tail_acc=tail_acc))

        # TODO: this should also be done with the ProgressMeter
        open(args.root_path + "/" + "log.log", "a+").write(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}\n'
                                                           .format(top1=top1, top5=top5))

    return top1.avg, losses.avg, head_acc, med_acc, tail_acc

if __name__ == '__main__':
    main()
