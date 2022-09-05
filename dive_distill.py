import argparse
import builtins
import os

from models import resnet_imagenet

os.environ['OPENBLAS_NUM_THREADS'] = '2'
import random
import time
import json
import warnings
import logging

import torch.multiprocessing as mp
import torch.nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist

from custum_dataset.new_dataset import get_dataset

from utils.utils import *
from utils.scheduler import GradualWarmupScheduler
from utils.BalancedSoftmaxLoss import create_loss
from utils.feature_similarity import similarityLoss, transfer_conv
from utils.mislas import LearnableWeightScaling

from models.reactnet_imagenet import reactnet as reactnet_imagenet, Classifier

from run_networks import model
from data import dataloader

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
parser.add_argument('-b', '--batch-size', default=128, type=int,
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
parser.add_argument('--gpu', default=None, type=int,
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
parser.add_argument('-T', type=float, default=2.0)
parser.add_argument('--tau', type=float, default=2.0)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--balms_T', type=str, default=None) # use this as balms teacher path

best_acc = 0
best_head = 0
best_med = 0
best_tail = 0


def main():
    args = parser.parse_args()
    teacher_opts = args.teacher_path.split('/')
    dataset, _, model_type = teacher_opts[7:10]
    model_type, args.imb_ratio = model_type.split('_balms_imba')
    args.dataset = dataset.strip('_LT').lower()
    args.imb_ratio = 1 / (int)(args.imb_ratio)

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

    if args.dist_url.endswith(':'):
        args.dist_url += (str)(np.random.randint(9000, 11000))

    if args.seed is not None:
        random.seed(args.seed)
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(args.seed)

        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # args.lr *= args.batch_size / 128
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc
    global best_head
    global best_med
    global best_tail
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    train_dataset, val_dataset = get_dataset(args)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    if args.dataset == 'inat':
        args.cls_num_list = train_dataset.cls_num_list
    else:
        args.cls_num_list = train_dataset.get_cls_num_list()
    config = {
        'model_dir':args.teacher_path,
    }

    teacher_model = model(config, data, test=True)
    args.teacher_path = os.path.join(os.getcwd(), args.teacher_path)
    print(args.teacher_path)
    assert os.path.isfile(args.teacher_path)

    loc = 'cuda:{}'.format(args.gpu)
    checkpoint = torch.load(args.teacher_path, map_location=loc)

    state_dict = checkpoint['state_dict_best']
    teacher_model.load_state_dict(state_dict)
    print('teacher weight load complete')

    student_enc = reactnet_imagenet(return_feat=True)
    student_enc = torch.nn.SyncBatchNorm.convert_sync_batchnorm(student_enc)
    student_fc = Classifier(num_classes=args.num_classes)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            student_enc.cuda(args.gpu)
            student_fc.cuda(args.gpu)
            teacher_model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            student_enc = torch.nn.parallel.DistributedDataParallel(student_enc, device_ids=[args.gpu],
                                                                    find_unused_parameters=True)
            student_fc = torch.nn.parallel.DistributedDataParallel(student_fc, device_ids=[args.gpu],
                                                                   find_unused_parameters=True)
            teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[args.gpu],
                                                                      find_unused_parameters=True)
        else:
            student_enc.cuda()
            student_fc.cuda()
            teacher_model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            student_enc = torch.nn.parallel.DistributedDataParallel(student_enc, find_unused_parameters=True)
            student_fc = torch.nn.parallel.DistributedDataParallel(student_fc, find_unused_parameters=True)
            teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, find_unused_parameters=True)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        student_enc = student_enc.cuda(args.gpu)
        student_fc = student_fc.cuda(args.gpu)
        teacher_model = teacher_model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in

        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    parameters = [{'params': student_enc.parameters()}]
    parameters.append({'params': student_fc.parameters()})
    optimizer = torch.optim.Adam(student_enc.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs)

    args.teacher_path = os.path.join(os.getcwd(), args.teacher_path)
    print(args.teacher_path)
    assert os.path.isfile(args.teacher_path)

    loc = 'cuda:{}'.format(args.gpu)
    checkpoint = torch.load(args.teacher_path, map_location=loc)

    state_dict = checkpoint['state_dict_best']
    teacher_model.load_state_dict(state_dict)
    print('teacher weight load complete')

    for p in teacher_model.parameters():
        p.requires_grad = False
    teacher_model.eval()

    cudnn.benchmark = True

    criterion_ce = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion = create_loss(cls_num_list=args.cls_num_list)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

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

    if args.evaluate:
        print(" start evaualteion **** ")
        validate(val_loader, student_enc, student_fc, criterion_ce, logger, args)
        return

    # mixed precision

    for epoch in range(args.start_epoch, args.epochs):
        # print('5')

        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        train(train_loader, teacher_model, lws_model, student_enc, student_fc, optimizer, criterion, epoch, args)

        acc, loss, head_acc, med_acc, tail_acc = validate(val_loader, student_enc, student_fc, criterion_ce, logger,
                                                          args)
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


def train(train_loader, teacher_model, lws_model, student_enc, student_fc, optimizer,
          criterion, epoch, args):
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
    teacher_model.eval()
    lws_model.eval()

    student_enc.train()
    student_fc.train()

    iters = len(train_loader)
    cls_num_list = torch.tensor(args.cls_num_list)
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        with torch.no_grad():
            t_feat, t_out = teacher_model(images)
            t_out = lws_model(t_out)
        _, s_out = student_enc(images)
        s_out = student_fc(s_out)

        bsce_loss = criterion(s_out, target)
        s_out = F.log_softmax(s_out / args.T)
        t_out = torch.sqrt(F.log_softmax(t_out / args.T))
        t_out /= t_out.sum(dim=1, keepdim=True)
        # kl_loss = ((F.kl_div(s_out, t_out, reduction='none').sum(1) * (t_out.argmax(1)==target)) * (args.T ** 2)).sum() / (t_out.argmax(1)==target).sum()
        kl_loss = F.kl_div(F.log_softmax(s_out / args.T), t_out) * (args.T ** 2)
        total_loss = 0.5 * kl_loss + 0.5 * bsce_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        acc1, acc5 = accuracy(s_out, target, topk=(1, 5))
        losses.update(total_loss.item())

        top1.update(acc1[0], s_out.size(0))
        top5.update(acc5[0], s_out.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, args)
        # break


def validate(val_loader, model, classifier, criterion, logger, args):
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

    model.eval()
    classifier.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            _, feat = model(images)
            output = classifier(feat)
            loss = criterion(output, target)

            total_s_out = torch.cat((total_s_out, output))
            total_labels = torch.cat((total_labels, target))

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure head, tail classwise accuracy
            _, predicted = output.max(1)
            target_one_hot = F.one_hot(target, args.num_classes)
            predict_one_hot = F.one_hot(predicted, args.num_classes)
            class_num = class_num + target_one_hot.sum(dim=0).to(torch.float)
            correct = correct + (target_one_hot + predict_one_hot == 2).sum(dim=0).to(torch.float)

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
