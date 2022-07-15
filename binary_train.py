"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""

import os
import argparse
import pprint
from data import dataloader
from run_networks import model
import warnings
import yaml
from utils import source_import, get_value
from custum_data.dataset import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=None, type=str)
parser.add_argument('--dataset', type=str, default='cifar100')
parser.add_argument('--imb_ratio', type=float, default=0.1)
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--test_open', default=False, action='store_true')
parser.add_argument('--output_logits', default=False)
parser.add_argument('--model_dir', type=str, default=None)
parser.add_argument('--save_feat', type=str, default='')

# KNN testing parameters 
parser.add_argument('--feat_type', type=str, default='cl2n')
parser.add_argument('--dist_type', type=str, default='l2')
parser.add_argument('--lr', type=float, default=0.1)

args = parser.parse_args()

num_class_dict = {
    'cifar10': 10,
    'cifar100': 100,
    'cub': 200,
    'imagenet': 1000,
    'inat': 8142,
    'fgvc': 100,
    'dogs': 120,
    'cars': 196,
    'flowers': 102,
    'dtd': 47,
    'caltech101': 102,
    'places': 365,
    'fruits': 24,
}

def update(config, args):
    # Change parameters
    config['model_dir'] = get_value(config['model_dir'], args.model_dir)
    config['training_opt']['batch_size'] = \
        get_value(config['training_opt']['batch_size'], args.batch_size)

    config['networks']['classifier']['params']['num_classes'] = num_class_dict[args.dataset]
    config['training_opt']['num_classes'] = num_class_dict[args.dataset]
    config['training_opt']['dataset'] = args.dataset

    config['networks']['classifier']['optim_params']['lr'] = args.lr
    config['networks']['feat_model']['optim_params']['lr'] = args.lr

    if config['model_dir']:
        config['model_dir'] = os.path.join(config['model_dir'],
                                           config['optimizer'],
                                           ('lr_' + (str)(args.lr))
                                           )
    config['training_opt']['log_dir'] = os.path.join(config['training_opt']['log_dir'],
                                                     config['optimizer'],
                                                     ('lr_' + (str)(args.lr))
                                                     )
    return config


# ============================================================================
# LOAD CONFIGURATIONS
with open(args.cfg) as f:
    config = yaml.load(f)
config = update(config, args)





log_dir = config['training_opt']['log_dir'].split('/')
if config['model_dir'] is not None:
    model_dir = config['model_dir'].split('/')
else:
    model_dir = None
if 'cifar' in args.dataset:
    config['training_opt']['imb_ratio'] = args.imb_ratio
    save_dir = args.dataset + '_' + (str)(args.imb_ratio)
    log_dir[2] = save_dir
    if model_dir:
        model_dir[2] = save_dir
else:
    log_dir[2] = args.dataset
    if model_dir:
        model_dir[2] = args.dataset
config['training_opt']['log_dir'] = '/'.join(log_dir)
if model_dir:
    config['model_dir'] = '/'.join(model_dir)

test_mode = args.test
test_open = args.test_open
if test_open:
    test_mode = True
output_logits = args.output_logits
training_opt = config['training_opt']
relatin_opt = config['memory']
dataset = training_opt['dataset']

if not os.path.isdir(training_opt['log_dir']):
    os.makedirs(training_opt['log_dir'])

pprint.pprint(config)

if not test_mode:

    sampler_defs = training_opt['sampler']
    if sampler_defs:
        if sampler_defs['type'] == 'MetaSampler':  # Add option for Meta Sampler
            learner = source_import(sampler_defs['def_file']).get_learner()(
                num_classes=training_opt['num_classes'],
                init_pow=sampler_defs.get('init_pow', 0.0),
                freq_path=sampler_defs.get('freq_path', None)
            ).cuda()
            sampler_dic = {
                'batch_sampler': True,
                'sampler': source_import(sampler_defs['def_file']).get_sampler(),
                'params': {'meta_learner': learner, 'batch_size': training_opt['batch_size']}
            }
    else:
        sampler_dic = None

    splits = ['train', 'train_plain', 'val']
    if dataset not in ['inat', 'imagenet']:
        splits.append('test')

    if sampler_defs and sampler_defs['type'] == 'MetaSampler':  # todo: use meta-sampler
        cbs_file = './data/ClassAwareSampler.py'
        cbs_sampler_dic = {
            'sampler': source_import(cbs_file).get_sampler(),
            'params': {'is_infinite': True}
        }
        meta = cbs_sampler_dic
    else:
        cbs_sampler_dic = None
        meta = False
    if training_opt['dataset'] in ['imagenet', 'places', 'inat']:
        training_opt['num_workers'] = 16
        training_opt['imb_ratio'] = None
    data = get_dataset(data_root='./dataset',
                       dataset=dataset,
                       sampler_dic=sampler_dic,
                       batch_size=training_opt['batch_size'],
                       num_workers=training_opt['num_workers'],
                       imb_ratio=training_opt['imb_ratio'],
                       meta=cbs_sampler_dic
                       )
    args.cls_num_list = data['train'].dataset.get_cls_num_list()
    # if 'BalancedSoftmaxLoss' in config['criterions']['PerformanceLoss']['def_file']:
    #     config['criterions']['PerformanceLoss']['loss_params']['cls_num_list'] = \
    #         data['train'].dataset.get_cls_num_list()

    if sampler_defs and sampler_defs['type'] == 'MetaSampler':  # todo: use meta-sampler
        training_model = model(config, data, test=False, meta_sample=True, learner=learner)
    else:
        training_model = model(config, data, test=False)

    training_model.train()

else:

    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data",
                            UserWarning)

    print('Under testing phase, we load training data simply to calculate \
           training data number for each class.')

    if 'iNaturalist' in training_opt['dataset']:
        splits = ['train', 'val']
        test_split = 'val'
    else:
        splits = ['train', 'val', 'test']
        test_split = 'test'
    if 'ImageNet' == training_opt['dataset']:
        splits = ['train', 'val']
        test_split = 'val'

    splits.append('train_plain')

    data = get_dataset(data_root='./dataset',
                       dataset=dataset,
                       batch_size=training_opt['batch_size'],
                       num_workers=training_opt['num_workers'],
                       imb_ratio=training_opt['imb_ratio'],
                       )

    training_model = model(config, data, test=True)
    # training_model.load_model()
    training_model.load_model(args.model_dir)
    if args.save_feat in ['train_plain', 'val', 'test']:
        saveit = True
        test_split = args.save_feat
    else:
        saveit = False

    training_model.eval(phase=test_split, openset=test_open, save_feat=saveit)

    if output_logits:
        training_model.output_logits(openset=test_open)

print('ALL COMPLETED.')
