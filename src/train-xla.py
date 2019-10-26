import numpy as np 
import pandas as pd
import logging

import torch
import torch.nn as nn
import torch.utils.data as D
import torch.nn.functional as F

import os, time, random, argparse

from dataset import RSNADataset
import schedulers
from utils import *
from transform import *

import torch_xla
import torch_xla.distributed.data_parallel as dp
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm

def set_seeds(seed=7117):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train_loop_fn(model, loader, device, context):
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([5,3,3,3,3,3]).to(device))
    
    log_loss = nn.BCEWithLogitsLoss(weight=torch.FloatTensor([2,1,1,1,1,1]).to(device), reduction='none')
    def metric_fn(outputs, target):
        return (log_loss(outputs, target).sum(-1) / log_loss.weight.sum()).mean()

    optimizer = context.getattr_or(
      'optimizer',
      lambda: torch.optim.AdamW(model.parameters(), lr=args.lr,
                                betas=(0.9, 0.999), weight_decay=args.weight_decay) 
    )

    lr_scheduler = context.getattr_or(
        'lr_scheduler', lambda: schedulers.wrap_optimizer_with_scheduler(
            optimizer,
            scheduler_type='WarmupAndExponentialDecayScheduler',
            scheduler_divisor=args.slr_divisor,
            scheduler_divide_every_n_epochs=args.slr_div_epochs,
            num_warmup_epochs=args.n_warmup,
            min_delta_to_update_lr=args.min_lr,
            num_steps_per_epoch=num_steps_per_epoch))
    
    score = MovingAverage(maxlen=500)
    metric = MovingAverage(maxlen=500)
    model.train()
    for x, (data, target) in loader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        xm.optimizer_step(optimizer)
        score(loss.item())
        metric(metric_fn(output, target).item())
        if x % args.log_steps == 0:
            logging.info('[{}]({:5d}) Moving average loss: {:.5f}, metric: {:.5f}'
                             .format(device, x, score.mean(), metric.mean()))
        if lr_scheduler:
            lr_scheduler.step()

def train():
    set_seeds()
    
    global num_steps_per_epoch   

    ## Create and wrap model ##
    model = model_from_name(args.model_name)
    
    devices = (
        xm.get_xla_supported_devices(max_devices=args.num_cores) if args.num_cores != 0 else [])
    model_parallel = dp.DataParallel(model,  device_ids=devices)
    logging.info('Model {} loaded and wrapped'.format(args.model_name))
    logging.info('')
    
    ## Create dataset and loader    
    patient, label = extract_patient(args.csv_file_path, return_label=True)
    
    ds = RSNADataset(patient, label, path=args.path, transform=train_transforms)
    loader = D.DataLoader(ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=args.num_workers)
    logging.info('Dataset created\n')
    logging.info('Start training model\n')

    num_steps_per_epoch = len(loader) // len(devices)

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        model_parallel(train_loop_fn, loader)
        logging.info('\nEpoch training time: {:.2f} minutes\n'.format((time.time() - start_time)/60**1))
        
    # Save weights
    state_dict = model_parallel.models[0].to('cpu').state_dict()
    torch.save(state_dict, args.save_pht)
    logging.info('')
    logging.info('Model saved')
    logging.info('')


parser = argparse.ArgumentParser(description='')

parser.add_argument('--model_name', default='resnet34', type=str, help='Model name from pretrained')

parser.add_argument('--epochs', default=30, type=int, help='Number of training epochs')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int, help='Number of wokers for loader')
parser.add_argument('--num_cores', default=8, type=int, help='Number of cores')

## optimizer params
parser.add_argument('--lr', default=5e-4, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for AdamW')

## scheduler params
parser.add_argument('--slr_divisor', default=5, type=float, help='Scheduler divisor')
parser.add_argument('--slr_div_epochs', default=20, type=float, help='Scheduler divide every N epochs')
parser.add_argument('--n_warmup', default=0.25, type=float, help='Number warmup epochs')
parser.add_argument('--min_lr', default=1e-5, type=float, help='Min delta to update lr')

## data and log files | variables
parser.add_argument('--log_steps', default=100, type=int, help='Infor about loss every N steps ')
parser.add_argument('--log_file', default='train-xla.log', type=str, help='Filename for logs')
parser.add_argument('--csv_file_path', default='data/stage_1_train.csv', type=str, help='CSV file with labels')
parser.add_argument('--path', default='data/train_images', type=str, help='Path to dicom files')
parser.add_argument('--save_pht', default='checkpoint/model.pth', type=str, help='Name for weights file')

args = parser.parse_args()

if __name__ == '__main__':
    
    logging.basicConfig(filename=args.log_file, filemode='w',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        level=logging.INFO)
    
    train()
    