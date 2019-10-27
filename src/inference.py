import numpy as np 
import pandas as pd
import logging

import torch
import torch.nn as nn
import torch.utils.data as D
import torch.nn.functional as F

import torch_xla
import torch_xla.distributed.data_parallel as dp
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm

import os, time, random, argparse

from dataset import RSNADataset
from utils import *
from transform import *

def infer_loop(model, loader, device, context):

    patientid = []
    score = np.empty((0,6))
    model.eval()
    for x, (data, patient) in enumerate(loader):
        output = model(data)
        patientid += list(patient)
        score = np.concatenate([score, output.detach().cpu().sigmoid().numpy()])
        
    return np.round_(score.flatten(), decimals=4), patientid

def inference():
    
    model = model_from_name(args.model_name)
    
    devices = (
        xm.get_xla_supported_devices(max_devices=args.num_cores) if args.num_cores != 0 else [])
    
    model.load_state_dict(torch.load(args.weight_file))
    
    model_parallel = dp.DataParallel(model,  device_ids=devices)
    
    patient = extract_patient(args.csv_file_path, return_label=False)
    ## For all predictions files must be multiple by num_cores*batch_size
    extra = args.num_cores * args.batch_size - len(patient) % args.num_cores * args.batch_size
    patient = np.concatenate([patient, patient[:extra]])
    
    ds = RSNADataset(patient, path=args.path, transform=train_transforms)
    loader = D.DataLoader(ds, batch_size=args.batch_size,
                          shuffle=False, num_workers=args.num_workers)
    
    result = model_parallel(infer_loop, loader)
    result = np.array(result)
    score, patientid = [result[:,i] for i in range(result.shape[-1])]
    score = np.concatenate(score)
    patientid = np.concatenate(patientid)
    
    prediction_to_df(score, patientid, args.subm_file)
    
parser = argparse.ArgumentParser(description='')

parser.add_argument('--model_name', default='resnet34', type=str, help='Model name from pretrained')

parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int, help='Number of wokers for loader')
parser.add_argument('--num_cores', default=8, type=int, help='Number of cores')

## data and log files | variables
parser.add_argument('--csv_file_path', default='data/stage_1_test.csv', type=str, help='CSV file with PatientID')
parser.add_argument('--path', default='data/test_images', type=str, help='Path to dicom files')
parser.add_argument('--weight_file', default='checkpoint/model.pth', type=str, help='Path to weiths files')
parser.add_argument('--subm_file', default='submision.csv', type=str, help='Name for weights file')

args = parser.parse_args()

if __name__ == '__main__':
    
    inference()
    
    
    