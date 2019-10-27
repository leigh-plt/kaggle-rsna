#!/bin/bash

TPU_IP_ADDRESS=10.10.10.2
MODEL_NAME='inception_v3'
USE_BF16=0
PATH_DATA=data
TRAIN=1
INFERENCE=1

while getopts ":i:m:p:bfr" opt; do
  case $opt in
    b) USE_BF16=1;;
    i) TPU_IP_ADDRESS=$OPTARG;;
    m) MODEL_NAME=$OPTARG;;
    p) PATH_DATA=$OPTARG ;;
    f) INFERENCE=0;;
    r) TRAIN=0;;
    
  esac
done

export XLA_USE_BF16=$USE_BF16
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
SAVE_WEIGHTS=checkpoint/model_$MODEL_NAME.pth

if [ $TRAIN -gt 0 ]
then
python src/train-xla.py \
            --model_name $MODEL_NAME \
            --log_file report/$MODEL_NAME-xla.log \
            --num_epochs 5 \
            --batch_size 16 \
            --log_steps 500 \
            --num_workers 8 \
            --num_cores 8 \
            --weight_decay 1e-4 \
            --lr 1e-4 \
            --slr_divisor 5 \
            --slr_div_epochs 0.85 \
            --n_warmup 0.15 \
            --min_lr 5e-6 \
            --csv_file_path $PATH_DATA/stage_1_train.csv \
            --path $PATH_DATA/stage_1_train_images\
            --save_pht $SAVE_WEIGHTS
fi

if [ $INFERENCE -gt 0 ]
then         
python src/inference.py \
            --model_name $MODEL_NAME \
            --batch_size 16 \
            --num_workers 8 \
            --num_cores 8 \
            --csv_file_path $PATH_DATA/stage_1_sample_submission.csv \
            --path $PATH_DATA/stage_1_test_images \
            --weight_file $SAVE_WEIGHTS  \
            --subm_file submissions/submission_$MODEL_NAME.csv   
fi