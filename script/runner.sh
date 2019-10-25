#!/bin/bash

TPU_IP_ADDRESS=10.128.19.2

MODEL_NAME='densenet169'
SAVE_WEIGHTS=checkpoint/model_$MODEL_NAME.pth

export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
export XLA_USE_BF16=1

python src/train-xla.py \
            --model_name $MODEL_NAME \
            --log_file report/$MODEL_NAME-xla.log \
            --epochs 3 \
            --batch_size 16 \
            --log_steps 500 \
            --num_workers 8 \
            --num_cores 8 \
            --weight_decay 1e-4 \
            --lr 1e-4 \
            --slr_divisor 4 \
            --slr_div_epochs 1. \
            --n_warmup 0.15 \
            --min_lr 1e-5 \
            --csv_file_path 'data/stage_1_train.csv' \
            --path 'data/stage_1_train_images' \
            --save_pht $SAVE_WEIGHTS
            
python src/inference.py \
            --model_name $MODEL_NAME \
            --batch_size 16 \
            --num_workers 8 \
            --num_cores 8 \
            --csv_file_path 'data/stage_1_sample_submission.csv' \
            --path 'data/stage_1_test_images' \
            --weight_file $SAVE_WEIGHTS  \
            --subm_file submissions/submission_$MODEL_NAME.csv   
            
sudo shutdown -h now