#!/bin/bash

TPU_IP_ADDRESS=10.128.19.2
MODEL_NAME='resnet34'
SAVE_WEIGHTS=checkpoint/model_$MODEL_NAME.pth

export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
export XLA_USE_BF16=1

python src/inference.py \
            --model_name $MODEL_NAME \
            --batch_size 23 \
            --num_workers 8 \
            --num_cores 8 \
            --csv_file_path 'data/stage_1_sample_submission.csv' \
            --path 'data/stage_1_test_images' \
            --weight_file $SAVE_WEIGHTS  \
            --subm_file submissions/submission_$MODEL_NAME.csv   