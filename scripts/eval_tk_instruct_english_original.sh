#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export TRANSFORMERS_CACHE=/home/yizhongw/.cache/huggingface
export CUDA_LAUNCH_BLOCKING=1
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0
export EXP_NAME=new_instruction_7

nohup python src/run_s2s.py \
    --do_eval \
    --predict_with_generate \
    --evaluation_strategy "no" \
    --model_name_or_path google/t5-efficient-tiny \
    --max_source_length 1024 \
    --max_target_length 128 \
    --generation_max_length 128 \
    --max_num_instances_per_task 100 \
    --max_num_instances_per_eval_task 100 \
    --add_task_name False \
    --add_task_definition True \
    --num_pos_examples 2 \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir data/splits/default \
    --task_dir data/tasks \
    --output_dir tk_outputs/output_english_$EXP_NAME \
    --overwrite_output_dir \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --per_device_eval_batch_size 4 \
    > tk_eval_log/eval_english_$EXP_NAME.out 2>&1 &

