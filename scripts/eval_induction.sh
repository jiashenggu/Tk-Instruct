#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export TRANSFORMERS_CACHE=/home/yizhongw/.cache/huggingface
export CUDA_LAUNCH_BLOCKING=1
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=7
export EXP_NAME=induction_multiple

# run_s2s_crud, output_xlingual_insert, eval_xlingual_insert.out

nohup python src/run_s2s_induction.py \
    --do_eval \
    --predict_with_generate \
    --evaluation_strategy "no" \
    --model_name_or_path allenai/mtk-instruct-3b-def-pos \
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
    --data_dir induction_data/ \
    --task_dir induction_data/tasks \
    --output_dir induction_outputs/output_xlingual_$EXP_NAME \
    --overwrite_output_dir \
    --cache_dir ./cache/ \
    --overwrite_cache \
    --per_device_eval_batch_size 4 \
    > induction_outputs/eval_xlingual_$EXP_NAME.out 2>&1 &