#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# max batch-size  is 2.
accelerate launch --config_file finetune/accelerate_config_machine_single.yaml --multi_gpu  --machine_rank 0 \
  finetune/train_cogvideox_lora.py
