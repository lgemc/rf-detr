#!/bin/bash
# Knowledge Distillation Training Script for RF-DETR
# Distills RF-DETR Large into RF-DETR Small with P3 layer addition

python run_distillation.py \
    --student_model small \
    --teacher_model small \
    --add_p3 \
    --student_weights rf-detr-small.pth \
    --teacher_weights best-small-teacher.pt \
    --dataset_file coco \
    --coco_path data/512 \
    --dataset_dir data/512 \
    --epochs 300 \
    --batch_size 1 \
    --grad_accum_steps 16 \
    --num_workers 4 \
    --lr_p3 1e-3 \
    --lr_rest 1e-4 \
    --weight_decay 1e-4 \
    --warmup_epochs 10 \
    --freeze_backbone_epochs 5 \
    --p3_warmup_epochs 10 \
    --temperature 1.0 \
    --alpha 0.9 \
    --output_dir output/distillation_small_p3 \
    --checkpoint_interval 10 \
    --eval_interval 5 \
    --device cpu \
    --wandb \
    --project rf-detr-distillation \
    --run_name small_p3_from_base_300ep
