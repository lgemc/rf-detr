#!/usr/bin/env python3
"""
Train RF-DETR with DINOv3 Large backbone on Wildlife dataset
Loads pretrained DINOv3 weights from checkpoint
"""

from rfdetr import RFDETRDinoV3Large

# Initialize model with DINOv3 Large backbone and custom checkpoint
model = RFDETRDinoV3Large(
    dinov3_weights_path='checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'
)

# Train
model.train(
    dataset_dir='/home/lmanrique/Do/WildlifeMapper/data/rfdetr_format_560/',
    epochs=100,
    batch_size=1,
    # resume="wildlife_output_dinov3/checkpoint.pth",
    grad_accum_steps=4,
    output_dir='wildlife_output_dinov3',
    checkpoint_interval=10,
    lr=1e-4,
    lr_encoder=1.5e-4,
    early_stopping=True,
    early_stopping_patience=15,
    tensorboard=True,
    wandb=True,
    num_workers=3,
    persistent_workers=True,
    class_weights=[1.2, 1.9, 1.16, 6.37, 12.12, 1.0]
)

print("Training completed! Check wildlife_output_dinov3/ for checkpoints.")
