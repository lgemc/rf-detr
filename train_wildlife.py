#!/usr/bin/env python3
"""
Train RF-DETR on Wildlife dataset
"""

from rfdetr import RFDETRNano

# Initialize model
model = RFDETRNano()

# Train
model.train(
    dataset_dir='/home/lmanrique/Do/WildlifeMapper/data/rfdetr_format',
    epochs=100,
    batch_size=6,
    # resume="wildlife_output/checkpoint.pth",
    grad_accum_steps=4,
    output_dir='wildlife_output',
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

print("Training completed! Check wildlife_output/ for checkpoints.")
