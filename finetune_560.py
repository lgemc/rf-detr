from rfdetr import RFDETRLarge
from lora import apply_lora_to_decoder

model = RFDETRLarge()

# Apply LoRA to the decoder transformer
model.model.model = apply_lora_to_decoder(
    model.model.model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    use_dora=True,
    apply_to_decoder=True,
    apply_to_backbone=True,
    apply_to_heads=False,
)

model.train(
    coco_path='data/560',
    dataset_dir='data/560',
    dataset_file="coco",
    train_annotations='annotations/instances_train2017.json',
    val_annotations='annotations/instances_val2017.json',
    img_size=560,
    epochs=100,
    batch_size=1,
    grad_accum_steps=16,
    lr=3e-4,
    output_dir='output/wildlife_560_finetune',
    wandb=True,
    resume='output/wildlife_560/checkpoint_best_ema.pth',
)