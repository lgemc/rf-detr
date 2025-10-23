from rfdetr import RFDETRSmall

model = RFDETRSmall()

model.train(
    coco_path='data/512_10_percent_phase1',
    dataset_dir='data/512_10_percent_phase1',
    dataset_file="coco",
    #train_annotations='annotations/instances_train2017_10pct.json',
    #val_annotations='annotations/instances_val2017_10pct.json',
    img_size=512,
    epochs=100,
    batch_size=2,
    grad_accum_steps=8,
    output_dir='output/wildlife_512_phase_2',
    wandb=True,
    # lr=0.0000100,
    # lr_encoder=0.0000100,
    use_varifocal_loss=True,
    # resume='checkpoint_512_small_weights_only.pth',
    # use_balanced_sampler=True,
    # global_img_folder='../animaldet/data/herdnet/raw/train',  # Path to full images
    # lr_drop=80,
)
