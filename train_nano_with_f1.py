from rfdetr import RFDETRSmall

model = RFDETRSmall()

model.train(
    coco_path='data/512_10_percent_phase1',
    dataset_dir='data/512_10_percent_phase1',
    dataset_file="coco",
    img_size=512,
    epochs=100,
    batch_size=2,
    grad_accum_steps=8,
    output_dir='output/wildlife_512_phase_2_with_f1',
    wandb=True,
    use_varifocal_loss=True,

    # Enable F1 metric calculation
    compute_f1=True,
    f1_center_threshold=50.0,  # Max center distance in pixels for matching
    f1_score_threshold=0.5,     # Confidence threshold for predictions
)
