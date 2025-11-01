import os
from rfdetr import RFDETRMedium
from rfdetr.config import RFDETRMediumConfig

class RFDETRBaseV3(RFDETRMedium):
    def get_model_config(self, **kwargs):
        return RFDETRMediumConfig(
            encoder="dinov3_small",
            projector_scale=["P3", "P4"],
            pretrain_weights=None,
            dinov3_repo_dir=os.path.join(os.getcwd(), 'dinov3'),
            dinov3_weights_path=os.path.expanduser('~/Downloads/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'),
            dinov3_prefer_hf=False,
            **kwargs
        )

model = RFDETRBaseV3()

model.train(
    dataset_dir='../animaldet/data/coco/',
    coco_path='../animaldet/data/coco/',
    dataset_file="coco",
    output_dir='coco_phase_1',
    epochs=100,
    batch_size=8,
    grad_accum_steps=8,
    num_workers=2,
    run_test=True,
    wandb=True,
)
