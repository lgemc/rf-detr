# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from functools import partial
import torch
import torch.nn.functional as F
from torch import nn

from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM, AutoConfig, AutoBackbone
from peft import LoraConfig, get_peft_model, PeftModel

from rfdetr.util.misc import NestedTensor, is_main_process

from rfdetr.models.backbone.base import BackboneBase
from rfdetr.models.backbone.projector import MultiScaleProjector
from rfdetr.models.backbone.dinov2 import DinoV2
from rfdetr.models.backbone.dinov3 import DinoV3

__all__ = ["Backbone"]


class Backbone(BackboneBase):
    """backbone."""
    def __init__(self,
                 name: str,
                 pretrained_encoder: str=None,
                 window_block_indexes: list=None,
                 drop_path=0.0,
                 out_channels=256,
                 out_feature_indexes: list=None,
                 projector_scale: list=None,
                 use_cls_token: bool = False,
                 freeze_encoder: bool = False,
                 layer_norm: bool = False,
                 target_shape: tuple[int, int] = (640, 640),
                 rms_norm: bool = False,
                 backbone_lora: bool = False,
                 gradient_checkpointing: bool = False,
                 load_dinov2_weights: bool = True,
                 patch_size: int = 14,
                 num_windows: int = 4,
                 positional_encoding_size: bool = False,
                 dinov3_weights_path: str = None,
                 ):
        super().__init__()
        # Parse backbone name to determine which backbone to use
        # Examples:
        # - "dinov2_base" or "dinov2_registers_windowed_base" -> DINOv2
        # - "dinov3_base" or "dinov3_storage_base" -> DINOv3
        name_parts = name.split("_")
        backbone_type = name_parts[0]

        assert backbone_type in ["dinov2", "dinov3"], f"Backbone type must be 'dinov2' or 'dinov3', got '{backbone_type}'"

        if backbone_type == "dinov2":
            # DINOv2 backbone with optional registers and windowed attention
            size = name_parts[-1]
            use_registers = False
            if "registers" in name_parts:
                use_registers = True
                name_parts.remove("registers")
            use_windowed_attn = False
            if "windowed" in name_parts:
                use_windowed_attn = True
                name_parts.remove("windowed")
            assert len(name_parts) == 2, "name should be dinov2, then either registers, windowed, both, or none, then the size"
            self.encoder = DinoV2(
                size=name_parts[-1],
                out_feature_indexes=out_feature_indexes,
                shape=target_shape,
                use_registers=use_registers,
                use_windowed_attn=use_windowed_attn,
                gradient_checkpointing=gradient_checkpointing,
                load_dinov2_weights=load_dinov2_weights,
                patch_size=patch_size,
                num_windows=num_windows,
                positional_encoding_size=positional_encoding_size,
            )
        elif backbone_type == "dinov3":
            # DINOv3 backbone with RoPE and storage tokens
            size = name_parts[-1]
            use_storage_tokens = True
            if "nostorage" in name_parts:
                use_storage_tokens = False
                name_parts.remove("nostorage")

            # For DINOv3, patch_size is always 16
            if patch_size != 16:
                print(f"Warning: DINOv3 uses patch_size=16, ignoring requested patch_size={patch_size}")
                patch_size = 16

            self.encoder = DinoV3(
                size=size,
                out_feature_indexes=out_feature_indexes,
                shape=target_shape,
                use_storage_tokens=use_storage_tokens,
                gradient_checkpointing=gradient_checkpointing,
                load_dinov3_weights=load_dinov2_weights,  # Reuse the same flag
                patch_size=patch_size,
                dinov3_weights_path=dinov3_weights_path,
            )
        # build encoder + projector as backbone module
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.projector_scale = projector_scale
        assert len(self.projector_scale) > 0
        # x[0]
        assert (
            sorted(self.projector_scale) == self.projector_scale
        ), "only support projector scale P3/P4/P5/P6 in ascending order."
        level2scalefactor = dict(P3=2.0, P4=1.0, P5=0.5, P6=0.25)
        scale_factors = [level2scalefactor[lvl] for lvl in self.projector_scale]

        self.projector = MultiScaleProjector(
            in_channels=self.encoder._out_feature_channels,
            out_channels=out_channels,
            scale_factors=scale_factors,
            layer_norm=layer_norm,
            rms_norm=rms_norm,
        )

        self._export = False

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export

        if isinstance(self.encoder, PeftModel):
            print("Merging and unloading LoRA weights")
            self.encoder.merge_and_unload()

    def forward(self, tensor_list: NestedTensor):
        """ """
        # (H, W, B, C)
        feats = self.encoder(tensor_list.tensors)
        feats = self.projector(feats)
        # x: [(B, C, H, W)]
        out = []
        for feat in feats:
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[
                0
            ]
            out.append(NestedTensor(feat, mask))
        return out

    def forward_export(self, tensors: torch.Tensor):
        feats = self.encoder(tensors)
        feats = self.projector(feats)
        out_feats = []
        out_masks = []
        for feat in feats:
            # x: [(B, C, H, W)]
            b, _, h, w = feat.shape
            out_masks.append(
                torch.zeros((b, h, w), dtype=torch.bool, device=feat.device)
            )
            out_feats.append(feat)
        return out_feats, out_masks

    def get_named_param_lr_pairs(self, args, prefix: str = "backbone.0"):
        num_layers = args.out_feature_indexes[-1] + 1
        backbone_key = "backbone.0.encoder"
        named_param_lr_pairs = {}
        for n, p in self.named_parameters():
            n = prefix + "." + n
            if backbone_key in n and p.requires_grad:
                lr = (
                    args.lr_encoder
                    * get_dinov2_lr_decay_rate(
                        n,
                        lr_decay_rate=args.lr_vit_layer_decay,
                        num_layers=num_layers,
                    )
                    * args.lr_component_decay**2
                )
                wd = args.weight_decay * get_dinov2_weight_decay_rate(n)
                named_param_lr_pairs[n] = {
                    "params": p,
                    "lr": lr,
                    "weight_decay": wd,
                }
        return named_param_lr_pairs


def get_dinov2_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    """
    Calculate lr decay rate for different ViT blocks.
    Works for both DINOv2 and DINOv3 backbones.

    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.
    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if "embeddings" in name or "patch_embed" in name:
            layer_id = 0
        elif ".layer." in name and ".residual." not in name:
            layer_id = int(name[name.find(".layer.") :].split(".")[2]) + 1
        elif ".blocks." in name:
            # DINOv3 uses .blocks. instead of .layer.
            try:
                layer_id = int(name.split(".blocks.")[1].split(".")[0]) + 1
            except (IndexError, ValueError):
                pass
    return lr_decay_rate ** (num_layers + 1 - layer_id)

def get_dinov2_weight_decay_rate(name, weight_decay_rate=1.0):
    """
    Calculate weight decay rate for different parameters.
    Works for both DINOv2 and DINOv3 backbones.
    """
    if (
        ("gamma" in name)
        or ("pos_embed" in name)
        or ("rope_embed" in name)  # DINOv3 RoPE embeddings
        or ("rel_pos" in name)
        or ("bias" in name)
        or ("norm" in name)
        or ("embeddings" in name)
        or ("cls_token" in name)
        or ("storage_tokens" in name)  # DINOv3 storage tokens
        or ("register_tokens" in name)  # DINOv2 register tokens
    ):
        weight_decay_rate = 0.0
    return weight_decay_rate
