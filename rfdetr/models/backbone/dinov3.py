# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DINOv3 integration
# Based on Meta's DINOv3 (https://github.com/facebookresearch/dinov3)
# Copyright (c) Meta Platforms, Inc. and affiliates.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from typing import List

# Size configurations for DINOv3 models
size_to_width = {
    "small": 384,
    "base": 768,
    "large": 1024,
    "giant": 1536,
}

size_to_model_name = {
    "small": "dinov3_vits16",
    "base": "dinov3_vitb16",
    "large": "dinov3_vitl16",
    "giant": "dinov3_vitg16",
}


class DinoV3(nn.Module):
    """
    DINOv3 backbone for RF-DETR.

    This implementation uses the official DINOv3 models from Meta with:
    - RoPE (Rotary Position Embeddings) instead of absolute position embeddings
    - Storage tokens for improved representation learning
    - SwiGLU FFN for better feature extraction

    Args:
        shape: Input image shape (height, width)
        out_feature_indexes: Which transformer layers to extract features from
        size: Model size ('small', 'base', 'large', 'giant')
        use_storage_tokens: Whether to use storage tokens (similar to register tokens)
        gradient_checkpointing: Enable gradient checkpointing to save memory
        load_dinov3_weights: Whether to load pretrained DINOv3 weights
        patch_size: Patch size for the ViT (default: 16 for DINOv3)
        dinov3_weights_path: Optional path to local DINOv3 weights
    """

    def __init__(
        self,
        shape=(640, 640),
        out_feature_indexes=[5, 11, 17, 23],
        size="base",
        use_storage_tokens=True,
        gradient_checkpointing=False,
        load_dinov3_weights=True,
        patch_size=16,
        dinov3_weights_path=None,
    ):
        super().__init__()

        self.shape = shape
        self.patch_size = patch_size
        self.out_feature_indexes = out_feature_indexes
        self.size = size

        # Get model name
        model_name = size_to_model_name.get(size)
        if model_name is None:
            raise ValueError(f"Unsupported size: {size}. Choose from {list(size_to_model_name.keys())}")

        # Try to load DINOv3 from the DEIMv2 directory if it exists
        dinov3_path = "/home/lmanrique/Do/DEIMv2/dinov3"
        if os.path.exists(dinov3_path):
            print(f"Loading DINOv3 from local path: {dinov3_path}")

            # If custom weights are provided, load architecture first, then load weights manually
            if dinov3_weights_path is not None and os.path.exists(dinov3_weights_path):
                print(f"Loading DINOv3 architecture for custom checkpoint")
                abs_weights_path = os.path.abspath(dinov3_weights_path)

                # Pass the weights path as a string - the hub will extract the hash
                # and automatically configure untie_global_and_local_cls_norm
                # For SAT493M checkpoints (hash: eadcf0ff), this enables local_cls_norm
                self.encoder = torch.hub.load(
                    dinov3_path, model_name, source='local',
                    pretrained=False,
                    weights=abs_weights_path  # Hub extracts hash and configures architecture
                )

                # Now manually load the custom checkpoint
                print(f"Loading custom DINOv3 weights from {dinov3_weights_path}")
                checkpoint = torch.load(abs_weights_path, map_location='cpu')
                self.encoder.load_state_dict(checkpoint, strict=True)
            elif load_dinov3_weights:
                # Load default pretrained weights
                self.encoder = torch.hub.load(
                    dinov3_path, model_name, source='local',
                    pretrained=True
                )
            else:
                # No weights at all
                self.encoder = torch.hub.load(
                    dinov3_path, model_name, source='local',
                    pretrained=False
                )
        else:
            print(f"Loading DINOv3 from torch hub (this might download the model)")
            # If the local DINOv3 repo doesn't exist, try loading from PyTorch Hub
            # Note: This requires internet connection and the official DINOv3 repo
            raise FileNotFoundError(
                f"DINOv3 repository not found at {dinov3_path}. "
                "Please clone DINOv3 from https://github.com/facebookresearch/dinov3 "
                "to /home/lmanrique/Do/DEIMv2/dinov3 or provide a custom path."
            )

        # Trim layers beyond the last feature index we need
        max_layer_idx = max(out_feature_indexes)
        if hasattr(self.encoder, 'blocks'):
            while len(self.encoder.blocks) > (max_layer_idx + 1):
                del self.encoder.blocks[-1]

        # Remove the head as we don't need it for detection
        if hasattr(self.encoder, 'head'):
            del self.encoder.head

        # Set gradient checkpointing if requested
        if gradient_checkpointing:
            if hasattr(self.encoder, 'set_grad_checkpointing'):
                self.encoder.set_grad_checkpointing(enable=True)
            else:
                print("Warning: Gradient checkpointing not supported by this DINOv3 model")

        self._out_feature_channels = [size_to_width[size]] * len(out_feature_indexes)
        self._export = False

    def export(self):
        """Prepare model for export (e.g., ONNX)"""
        if self._export:
            return
        self._export = True

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through DINOv3 backbone.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            List of feature tensors at different scales
        """
        B, C, H, W = x.shape

        # Get intermediate layer features from DINOv3
        # DINOv3's get_intermediate_layers returns (features, cls_token) tuples
        features = self.encoder.get_intermediate_layers(
            x,
            n=self.out_feature_indexes,
            return_class_token=False,
            norm=True,
            reshape=False,  # Keep as (B, N, C) format
        )

        # Reshape features from (B, N, C) to (B, C, H, W)
        output_features = []
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

        for feat in features:
            # feat shape: (B, N, C) where N = h_patches * w_patches
            # Reshape to (B, C, H_patches, W_patches)
            feat = feat.permute(0, 2, 1)  # (B, C, N)
            feat = feat.reshape(B, -1, h_patches, w_patches)  # (B, C, H_patches, W_patches)
            output_features.append(feat)

        return output_features
