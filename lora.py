# ------------------------------------------------------------------------
# RF-DETR LoRA Implementation
# ------------------------------------------------------------------------
# Based on LoRA: Low-Rank Adaptation of Large Language Models
# https://arxiv.org/abs/2106.09685
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def apply_lora_to_decoder(
    model,
    r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = True,
    target_modules: list = None,
    apply_to_backbone: bool = False,
    apply_to_decoder: bool = True,
    apply_to_heads: bool = False,
    quantize: bool = False,
    quantize_bits: int = 8,
):
    """
    Apply LoRA (Low-Rank Adaptation) to RF-DETR model.

    This function applies LoRA to specific components of the RF-DETR model:
    - Transformer decoder self-attention layers (in_proj for Q, K, V and out_proj)
    - Transformer decoder FFN layers (linear1, linear2)
    - Optionally: backbone encoder, detection heads

    Args:
        model: RF-DETR model instance (LWDETR)
        r: LoRA rank (default: 16)
        lora_alpha: LoRA scaling parameter (default: 16)
        lora_dropout: Dropout probability for LoRA layers (default: 0.0)
        use_dora: Whether to use DoRA (Weight-Decomposed Low-Rank Adaptation) (default: True)
        target_modules: List of module names to apply LoRA to. If None, uses defaults.
        apply_to_backbone: Whether to apply LoRA to the backbone encoder (default: False)
        apply_to_decoder: Whether to apply LoRA to the transformer decoder (default: True)
        apply_to_heads: Whether to apply LoRA to detection heads (default: False)
        quantize: Whether to quantize the model (default: False)
        quantize_bits: Number of bits for quantization - 4 or 8 (default: 8)

    Returns:
        Modified model with LoRA applied

    Example:
        >>> from rfdetr import RFDETRLarge
        >>> from lora import apply_lora_to_decoder
        >>>
        >>> model = RFDETRLarge()
        >>> model.model = apply_lora_to_decoder(
        ...     model.model,
        ...     r=16,
        ...     lora_alpha=16,
        ...     use_dora=True,
        ...     apply_to_decoder=True
        ... )
    """

    if target_modules is None:
        target_modules = []

        if apply_to_decoder:
            # Decoder self-attention: MultiheadAttention uses in_proj_weight/bias for Q,K,V
            # and out_proj for output projection
            target_modules.extend([
                "in_proj",      # Covers Q, K, V projections in MultiheadAttention
                "out_proj",     # Output projection in MultiheadAttention
                "linear1",      # First FFN layer
                "linear2",      # Second FFN layer
            ])

        if apply_to_heads:
            # Detection heads
            target_modules.extend([
                "class_embed",  # Classification head
                "bbox_embed",   # Bounding box regression head
            ])

        if apply_to_backbone:
            # Backbone encoder (DINOv2)
            target_modules.extend([
                "q_proj", "v_proj", "k_proj",  # Standard attention projections
                "qkv",                          # Fused QKV projection
                "query", "key", "value",        # Alternative naming
                "cls_token", "register_tokens", # DINOv2 specific
            ])

    print(f"Applying LoRA to RF-DETR model")
    print(f"  LoRA rank (r): {r}")
    print(f"  LoRA alpha: {lora_alpha}")
    print(f"  LoRA dropout: {lora_dropout}")
    print(f"  Use DoRA: {use_dora}")
    print(f"  Quantize: {quantize}")
    if quantize:
        print(f"  Quantization bits: {quantize_bits}")
    print(f"  Target modules: {target_modules}")
    print(f"  Apply to backbone: {apply_to_backbone}")
    print(f"  Apply to decoder: {apply_to_decoder}")
    print(f"  Apply to heads: {apply_to_heads}")

    # Prepare model for quantization if requested
    if quantize:
        print(f"\nPreparing model for {quantize_bits}-bit training...")

        # Use PEFT's prepare_model_for_kbit_training
        # This enables gradient checkpointing and casts layer norms to fp32
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

        # Note: For full quantization support, the model needs to be loaded with
        # BitsAndBytesConfig during initialization. Since RF-DETR uses custom
        # checkpoint loading, we can only prepare for k-bit training here.
        # The actual quantization would need to happen during model creation.
        print(f"Model prepared for {quantize_bits}-bit training")
        print("Note: Full quantization requires BitsAndBytesConfig during model loading.")

    # Configure LoRA
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        target_modules=target_modules,
        modules_to_save=None,  # Don't save any modules separately
    )

    # Apply LoRA using PEFT
    model = get_peft_model(model, lora_config)

    # Ensure only LoRA parameters are trainable
    # This prevents issues with parameter grouping in the optimizer
    for name, param in model.named_parameters():
        if 'lora' not in name.lower():
            param.requires_grad = False

    # Override the backbone's get_named_param_lr_pairs method to return empty dict
    # This prevents the parameter grouping logic from creating duplicate groups
    original_backbone = model.backbone[0]
    def get_named_param_lr_pairs_override(args, prefix=""):
        return {}
    original_backbone.get_named_param_lr_pairs = get_named_param_lr_pairs_override

    # Print trainable parameters
    trainable_params = 0
    frozen_params = 0
    all_params = 0

    print("\n" + "="*80)
    print("Trainable Parameters:")
    print("="*80)

    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"  ✓ {name}: {param.numel():,}")
        else:
            frozen_params += param.numel()

    print("\n" + "="*80)
    print("LoRA Summary:")
    print("="*80)
    print(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
    print(f"  Frozen params: {frozen_params:,} ({100 * frozen_params / all_params:.2f}%)")
    print(f"  All params: {all_params:,}")
    print(f"  Memory reduction: {100 * (1 - trainable_params / all_params):.2f}%")
    print("="*80 + "\n")

    return model


def print_lora_parameters(model):
    """
    Print information about LoRA parameters in the model.

    Args:
        model: Model with LoRA applied
    """
    print("\n" + "="*80)
    print("LoRA Parameters Summary")
    print("="*80)

    lora_params = {}
    for name, module in model.named_modules():
        if "lora" in name.lower():
            param_count = sum(p.numel() for p in module.parameters())
            lora_params[name] = param_count

    if lora_params:
        for name, count in sorted(lora_params.items()):
            print(f"  {name}: {count:,} parameters")
        print(f"\nTotal LoRA parameters: {sum(lora_params.values()):,}")
    else:
        print("  No LoRA parameters found in model")

    print("="*80 + "\n")


def get_lora_state_dict(model):
    """
    Extract only the LoRA parameters from the model state dict.
    This is useful for saving only the LoRA weights.

    Args:
        model: Model with LoRA applied

    Returns:
        Dictionary containing only LoRA parameters
    """
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora" in name.lower() or param.requires_grad:
            lora_state_dict[name] = param
    return lora_state_dict


def save_lora_weights(model, path: str):
    """
    Save only the LoRA weights to a file.

    Args:
        model: Model with LoRA applied
        path: Path to save the weights
    """
    lora_state_dict = get_lora_state_dict(model)
    torch.save(lora_state_dict, path)
    print(f"LoRA weights saved to {path}")
    print(f"  Saved {len(lora_state_dict)} parameters")
    total_size = sum(p.numel() * p.element_size() for p in lora_state_dict.values())
    print(f"  File size: {total_size / 1024 / 1024:.2f} MB")