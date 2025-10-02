# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RF-DETR is a real-time, transformer-based object detection model. The codebase is built on PyTorch and provides:
- Pre-trained models (Nano, Small, Medium, Base, Large) for COCO object detection
- Fine-tuning capabilities for custom datasets in COCO format
- ONNX export for deployment
- Integration with Roboflow for dataset management

## Code Architecture

### Core Components

**Model Classes** (`rfdetr/detr.py`):
- `RFDETR` - Base class implementing training, inference, and optimization
- `RFDETRNano`, `RFDETRSmall`, `RFDETRMedium`, `RFDETRBase`, `RFDETRLarge` - Variant-specific implementations
- Each model automatically downloads pre-trained weights from Google Cloud Storage on first use

**Configuration** (`rfdetr/config.py`):
- `ModelConfig` - Base configuration with encoder, decoder layers, hidden dims, resolution
- `TrainConfig` - Training hyperparameters (lr, batch_size, epochs, early stopping, etc.)
- Model-specific configs define architecture differences (patch_size, num_windows, dec_layers)

**Training Pipeline** (`rfdetr/main.py`):
- `Model` class handles the main training loop
- `HOSTED_MODELS` dict maps weight filenames to GCS URLs
- Supports distributed training, EMA, gradient checkpointing, early stopping
- Integrates with TensorBoard and Weights & Biases for metrics logging

**Model Architecture** (`rfdetr/models/`):
- `lwdetr.py` - LW-DETR model implementation (foundation for RF-DETR)
- `transformer.py` - Transformer encoder/decoder
- `backbone/` - DINOv2 vision transformer backbones with windowed attention
- `matcher.py` - Hungarian matcher for detection losses
- `ops/` - Multi-scale deformable attention operators

**Data Loading** (`rfdetr/datasets/`):
- `coco.py` - COCO dataset loader (supports both official COCO and Roboflow COCO format)
- `transforms.py` - Data augmentation pipeline
- Expects datasets in `{dataset_dir}/train/` and `{dataset_dir}/test/` with `_annotations.coco.json`

**Deployment** (`rfdetr/deploy/`):
- `export.py` - ONNX export functionality
- `benchmark.py` - Performance benchmarking utilities

### Key Design Patterns

1. **Model Reinitialization**: When fine-tuning on datasets with different class counts, the detection head is automatically reinitialized (see `rfdetr/detr.py:138`)

2. **Inference Optimization**: The `.optimize_for_inference()` method creates a separate optimized model using torch.jit.trace for 2x speedup

3. **Weight Management**: Pre-trained weights are automatically downloaded and cached locally using the `HOSTED_MODELS` mapping

4. **Class Names**: Models store class names from the training dataset in `model.class_names` for inference interpretation

## Development Commands

### Installation
```bash
# Install in development mode
pip install -e .

# Install with optional dependencies
pip install -e ".[onnxexport,metrics,docs]"
```

### Training
```bash
# Train via CLI
rfdetr --coco_dir path/to/dataset

# Train from Roboflow workspace
rfdetr --api_key YOUR_KEY --workspace WORKSPACE --project_name PROJECT

# Train programmatically
python -c "from rfdetr import RFDETRBase; model = RFDETRBase(); model.train(dataset_dir='path/to/dataset', epochs=100, batch_size=4)"
```

### Inference
```bash
# Run inference programmatically
python -c "
from rfdetr import RFDETRBase
from PIL import Image

model = RFDETRBase()
model.optimize_for_inference()  # Optional: 2x speedup
detections = model.predict(Image.open('image.jpg'), threshold=0.5)
"
```

### Documentation
```bash
# Build and serve documentation locally
mkdocs serve

# Deploy documentation (requires mike)
mike deploy --push --update-aliases VERSION latest
```

### Building and Publishing
```bash
# Build package
python -m build

# Publish to PyPI (automated via GitHub Actions)
twine upload dist/*
```

## Important Implementation Details

### Dataset Format
- Training expects COCO JSON format with annotations in `{dataset_dir}/train/_annotations.coco.json`
- Test set in `{dataset_dir}/test/_annotations.coco.json`
- Class names are extracted from annotations categories
- Supports Roboflow COCO export format out of the box

### Model Variants
Each variant has different:
- `resolution` (384-576px) - input image size
- `patch_size` (14-16) - ViT patch size
- `num_windows` (2-4) - windowed attention windows
- `dec_layers` (2-4) - number of decoder layers
- `encoder` - dinov2_windowed_small or dinov2_windowed_base

### Training Configuration
Key hyperparameters in `TrainConfig`:
- `lr_encoder` - backbone learning rate (typically higher than decoder)
- `lr_vit_layer_decay` (0.8) - layer-wise learning rate decay for ViT
- `ema_decay` (0.993) - exponential moving average for model weights
- `multi_scale` (True) - multi-scale training augmentation
- `gradient_checkpointing` (False) - reduces memory at cost of speed

### Contribution Guidelines
- Use Google-style docstrings (see CONTRIBUTING.md:45-68)
- Type hints are mandatory for all function signatures
- Sign CLA by commenting "I have read the CLA Document and I sign the CLA"
- All contributions licensed under Apache 2.0

## File Organization

```
rfdetr/
├── __init__.py           # Exports model classes
├── config.py             # Pydantic configs for models and training
├── detr.py              # Main RFDETR class with train/predict/export
├── main.py              # Training pipeline and Model class
├── engine.py            # Training/evaluation loops
├── models/              # Model architecture
│   ├── lwdetr.py        # Build model and criterion
│   ├── transformer.py   # Transformer implementation
│   ├── backbone/        # DINOv2 backbones
│   └── ops/             # Deformable attention
├── datasets/            # Data loading
│   ├── coco.py
│   └── transforms.py
├── util/                # Utilities
│   ├── metrics.py       # TensorBoard/WandB logging
│   ├── early_stopping.py
│   └── box_ops.py
├── deploy/              # Export and benchmarking
└── cli/                 # CLI entry point (rfdetr command)
```

## References

The implementation builds upon:
- LW-DETR (https://github.com/Atten4Vis/LW-DETR)
- DINOv2 (https://arxiv.org/pdf/2304.07193)
- Deformable DETR (https://arxiv.org/pdf/2010.04159)
