"""
Knowledge Distillation Training Script for RF-DETR

This script demonstrates how to run knowledge distillation from a large teacher
model to a small student model with P3 layer addition.

Based on the principles from "Knowledge distillation: A good teacher is patient and consistent"
"""

import argparse
import os
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import rfdetr.util.misc as utils

from rfdetr.config import RFDETRSmallConfig, RFDETRLargeConfig
from rfdetr.datasets import build_dataset
from rfdetr.models import build_criterion_and_postprocessors
from distil import (
    create_distillation_setup,
    StableOptimizer,
    CosineScheduleWithWarmup,
    GradualUnfreezing,
    P3LayerInjector
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def get_args_parser():
    parser = argparse.ArgumentParser('RF-DETR Knowledge Distillation', add_help=False)

    # Model configs
    parser.add_argument('--student_model', default='small', type=str,
                        choices=['nano', 'small', 'medium'],
                        help='Student model size')
    parser.add_argument('--teacher_model', default='large', type=str,
                        choices=['nano', 'small', 'medium', 'base', 'large'],
                        help='Teacher model size')
    parser.add_argument('--add_p3', action='store_true',
                        help='Add P3 layer to student model')

    # Pretrained weights
    parser.add_argument('--student_weights', type=str, default='rf-detr-small.pth',
                        help='Path to pretrained student weights')
    parser.add_argument('--teacher_weights', type=str, default='best-small-teacher.pt',
                        help='Path to pretrained teacher weights')

    # Dataset
    parser.add_argument('--dataset_file', default='coco', type=str)
    parser.add_argument('--coco_path', type=str, default='data/512')
    parser.add_argument('--dataset_dir', type=str, default='data/512')

    # Distillation settings
    parser.add_argument('--temperature', default=1.0, type=float,
                        help='Temperature for distillation loss')
    parser.add_argument('--alpha', default=0.9, type=float,
                        help='Weight for distillation loss (vs task loss)')
    parser.add_argument('--use_mixup', action='store_true', default=False,
                        help='Use aggressive mixup for function matching')
    parser.add_argument('--mixup_p', default=0.0, type=float,
                        help='Probability of applying mixup')
    parser.add_argument('--feature_distillation', action='store_true',
                        help='Enable feature-level distillation')

    # Training settings - PATIENT TRAINING (long schedule)
    parser.add_argument('--epochs', default=300, type=int,
                        help='Number of training epochs (be patient!)')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--grad_accum_steps', default=16, type=int,
                        help='Gradient accumulation steps')
    parser.add_argument('--num_workers', default=4, type=int)

    # Optimizer settings with P3 stabilization
    parser.add_argument('--lr_p3', default=1e-3, type=float,
                        help='Learning rate for P3 layer (higher for random init)')
    parser.add_argument('--lr_rest', default=1e-4, type=float,
                        help='Learning rate for pretrained parts')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--optimizer', default='adamw', type=str,
                        choices=['adam', 'adamw'])

    # Learning rate schedule
    parser.add_argument('--warmup_epochs', default=10, type=int,
                        help='Number of warmup epochs')
    parser.add_argument('--min_lr_ratio', default=0.01, type=float,
                        help='Minimum LR as ratio of initial LR')

    # Gradual unfreezing for P3 stability
    parser.add_argument('--freeze_backbone_epochs', default=5, type=int,
                        help='Freeze backbone while training P3 initially')
    parser.add_argument('--p3_warmup_epochs', default=10, type=int,
                        help='Total epochs for P3 warmup')

    # Checkpointing and logging
    parser.add_argument('--output_dir', default='output/distillation',
                        help='Path to save outputs')
    parser.add_argument('--checkpoint_interval', default=10, type=int,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', default='', help='Resume from checkpoint')
    parser.add_argument('--eval_interval', default=5, type=int,
                        help='Evaluate every N epochs')

    # Device
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training')
    parser.add_argument('--seed', default=42, type=int)

    # Wandb
    parser.add_argument('--wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--project', default='rf-detr-distillation', type=str)
    parser.add_argument('--run_name', default=None, type=str)

    return parser


def build_model_configs(args):
    """Build student and teacher model configurations."""
    # Student config
    if args.student_model == 'nano':
        from rfdetr.config import RFDETRNanoConfig
        student_config = RFDETRNanoConfig()
    elif args.student_model == 'small':
        student_config = RFDETRSmallConfig()
    elif args.student_model == 'medium':
        from rfdetr.config import RFDETRMediumConfig
        student_config = RFDETRMediumConfig()
    else:
        raise ValueError(f"Unknown student model: {args.student_model}")

    # Teacher config
    if args.teacher_model == 'nano':
        from rfdetr.config import RFDETRNanoConfig
        teacher_config = RFDETRNanoConfig()
    elif args.teacher_model == 'small':
        teacher_config = RFDETRSmallConfig()
    elif args.teacher_model == 'medium':
        from rfdetr.config import RFDETRMediumConfig
        teacher_config = RFDETRMediumConfig()
    elif args.teacher_model == 'base':
        from rfdetr.config import RFDETRBaseConfig
        teacher_config = RFDETRBaseConfig()
    elif args.teacher_model == 'large':
        teacher_config = RFDETRLargeConfig()
    else:
        raise ValueError(f"Unknown teacher model: {args.teacher_model}")

    # Update with command line args
    student_config.pretrain_weights = args.student_weights
    teacher_config.pretrain_weights = args.teacher_weights
    student_config.device = args.device
    teacher_config.device = args.device

    return student_config, teacher_config


def train_one_epoch(
    distiller,
    data_loader,
    optimizer,
    scheduler,
    criterion,
    epoch,
    grad_accum_steps,
    gradual_unfreezer=None,
    device='cuda',
    print_freq=10
):
    """Train for one epoch with gradient accumulation."""
    import time

    distiller.student.train()
    distiller.teacher.eval()

    # Update gradual unfreezing if applicable
    if gradual_unfreezer is not None:
        gradual_unfreezer.step_epoch()
        print(f"Epoch {epoch}: Updated parameter freezing state")

    total_loss = 0.0
    total_distill_loss = 0.0
    total_task_loss = 0.0
    optimizer.zero_grad()

    start_time = time.time()
    num_batches = len(data_loader)

    for i, (samples, targets) in enumerate(data_loader):
        iter_start_time = time.time()

        # Move to device
        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in targets]

        # Distillation step (includes consistent mixup)
        losses = distiller.distill_step(
            samples,
            targets=targets,
            task_criterion=criterion
        )

        # Scale loss by accumulation steps
        loss = losses['loss_total'] / grad_accum_steps
        loss.backward()

        # Gradient accumulation
        if (i + 1) % grad_accum_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(distiller.student.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += losses['loss_total'].item()

        # Track individual loss components
        if 'loss_distill_logits' in losses:
            total_distill_loss += losses['loss_distill_logits'].item()
        task_loss_sum = sum(v.item() for k, v in losses.items() if k.startswith('task_'))
        total_task_loss += task_loss_sum

        # Logging
        if (i + 1) % print_freq == 0 or (i + 1) == num_batches:
            avg_loss = total_loss / (i + 1)
            avg_distill = total_distill_loss / (i + 1) if total_distill_loss > 0 else 0.0
            avg_task = total_task_loss / (i + 1) if total_task_loss > 0 else 0.0
            lrs = scheduler.get_last_lr()

            # Calculate time estimates
            iter_time = time.time() - iter_start_time
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (num_batches - (i + 1))

            print(f"[Epoch {epoch}] Step [{i + 1}/{num_batches}] "
                  f"Loss: {avg_loss:.4f} (Distill: {avg_distill:.4f}, Task: {avg_task:.4f}) | "
                  f"LR: {lrs[0]:.2e} | "
                  f"Time: {iter_time:.2f}s/iter | ETA: {eta/60:.1f}min")

    return total_loss / len(data_loader)


@torch.no_grad()
def evaluate(distiller, data_loader, criterion, postprocessors, device='cuda'):
    """Evaluate the student model."""
    import time
    from rfdetr.f1_metric import collect_predictions_and_gts, calculate_f1

    distiller.student.eval()

    total_loss = 0.0
    num_batches = len(data_loader)
    start_time = time.time()

    all_results = []
    all_targets = []

    for i, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in targets]

        # Forward through student only
        outputs = distiller.forward_student(samples)

        # Compute task loss
        losses = criterion(outputs, targets)
        total_loss += sum(losses.values()).item()

        # Post-process outputs for F1 calculation
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        all_results.extend(results)
        all_targets.extend(targets)

        # Progress update
        if (i + 1) % 10 == 0 or (i + 1) == num_batches:
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (num_batches - (i + 1))
            print(f"[Validation] Step [{i + 1}/{num_batches}] | ETA: {eta:.1f}s")

    # Calculate F1 scores
    predictions, ground_truths = collect_predictions_and_gts(all_results, all_targets)
    f1_metrics = calculate_f1(predictions, ground_truths, center_threshold=50.0, score_threshold=0.5)

    avg_loss = total_loss / len(data_loader)

    return avg_loss, f1_metrics


def save_checkpoint(epoch, distiller, optimizer, scheduler, args, filename):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'student_state_dict': distiller.student.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': {
            'current_step': scheduler.current_step,
            'warmup_steps': scheduler.warmup_steps,
            'total_steps': scheduler.total_steps,
        },
        'args': args,
    }

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(checkpoint, filename)
    print(f"Saved checkpoint to {filename}")


def main(args):
    """Main training function."""
    # Setup
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    # Build model configs
    student_config, teacher_config = build_model_configs(args)

    # Create distillation setup
    print("=" * 80)
    print("Creating distillation setup...")
    print("=" * 80)

    setup = create_distillation_setup(
        student_config=student_config.dict(),
        teacher_config=teacher_config.dict(),
        pretrained_student_path=args.student_weights if os.path.exists(args.student_weights) else None,
        pretrained_teacher_path=args.teacher_weights if os.path.exists(args.teacher_weights) else None,
        add_p3_to_student=args.add_p3,
        device=args.device,
        temperature=args.temperature,
        alpha=args.alpha,
        use_mixup=args.use_mixup,
        mixup_p=args.mixup_p,
        feature_distillation=args.feature_distillation
    )

    distiller = setup['distiller']
    student = setup['student']
    p3_params = setup['p3_params']

    print(f"\nStudent model: {args.student_model}")
    print(f"Teacher model: {args.teacher_model}")
    print(f"P3 layer added: {args.add_p3}")
    print(f"P3 parameters found: {len(p3_params)}")

    # Build datasets
    print("\n" + "=" * 80)
    print("Building datasets...")
    print("=" * 80)

    # Convert student config to args format for dataset building
    class DatasetArgs:
        def __init__(self, config_dict, cmd_args):
            # Copy all args from student config
            for k, v in config_dict.items():
                setattr(self, k, v)
            # Override with command line args
            self.dataset_file = cmd_args.dataset_file
            self.coco_path = cmd_args.coco_path
            self.dataset_dir = cmd_args.dataset_dir
            self.num_workers = cmd_args.num_workers

            # Add dataset-specific attributes with defaults
            self.multi_scale = True
            self.expanded_scales = True
            self.do_random_resize_via_padding = False
            self.square_resize_div_64 = True

            # Add matcher and criterion attributes with defaults
            self.set_cost_class = 2
            self.set_cost_bbox = 5
            self.set_cost_giou = 2
            self.bbox_loss_coef = 5
            self.giou_loss_coef = 2
            self.focal_alpha = 0.25
            self.aux_loss = True
            self.use_varifocal_loss = False
            self.use_position_supervised_loss = False
            self.mask_ce_loss_coef = 1.0
            self.mask_dice_loss_coef = 1.0
            self.mask_point_sample_ratio = 0.0

    dataset_args = DatasetArgs(student_config.model_dump(), args)

    dataset_train = build_dataset(image_set='train', args=dataset_args, resolution=student_config.resolution)
    dataset_val = build_dataset(image_set='val', args=dataset_args, resolution=student_config.resolution)

    print(f"Dataset: {args.dataset_file}")
    print(f"Dataset path: {args.dataset_dir}")
    print(f"Train samples: {len(dataset_train)}")
    print(f"Val samples: {len(dataset_val)}")

    # Build data loaders
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    effective_batch_size = args.batch_size * args.grad_accum_steps

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, effective_batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers
    )

    data_loader_val = DataLoader(
        dataset_val,
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers
    )

    print(f"Train batches: {len(data_loader_train)}")
    print(f"Val batches: {len(data_loader_val)}")

    # Setup optimizer with separate LRs for P3
    print("\n" + "=" * 80)
    print("Setting up optimizer...")
    print("=" * 80)

    optimizer = StableOptimizer(
        model=distiller.student,
        p3_params=p3_params,
        lr_p3=args.lr_p3,
        lr_rest=args.lr_rest,
        weight_decay=args.weight_decay,
        optimizer_type=args.optimizer
    )

    print(f"Optimizer: {args.optimizer}")
    print(f"LR for P3 layer: {args.lr_p3}")
    print(f"LR for pretrained parts: {args.lr_rest}")

    # Build criterion
    print("\n" + "=" * 80)
    print("Building criterion...")
    print("=" * 80)

    criterion, postprocessors = build_criterion_and_postprocessors(dataset_args)
    criterion.to(device)

    print("Criterion built successfully")

    # Setup scheduler
    num_batches_per_epoch = len(data_loader_train)
    total_steps = args.epochs * num_batches_per_epoch
    warmup_steps = args.warmup_epochs * num_batches_per_epoch

    scheduler = CosineScheduleWithWarmup(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=args.min_lr_ratio
    )

    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    # Setup gradual unfreezing for P3 stability
    gradual_unfreezer = None
    if args.add_p3 and len(p3_params) > 0:
        print("\n" + "=" * 80)
        print("Setting up gradual unfreezing for P3 stability...")
        print("=" * 80)

        gradual_unfreezer = GradualUnfreezing(
            model=distiller.student,
            p3_params=p3_params,
            warmup_epochs=args.p3_warmup_epochs,
            freeze_backbone_epochs=args.freeze_backbone_epochs
        )

        print(f"Will freeze backbone for {args.freeze_backbone_epochs} epochs")
        print(f"Total P3 warmup: {args.p3_warmup_epochs} epochs")

    # Print training plan
    print("\n" + "=" * 80)
    print("TRAINING PLAN - Patient and Consistent Teaching")
    print("=" * 80)
    print(f"Total epochs: {args.epochs} (be patient!)")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.grad_accum_steps}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum_steps}")
    print(f"Temperature: {args.temperature}")
    print(f"Alpha (distillation weight): {args.alpha}")
    print(f"Mixup enabled: {args.use_mixup}")
    print(f"Feature distillation: {args.feature_distillation}")

    print("\n" + "=" * 80)
    print("Key Principles from the Paper:")
    print("=" * 80)
    print("1. CONSISTENT: Teacher and student see identical augmented inputs")
    print("2. PATIENT: Training for many epochs without overfitting")
    print("3. AGGRESSIVE MIXUP: Function matching via data augmentation")
    print("4. STABLE P3 INIT: Gradual unfreezing for new random layers")
    print("=" * 80)

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save configuration
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"\nSaved configuration to {config_path}")

    # Initialize wandb if requested
    if args.wandb and WANDB_AVAILABLE:
        run_name = args.run_name or f"distill_{args.student_model}_from_{args.teacher_model}"
        wandb.init(
            project=args.project,
            name=run_name,
            config=vars(args)
        )
        print(f"Initialized Weights & Biases: {args.project}/{run_name}")

    print("\n" + "=" * 80)
    print("Starting training!")
    print("=" * 80)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print('=' * 80)

        # Train
        train_loss = train_one_epoch(
            distiller=distiller,
            data_loader=data_loader_train,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            epoch=epoch + 1,
            grad_accum_steps=args.grad_accum_steps,
            gradual_unfreezer=gradual_unfreezer,
            device=device,
            print_freq=10
        )

        print(f"Train loss: {train_loss:.4f}")

        # Log to wandb
        if args.wandb and WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'lr_p3': scheduler.get_last_lr()[0] if len(p3_params) > 0 else 0.0,
                'lr_rest': scheduler.get_last_lr()[1] if len(scheduler.get_last_lr()) > 1 else scheduler.get_last_lr()[0],
            })

        # Evaluate
        if (epoch + 1) % args.eval_interval == 0 or (epoch + 1) == args.epochs:
            val_loss, f1_metrics = evaluate(distiller, data_loader_val, criterion, postprocessors, device)
            print(f"Val loss: {val_loss:.4f}")
            print(f"F1: {f1_metrics['f1']:.4f} | Precision: {f1_metrics['precision']:.4f} | Recall: {f1_metrics['recall']:.4f}")
            print(f"TP: {f1_metrics['tp']} | FP: {f1_metrics['fp']} | FN: {f1_metrics['fn']}")

            if args.wandb and WANDB_AVAILABLE:
                wandb.log({
                    'val_loss': val_loss,
                    'val_f1': f1_metrics['f1'],
                    'val_precision': f1_metrics['precision'],
                    'val_recall': f1_metrics['recall']
                })

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    epoch=epoch,
                    distiller=distiller,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                    filename=os.path.join(args.output_dir, 'checkpoint_best.pth')
                )
                print(f"New best model! Val loss: {val_loss:.4f} | F1: {f1_metrics['f1']:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            save_checkpoint(
                epoch=epoch,
                distiller=distiller,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                filename=os.path.join(args.output_dir, f'checkpoint_{epoch + 1}.pth')
            )

        # Save latest checkpoint
        save_checkpoint(
            epoch=epoch,
            distiller=distiller,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            filename=os.path.join(args.output_dir, 'checkpoint_latest.pth')
        )

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.4f}")

    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RF-DETR Distillation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
