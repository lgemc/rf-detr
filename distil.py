"""
Knowledge Distillation for RF-DETR
Based on "Knowledge distillation: A good teacher is patient and consistent" (Beyer et al., 2022)

This module implements function matching distillation with:
1. Consistent teaching: Teacher and student receive identical augmented inputs
2. Patience: Long training schedules with no overfitting
3. P3 layer integration: Adds P3 layer to small model and stabilizes training
"""

import copy
import math
from typing import Optional, Dict, List, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class P3LayerInjector:
    """
    Adds a P3 layer to a model that doesn't have one by default.
    Handles careful initialization to avoid destabilizing training.
    """

    @staticmethod
    def add_p3_layer(model, hidden_dim: int = 256):
        """
        Adds P3 layer to the student model's projector_scale configuration.

        Args:
            model: The RF-DETR model to modify
            hidden_dim: Hidden dimension for the P3 projector

        Returns:
            The modified model with P3 layer added
        """
        # Check if P3 already exists
        if hasattr(model, 'args') and 'P3' in model.args.projector_scale:
            print("P3 layer already exists in model")
            return model

        # Add P3 to projector_scale configuration
        if hasattr(model, 'args'):
            current_scales = list(model.args.projector_scale)
            if 'P3' not in current_scales:
                # Add P3 at the beginning (finest scale)
                model.args.projector_scale = ['P3'] + current_scales
                print(f"Updated projector_scale to: {model.args.projector_scale}")

        # The backbone will need to be updated to include P3 projector
        # This requires modifying the backbone's scale_embed module
        if hasattr(model, 'model') and hasattr(model.model, 'backbone'):
            backbone = model.model.backbone
            if hasattr(backbone, 'scale_embed'):
                # Create new P3 projector
                p3_projector = nn.Sequential(
                    nn.Conv2d(
                        backbone.hidden_dim,
                        hidden_dim,
                        kernel_size=1,
                        bias=False
                    ),
                    nn.GroupNorm(32, hidden_dim)
                )

                # Initialize with small weights for stability
                for module in p3_projector.modules():
                    if isinstance(module, nn.Conv2d):
                        # Xavier initialization with small scale
                        nn.init.xavier_uniform_(module.weight, gain=0.1)
                    elif isinstance(module, nn.GroupNorm):
                        nn.init.constant_(module.weight, 1.0)
                        nn.init.constant_(module.bias, 0.0)

                # Add to scale_embed as a new module
                if isinstance(backbone.scale_embed, nn.ModuleList):
                    backbone.scale_embed.insert(0, p3_projector)
                elif isinstance(backbone.scale_embed, nn.ModuleDict):
                    backbone.scale_embed['P3'] = p3_projector
                else:
                    print(f"Warning: Unknown scale_embed type: {type(backbone.scale_embed)}")

        return model

    @staticmethod
    def get_p3_parameters(model):
        """
        Extract P3-specific parameters for separate learning rate scheduling.

        Args:
            model: The model with P3 layer

        Returns:
            List of P3 parameters
        """
        p3_params = []

        if hasattr(model, 'model') and hasattr(model.model, 'backbone'):
            backbone = model.model.backbone
            if hasattr(backbone, 'scale_embed'):
                if isinstance(backbone.scale_embed, nn.ModuleList) and len(backbone.scale_embed) > 0:
                    # First element should be P3
                    p3_params.extend(backbone.scale_embed[0].parameters())
                elif isinstance(backbone.scale_embed, nn.ModuleDict) and 'P3' in backbone.scale_embed:
                    p3_params.extend(backbone.scale_embed['P3'].parameters())

        return p3_params


class DistillationLoss(nn.Module):
    """
    KL-divergence based distillation loss as described in the paper.
    Implements temperature-scaled soft target matching.
    """

    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: Temperature parameter for softening predictions
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between teacher and student predictions.

        Args:
            student_logits: Student model output logits [B, N, C]
            teacher_logits: Teacher model output logits [B, N, C]

        Returns:
            KL divergence loss
        """
        # Apply temperature scaling
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)

        # KL divergence: KL(teacher || student)
        kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')

        # Scale by temperature^2 as per convention
        kl_loss = kl_loss * (self.temperature ** 2)

        return kl_loss


class FeatureDistillationLoss(nn.Module):
    """
    Optional feature-level distillation for intermediate representations.
    """

    def __init__(self, distance_type: Literal["l2", "cosine"] = "l2"):
        super().__init__()
        self.distance_type = distance_type

    def forward(self, student_features: torch.Tensor, teacher_features: torch.Tensor) -> torch.Tensor:
        """
        Compute distance between student and teacher features.

        Args:
            student_features: Student intermediate features
            teacher_features: Teacher intermediate features

        Returns:
            Feature distillation loss
        """
        if self.distance_type == "l2":
            return F.mse_loss(student_features, teacher_features)
        elif self.distance_type == "cosine":
            # Cosine distance: 1 - cosine_similarity
            student_norm = F.normalize(student_features, p=2, dim=-1)
            teacher_norm = F.normalize(teacher_features, p=2, dim=-1)
            return 1.0 - (student_norm * teacher_norm).sum(dim=-1).mean()
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")


class MixupAugmentation:
    """
    Aggressive mixup augmentation for function matching.
    Samples mixing coefficients uniformly from [0, 1].
    """

    def __init__(self, p: float = 0.0, n: int = 2):
        """
        Args:
            p: Probability of applying mixup
            n: Number of images to mix (default 2)
        """
        self.p = p
        self.n = n

    def __call__(self, images, targets: Optional[Dict] = None):
        """
        Apply mixup augmentation to a batch of images.

        Args:
            images: Batch of images (either Tensor [B, C, H, W] or NestedTensor)
            targets: Optional targets to mix

        Returns:
            Mixed images and optionally mixed targets
        """
        if torch.rand(1).item() > self.p:
            return images, targets

        # Handle NestedTensor
        from rfdetr.util.misc import NestedTensor
        is_nested = isinstance(images, NestedTensor)
        if is_nested:
            image_tensor = images.tensors
            mask = images.mask
        else:
            image_tensor = images
            mask = None

        batch_size = image_tensor.size(0)

        # Sample mixing coefficients uniformly from [0, 1]
        lam = torch.rand(batch_size, 1, 1, 1, device=image_tensor.device)

        # Random permutation for mixing
        indices = torch.randperm(batch_size, device=image_tensor.device)

        # Mix images
        mixed_image_tensor = lam * image_tensor + (1 - lam) * image_tensor[indices]

        # Mix masks if present
        if mask is not None:
            # For masks, use logical OR to combine padding masks
            mixed_mask = mask | mask[indices]
        else:
            mixed_mask = None

        # Reconstruct NestedTensor if input was NestedTensor
        if is_nested:
            mixed_images = NestedTensor(mixed_image_tensor, mixed_mask)
        else:
            mixed_images = mixed_image_tensor

        # Mix targets if provided
        if targets is not None:
            mixed_targets = []
            for i in range(batch_size):
                # Combine targets from both images
                target_a = targets[i]
                target_b = targets[indices[i]]

                # Simple concatenation of boxes and labels
                mixed_target = {
                    'boxes': torch.cat([target_a['boxes'], target_b['boxes']], dim=0),
                    'labels': torch.cat([target_a['labels'], target_b['labels']], dim=0),
                }

                if 'masks' in target_a:
                    mixed_target['masks'] = torch.cat([target_a['masks'], target_b['masks']], dim=0)

                mixed_targets.append(mixed_target)

            return mixed_images, mixed_targets

        return mixed_images, None


class GradualUnfreezing:
    """
    Gradual unfreezing strategy to stabilize training when P3 is randomly initialized.
    """

    def __init__(
        self,
        model,
        p3_params: List[torch.nn.Parameter],
        warmup_epochs: int = 10,
        freeze_backbone_epochs: int = 5
    ):
        """
        Args:
            model: The student model
            p3_params: P3 layer parameters
            warmup_epochs: Number of epochs to warmup P3 layer
            freeze_backbone_epochs: Number of epochs to freeze backbone
        """
        self.model = model
        self.p3_params = set(p3_params)
        self.warmup_epochs = warmup_epochs
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.current_epoch = 0

    def step_epoch(self):
        """Update freezing state based on current epoch."""
        self.current_epoch += 1

        if self.current_epoch <= self.freeze_backbone_epochs:
            # Freeze everything except P3
            self._freeze_all_except_p3()
        elif self.current_epoch <= self.warmup_epochs:
            # Gradually unfreeze
            self._unfreeze_all()
        else:
            # Everything unfrozen
            self._unfreeze_all()

    def _freeze_all_except_p3(self):
        """Freeze all parameters except P3 layer."""
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.p3_params:
            param.requires_grad = True

    def _unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = True


class StableOptimizer:
    """
    Optimizer wrapper with separate learning rates for P3 and rest of model.
    Implements careful learning rate scheduling to avoid destabilization.
    """

    def __init__(
        self,
        model,
        p3_params: List[torch.nn.Parameter],
        lr_p3: float = 1e-3,
        lr_rest: float = 1e-4,
        weight_decay: float = 1e-4,
        optimizer_type: Literal["adam", "adamw"] = "adamw"
    ):
        """
        Args:
            model: The model to optimize
            p3_params: P3 layer parameters
            lr_p3: Learning rate for P3 layer (higher for random init)
            lr_rest: Learning rate for pretrained parts
            weight_decay: Weight decay coefficient
            optimizer_type: Type of optimizer
        """
        # Separate parameter groups
        p3_param_ids = {id(p) for p in p3_params}

        param_groups = [
            {
                'params': [p for p in model.parameters() if id(p) in p3_param_ids],
                'lr': lr_p3,
                'name': 'p3_layer'
            },
            {
                'params': [p for p in model.parameters() if id(p) not in p3_param_ids],
                'lr': lr_rest,
                'name': 'pretrained'
            }
        ]

        if optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(param_groups, weight_decay=weight_decay)
        elif optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    def step(self):
        """Perform optimization step."""
        self.optimizer.step()

    def zero_grad(self):
        """Zero out gradients."""
        self.optimizer.zero_grad()

    def state_dict(self):
        """Get optimizer state dict."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Load optimizer state dict."""
        self.optimizer.load_state_dict(state_dict)


class CosineScheduleWithWarmup:
    """
    Cosine learning rate schedule with linear warmup.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0
    ):
        """
        Args:
            optimizer: The optimizer to schedule
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            min_lr_ratio: Minimum LR as ratio of initial LR
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group['lr'] for group in optimizer.optimizer.param_groups]
        self.current_step = 0

    def step(self):
        """Update learning rates."""
        self.current_step += 1

        for param_group, base_lr in zip(self.optimizer.optimizer.param_groups, self.base_lrs):
            if self.current_step < self.warmup_steps:
                # Linear warmup
                lr = base_lr * (self.current_step / self.warmup_steps)
            else:
                # Cosine decay
                progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                lr = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (
                    1 + math.cos(math.pi * progress)
                )
                lr = base_lr * lr

            param_group['lr'] = lr

    def get_last_lr(self):
        """Get current learning rates."""
        return [group['lr'] for group in self.optimizer.optimizer.param_groups]


class ConsistentTeacherDistiller:
    """
    Main distillation coordinator implementing the "patient and consistent teacher" approach.
    """

    def __init__(
        self,
        student_model,
        teacher_model,
        temperature: float = 1.0,
        alpha: float = 0.9,
        feature_distillation: bool = False,
        use_mixup: bool = False,
        mixup_p: float = 0.0,
        device: str = "cuda"
    ):
        """
        Args:
            student_model: The student model (small)
            teacher_model: The teacher model (large)
            temperature: Temperature for KL divergence
            alpha: Weight for distillation loss (1-alpha for task loss)
            feature_distillation: Whether to use feature-level distillation
            use_mixup: Whether to use mixup augmentation
            mixup_p: Probability of mixup
            device: Device to use
        """
        self.student = student_model
        self.teacher = teacher_model
        self.device = device

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        # Loss functions
        self.distillation_loss = DistillationLoss(temperature=temperature)
        self.feature_loss = FeatureDistillationLoss() if feature_distillation else None
        self.alpha = alpha

        # Mixup for function matching
        self.mixup = MixupAugmentation(p=mixup_p) if use_mixup else None

        # Move to device
        self.student.to(device)
        self.teacher.to(device)

    def compute_loss(
        self,
        student_outputs: Dict,
        teacher_outputs: Dict,
        targets: Optional[List[Dict]] = None,
        task_criterion = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined distillation and task losses.

        Args:
            student_outputs: Student model outputs
            teacher_outputs: Teacher model outputs (detached)
            targets: Ground truth targets
            task_criterion: Task-specific criterion (e.g., detection loss)

        Returns:
            Dictionary of losses
        """
        losses = {}

        # Distillation loss on logits
        if 'pred_logits' in student_outputs and 'pred_logits' in teacher_outputs:
            student_logits = student_outputs['pred_logits']
            teacher_logits = teacher_outputs['pred_logits'].detach()

            # Check if dimensions match
            if student_logits.shape == teacher_logits.shape:
                distill_loss = self.distillation_loss(student_logits, teacher_logits)
                losses['loss_distill_logits'] = distill_loss
            elif student_logits.shape[0] == teacher_logits.shape[0] and student_logits.shape[2] == teacher_logits.shape[2]:
                # Same batch size and num_classes, but different number of queries
                # Align by selecting top-K student queries based on max class probabilities
                B, N_student, C = student_logits.shape
                N_teacher = teacher_logits.shape[1]

                if N_student > N_teacher:
                    # Take top-K most confident queries from student
                    max_probs = student_logits.softmax(dim=-1).max(dim=-1)[0]  # [B, N_student]
                    top_k_indices = torch.topk(max_probs, k=N_teacher, dim=1)[1]  # [B, N_teacher]

                    # Gather the top-K queries for each batch
                    top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, C)  # [B, N_teacher, C]
                    student_logits_aligned = torch.gather(student_logits, dim=1, index=top_k_indices_expanded)

                    distill_loss = self.distillation_loss(student_logits_aligned, teacher_logits)
                    losses['loss_distill_logits'] = distill_loss
                elif N_teacher > N_student:
                    # Take top-K most confident queries from teacher
                    max_probs = teacher_logits.softmax(dim=-1).max(dim=-1)[0]  # [B, N_teacher]
                    top_k_indices = torch.topk(max_probs, k=N_student, dim=1)[1]  # [B, N_student]

                    # Gather the top-K queries for each batch
                    top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, C)  # [B, N_student, C]
                    teacher_logits_aligned = torch.gather(teacher_logits, dim=1, index=top_k_indices_expanded)

                    distill_loss = self.distillation_loss(student_logits, teacher_logits_aligned)
                    losses['loss_distill_logits'] = distill_loss
            else:
                print(f"Warning: Skipping logit distillation due to incompatible shapes: "
                      f"student {student_logits.shape} vs teacher {teacher_logits.shape}")

        # Optional: Feature distillation
        if self.feature_loss and 'features' in student_outputs and 'features' in teacher_outputs:
            feature_loss = self.feature_loss(
                student_outputs['features'],
                teacher_outputs['features'].detach()
            )
            losses['loss_distill_features'] = feature_loss

        # Task loss (if targets provided)
        if targets is not None and task_criterion is not None:
            task_losses = task_criterion(student_outputs, targets)
            for k, v in task_losses.items():
                losses[f'task_{k}'] = v * (1 - self.alpha)

        # Total loss
        total_loss = sum(losses.values())
        losses['loss_total'] = total_loss

        return losses

    @torch.no_grad()
    def forward_teacher(self, images: torch.Tensor):
        """
        Forward pass through teacher (no gradients).

        Args:
            images: Input images

        Returns:
            Teacher outputs
        """
        return self.teacher(images)

    def forward_student(self, images: torch.Tensor):
        """
        Forward pass through student.

        Args:
            images: Input images

        Returns:
            Student outputs
        """
        return self.student(images)

    def distill_step(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict]] = None,
        task_criterion = None
    ) -> Dict[str, torch.Tensor]:
        """
        Single distillation step with consistent inputs to teacher and student.

        Args:
            images: Input images (already augmented consistently)
            targets: Optional ground truth targets
            task_criterion: Optional task-specific criterion

        Returns:
            Dictionary of losses
        """
        # Apply mixup if enabled (consistent for both teacher and student)
        if self.mixup is not None:
            images, targets = self.mixup(images, targets)

        # Forward pass through teacher (no gradients)
        with torch.no_grad():
            teacher_outputs = self.forward_teacher(images)

        # Forward pass through student
        student_outputs = self.forward_student(images)

        # Compute losses
        losses = self.compute_loss(
            student_outputs,
            teacher_outputs,
            targets,
            task_criterion
        )

        return losses


def create_distillation_setup(
    student_config: Dict,
    teacher_config: Dict,
    pretrained_student_path: Optional[str] = None,
    pretrained_teacher_path: Optional[str] = None,
    add_p3_to_student: bool = True,
    device: str = "cuda",
    **distill_kwargs
) -> Dict:
    """
    Complete setup for knowledge distillation.

    Args:
        student_config: Student model configuration
        teacher_config: Teacher model configuration
        pretrained_student_path: Path to pretrained student weights
        pretrained_teacher_path: Path to pretrained teacher weights
        add_p3_to_student: Whether to add P3 layer to student
        device: Device to use
        **distill_kwargs: Additional kwargs for ConsistentTeacherDistiller

    Returns:
        Dictionary containing distiller, optimizer, scheduler, etc.
    """
    from rfdetr.main import Model

    # Build student
    print("Building student model...")
    student = Model(**student_config)

    # Load pretrained student weights
    if pretrained_student_path:
        print(f"Loading pretrained student weights from {pretrained_student_path}")
        checkpoint = torch.load(pretrained_student_path, map_location='cpu', weights_only=False)
        if 'model' in checkpoint:
            student.model.load_state_dict(checkpoint['model'], strict=False)
        else:
            student.model.load_state_dict(checkpoint, strict=False)

    # Add P3 layer to student if requested
    p3_params = []
    if add_p3_to_student:
        print("Adding P3 layer to student model...")
        student = P3LayerInjector.add_p3_layer(student, hidden_dim=student_config.get('hidden_dim', 256))
        p3_params = P3LayerInjector.get_p3_parameters(student)
        print(f"Found {len(p3_params)} P3 parameters")

    # Build teacher
    print("Building teacher model...")
    teacher = Model(**teacher_config)

    # Load pretrained teacher weights
    if pretrained_teacher_path:
        print(f"Loading pretrained teacher weights from {pretrained_teacher_path}")
        checkpoint = torch.load(pretrained_teacher_path, map_location='cpu', weights_only=False)
        if 'model' in checkpoint:
            teacher.model.load_state_dict(checkpoint['model'], strict=False)
        else:
            teacher.model.load_state_dict(checkpoint, strict=False)

    # Match student num_classes to teacher num_classes
    # class_embed can be either a Linear layer or a ModuleList
    if isinstance(teacher.model.class_embed, nn.ModuleList):
        teacher_num_classes = teacher.model.class_embed[-1].weight.shape[0]
    else:
        teacher_num_classes = teacher.model.class_embed.weight.shape[0]

    if isinstance(student.model.class_embed, nn.ModuleList):
        student_num_classes = student.model.class_embed[-1].weight.shape[0]
    else:
        student_num_classes = student.model.class_embed.weight.shape[0]

    if teacher_num_classes != student_num_classes:
        print(f"Adjusting student num_classes from {student_num_classes} to {teacher_num_classes}")

        # Reinitialize the class embedding layers to match teacher
        if isinstance(student.model.class_embed, nn.ModuleList):
            for i, class_embed in enumerate(student.model.class_embed):
                in_features = class_embed.weight.shape[1]
                student.model.class_embed[i] = nn.Linear(in_features, teacher_num_classes)
                # Initialize with small weights
                nn.init.normal_(student.model.class_embed[i].weight, std=0.01)
                nn.init.constant_(student.model.class_embed[i].bias, 0)
        else:
            in_features = student.model.class_embed.weight.shape[1]
            student.model.class_embed = nn.Linear(in_features, teacher_num_classes)
            # Initialize with small weights
            nn.init.normal_(student.model.class_embed.weight, std=0.01)
            nn.init.constant_(student.model.class_embed.bias, 0)

        # Update student config
        student_config['num_classes'] = teacher_num_classes
        print(f"Student model now has {teacher_num_classes} classes")

    # Create distiller
    print("Creating distiller...")
    distiller = ConsistentTeacherDistiller(
        student_model=student.model,
        teacher_model=teacher.model,
        device=device,
        **distill_kwargs
    )

    return {
        'distiller': distiller,
        'student': student,
        'teacher': teacher,
        'p3_params': p3_params
    }
