# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable
import random

import torch
import torch.nn.functional as F

import rfdetr.util.misc as utils
from rfdetr.datasets.coco_eval import CocoEvaluator
from rfdetr.datasets.coco import compute_multi_scale_scales
from rfdetr.util.metrics import evaluate_detection_f1

try:
    from torch.amp import autocast, GradScaler
    DEPRECATED_AMP = False
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    DEPRECATED_AMP = True
from typing import DefaultDict, List, Callable
from rfdetr.util.misc import NestedTensor
import numpy as np

def get_autocast_args(args):
    if DEPRECATED_AMP:
        return {'enabled': args.amp, 'dtype': torch.bfloat16}
    else:
        return {'device_type': 'cuda', 'enabled': args.amp, 'dtype': torch.bfloat16}


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    batch_size: int,
    max_norm: float = 0,
    ema_m: torch.nn.Module = None,
    schedules: dict = {},
    num_training_steps_per_epoch=None,
    vit_encoder_num_layers=None,
    args=None,
    callbacks: DefaultDict[str, List[Callable]] = None,
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10
    start_steps = epoch * num_training_steps_per_epoch

    print("Grad accum steps: ", args.grad_accum_steps)
    print("Total batch size: ", batch_size * utils.get_world_size())

    # Add gradient scaler for AMP
    if DEPRECATED_AMP:
        scaler = GradScaler(enabled=args.amp)
    else:
        scaler = GradScaler('cuda', enabled=args.amp)

    optimizer.zero_grad()
    assert batch_size % args.grad_accum_steps == 0
    sub_batch_size = batch_size // args.grad_accum_steps
    print("LENGTH OF DATA LOADER:", len(data_loader))
    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        it = start_steps + data_iter_step
        callback_dict = {
            "step": it,
            "model": model,
            "epoch": epoch,
        }
        for callback in callbacks["on_train_batch_start"]:
            callback(callback_dict)
        if "dp" in schedules:
            if args.distributed:
                model.module.update_drop_path(
                    schedules["dp"][it], vit_encoder_num_layers
                )
            else:
                model.update_drop_path(schedules["dp"][it], vit_encoder_num_layers)
        if "do" in schedules:
            if args.distributed:
                model.module.update_dropout(schedules["do"][it])
            else:
                model.update_dropout(schedules["do"][it])

        if args.multi_scale and not args.do_random_resize_via_padding:
            scales = compute_multi_scale_scales(args.resolution, args.expanded_scales, args.patch_size, args.num_windows)
            random.seed(it)
            scale = random.choice(scales)
            with torch.inference_mode():
                samples.tensors = F.interpolate(samples.tensors, size=scale, mode='bilinear', align_corners=False)
                samples.mask = F.interpolate(samples.mask.unsqueeze(1).float(), size=scale, mode='nearest').squeeze(1).bool()

        for i in range(args.grad_accum_steps):
            start_idx = i * sub_batch_size
            final_idx = start_idx + sub_batch_size
            new_samples_tensors = samples.tensors[start_idx:final_idx]
            new_samples = NestedTensor(new_samples_tensors, samples.mask[start_idx:final_idx])
            new_samples = new_samples.to(device)
            new_targets = [{k: v.to(device) for k, v in t.items()} for t in targets[start_idx:final_idx]]

            with autocast(**get_autocast_args(args)):
                outputs = model(new_samples, new_targets)
                loss_dict = criterion(outputs, new_targets)
                weight_dict = criterion.weight_dict
                losses = sum(
                    (1 / args.grad_accum_steps) * loss_dict[k] * weight_dict[k]
                    for k in loss_dict.keys()
                    if k in weight_dict
                )


            scaler.scale(losses).backward()

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k:  v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print(loss_dict_reduced)
            raise ValueError("Loss is {}, stopping training".format(loss_value))

        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad()
        if ema_m is not None:
            if epoch >= 0:
                ema_m.update(model)
        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def coco_extended_metrics(coco_eval):
    """
    Safe version: ignores the –1 sentinel entries so precision/F1 never explode.
    """

    iou_thrs, rec_thrs = coco_eval.params.iouThrs, coco_eval.params.recThrs
    iou50_idx, area_idx, maxdet_idx = (
        int(np.argwhere(np.isclose(iou_thrs, 0.50))), 0, 2)

    P = coco_eval.eval["precision"]
    S = coco_eval.eval["scores"]

    prec_raw = P[iou50_idx, :, :, area_idx, maxdet_idx]

    prec = prec_raw.copy().astype(float)
    prec[prec < 0] = np.nan

    f1_cls   = 2 * prec * rec_thrs[:, None] / (prec + rec_thrs[:, None])
    f1_macro = np.nanmean(f1_cls, axis=1)

    best_j   = int(f1_macro.argmax())

    macro_precision = float(np.nanmean(prec[best_j]))
    macro_recall    = float(rec_thrs[best_j])
    macro_f1        = float(f1_macro[best_j])

    score_vec = S[iou50_idx, best_j, :, area_idx, maxdet_idx].astype(float)
    score_vec[prec_raw[best_j] < 0] = np.nan
    score_thr = float(np.nanmean(score_vec))

    map_50_95, map_50 = float(coco_eval.stats[0]), float(coco_eval.stats[1])

    per_class = []
    cat_ids = coco_eval.params.catIds
    cat_id_to_name = {c["id"]: c["name"] for c in coco_eval.cocoGt.loadCats(cat_ids)}
    for k, cid in enumerate(cat_ids):
        p_slice = P[:, :, k, area_idx, maxdet_idx]
        valid   = p_slice > -1
        ap_50_95 = float(p_slice[valid].mean()) if valid.any() else float("nan")
        ap_50    = float(p_slice[iou50_idx][p_slice[iou50_idx] > -1].mean()) if (p_slice[iou50_idx] > -1).any() else float("nan")

        pc = float(prec[best_j, k]) if prec_raw[best_j, k] > -1 else float("nan")
        rc = macro_recall

        #Doing to this to filter out dataset class
        if np.isnan(ap_50_95) or np.isnan(ap_50) or np.isnan(pc) or np.isnan(rc):
            continue

        per_class.append({
            "class"      : cat_id_to_name[int(cid)],
            "map@50:95"  : ap_50_95,
            "map@50"     : ap_50,
            "precision"  : pc,
            "recall"     : rc,
        })

    per_class.append({
        "class"     : "all",
        "map@50:95" : map_50_95,
        "map@50"    : map_50,
        "precision" : macro_precision,
        "recall"    : macro_recall,
    })

    return {
        "class_map": per_class,
        "map"      : map_50,
        "precision": macro_precision,
        "recall"   : macro_recall
    }

def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, args=None):
    model.eval()
    if args.fp16_eval:
        model.half()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Test:"

    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    # For center-based F1 metric
    all_predictions = []
    all_ground_truths = []

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if args.fp16_eval:
            samples.tensors = samples.tensors.half()

        # Add autocast for evaluation
        with autocast(**get_autocast_args(args)):
            outputs = model(samples)

        if args.fp16_eval:
            for key in outputs.keys():
                if key == "enc_outputs":
                    for sub_key in outputs[key].keys():
                        outputs[key][sub_key] = outputs[key][sub_key].float()
                elif key == "aux_outputs":
                    for idx in range(len(outputs[key])):
                        for sub_key in outputs[key][idx].keys():
                            outputs[key][idx][sub_key] = outputs[key][idx][
                                sub_key
                            ].float()
                else:
                    outputs[key] = outputs[key].float()

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors["bbox"](outputs, orig_target_sizes)
        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, results)
        }
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # Collect predictions and ground truths for F1 metric
        for target, output in zip(targets, results):
            image_id = target["image_id"].item()

            # Add predictions
            if len(output["boxes"]) > 0:
                boxes = output["boxes"].float().cpu().numpy()
                # Convert from [x1, y1, x2, y2] to [x, y, w, h]
                boxes_xywh = boxes.copy()
                boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
                boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # height

                scores = output["scores"].float().cpu().numpy()
                labels = output["labels"].float().cpu().numpy()

                for box, score, label in zip(boxes_xywh, scores, labels):
                    all_predictions.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": box.tolist(),
                        "score": float(score)
                    })

            # Add ground truths
            gt_boxes = target["boxes"].float().cpu().numpy()
            if len(gt_boxes) > 0:
                # Denormalize ground truth boxes from [0,1] to pixel coordinates
                # Ground truth boxes are in normalized cxcywh format (center_x, center_y, width, height)
                orig_h, orig_w = target["orig_size"].cpu().numpy()
                gt_boxes_denorm = gt_boxes.copy()
                gt_boxes_denorm[:, [0, 2]] *= orig_w  # cx and width
                gt_boxes_denorm[:, [1, 3]] *= orig_h  # cy and height

                # Convert from [cx, cy, w, h] to [x, y, w, h]
                gt_boxes_xywh = gt_boxes_denorm.copy()
                gt_boxes_xywh[:, 0] = gt_boxes_denorm[:, 0] - gt_boxes_denorm[:, 2] / 2  # x = cx - w/2
                gt_boxes_xywh[:, 1] = gt_boxes_denorm[:, 1] - gt_boxes_denorm[:, 3] / 2  # y = cy - h/2
                # width and height stay the same at positions 2 and 3

                gt_labels = target["labels"].float().cpu().numpy()
                for box, label in zip(gt_boxes_xywh, gt_labels):
                    all_ground_truths.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": box.tolist()
                    })

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        results_json = coco_extended_metrics(coco_evaluator.coco_eval["bbox"])
        stats["results_json"] = results_json
        if "bbox" in postprocessors.keys():
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()

        if "segm" in postprocessors.keys():
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    # Compute center-based F1 metrics
    if all_predictions and all_ground_truths:
        # Debug: Log collected data
        print(f"\n[F1 DEBUG] Data collected for F1 evaluation:")
        print(f"  Total predictions collected: {len(all_predictions)}")
        print(f"  Total ground truths collected: {len(all_ground_truths)}")

        # Debug: Log score distribution of all predictions
        all_scores = [p['score'] for p in all_predictions]
        print(f"  Prediction score distribution:")
        print(f"    Min: {min(all_scores):.4f}")
        print(f"    Max: {max(all_scores):.4f}")
        print(f"    Mean: {np.mean(all_scores):.4f}")
        print(f"    Median: {np.median(all_scores):.4f}")
        print(f"    Scores >= 0.5: {sum(1 for s in all_scores if s >= 0.5)}/{len(all_scores)} ({100*sum(1 for s in all_scores if s >= 0.5)/len(all_scores):.1f}%)")
        print(f"    Scores >= 0.3: {sum(1 for s in all_scores if s >= 0.3)}/{len(all_scores)} ({100*sum(1 for s in all_scores if s >= 0.3)/len(all_scores):.1f}%)")
        print(f"    Scores >= 0.1: {sum(1 for s in all_scores if s >= 0.1)}/{len(all_scores)} ({100*sum(1 for s in all_scores if s >= 0.1)/len(all_scores):.1f}%)")

        # Debug: Log sample predictions
        print(f"\n[F1 DEBUG] Sample predictions (first 5):")
        for i, pred in enumerate(all_predictions[:5]):
            print(f"  {i}: img_id={pred['image_id']}, class={pred['category_id']}, score={pred['score']:.4f}, bbox={pred['bbox']}")

        # Debug: Log sample ground truths
        print(f"\n[F1 DEBUG] Sample ground truths (first 5):")
        for i, gt in enumerate(all_ground_truths[:5]):
            print(f"  {i}: img_id={gt['image_id']}, class={gt['category_id']}, bbox={gt['bbox']}")

        # Debug: Log class distribution
        from collections import Counter
        pred_classes = Counter([p['category_id'] for p in all_predictions])
        gt_classes = Counter([g['category_id'] for g in all_ground_truths])
        print(f"\n[F1 DEBUG] Class distribution:")
        print(f"  Prediction classes: {dict(pred_classes)}")
        print(f"  Ground truth classes: {dict(gt_classes)}")

        # Get class names from base_ds
        class_names = {cat['id']: cat['name'] for cat in base_ds.dataset['categories']}
        f1_metrics = evaluate_detection_f1(
            predictions=all_predictions,
            ground_truths=all_ground_truths,
            center_threshold=50.0,
            score_threshold=0.3,
            class_names=class_names
        )
        stats["f1_metrics"] = f1_metrics
        print(f"\nCenter-based F1 Metrics (threshold=50px):")
        print(f"  Overall F1: {f1_metrics['overall']['f1_score']:.4f}")
        print(f"  Precision: {f1_metrics['overall']['precision']:.4f}")
        print(f"  Recall: {f1_metrics['overall']['recall']:.4f}")

    return stats, coco_evaluator