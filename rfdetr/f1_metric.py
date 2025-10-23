"""Simple F1 metric calculator for object detection.

Adapted from animaldet evaluation metrics.
"""

from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
import torch


def calculate_box_center(bbox: List[float]) -> Tuple[float, float]:
    """Calculate center of a bounding box in format [x, y, width, height]."""
    x, y, w, h = bbox
    return (x + w / 2, y + h / 2)


def center_distance(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate Euclidean distance between centers of two boxes."""
    cx1, cy1 = calculate_box_center(bbox1)
    cx2, cy2 = calculate_box_center(bbox2)
    return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def collect_predictions_and_gts(results: List[Dict], targets: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Collect predictions and ground truths from model outputs."""
    all_predictions = []
    all_ground_truths = []

    for target, output in zip(targets, results):
        image_id = target["image_id"].item() if torch.is_tensor(target["image_id"]) else target["image_id"]

        # Add predictions
        if "boxes" in output and len(output["boxes"]) > 0:
            boxes = output["boxes"].float().cpu().numpy() if torch.is_tensor(output["boxes"]) else output["boxes"]

            # Convert from [x1, y1, x2, y2] to [x, y, w, h]
            boxes_xywh = boxes.copy()
            boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
            boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # height

            scores = output["scores"].float().cpu().numpy() if torch.is_tensor(output["scores"]) else output["scores"]
            labels = output["labels"].float().cpu().numpy() if torch.is_tensor(output["labels"]) else output["labels"]

            for box, score, label in zip(boxes_xywh, scores, labels):
                all_predictions.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": box.tolist() if hasattr(box, 'tolist') else box,
                    "score": float(score)
                })

        # Add ground truths
        if "boxes" in target and len(target["boxes"]) > 0:
            gt_boxes = target["boxes"].float().cpu().numpy() if torch.is_tensor(target["boxes"]) else target["boxes"]

            # Denormalize if needed
            if gt_boxes.max() <= 1.0:
                orig_h, orig_w = target["orig_size"].cpu().numpy() if torch.is_tensor(target["orig_size"]) else target["orig_size"]
                gt_boxes_denorm = gt_boxes.copy()
                gt_boxes_denorm[:, [0, 2]] *= orig_w  # cx and width
                gt_boxes_denorm[:, [1, 3]] *= orig_h  # cy and height
            else:
                gt_boxes_denorm = gt_boxes

            # Convert from [cx, cy, w, h] to [x, y, w, h]
            gt_boxes_xywh = gt_boxes_denorm.copy()
            gt_boxes_xywh[:, 0] = gt_boxes_denorm[:, 0] - gt_boxes_denorm[:, 2] / 2  # x = cx - w/2
            gt_boxes_xywh[:, 1] = gt_boxes_denorm[:, 1] - gt_boxes_denorm[:, 3] / 2  # y = cy - h/2

            gt_labels = target["labels"].float().cpu().numpy() if torch.is_tensor(target["labels"]) else target["labels"]

            for box, label in zip(gt_boxes_xywh, gt_labels):
                all_ground_truths.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": box.tolist() if hasattr(box, 'tolist') else box
                })

    return all_predictions, all_ground_truths


def calculate_f1(predictions: List[Dict], ground_truths: List[Dict],
                 center_threshold: float = 50.0, score_threshold: float = 0.5) -> Dict:
    """Calculate F1 score with center-based matching."""

    # Filter predictions by score
    predictions = [p for p in predictions if p.get('score', 1.0) >= score_threshold]

    # Group by image_id
    preds_by_image = defaultdict(list)
    gts_by_image = defaultdict(list)

    for pred in predictions:
        preds_by_image[pred['image_id']].append(pred)
    for gt in ground_truths:
        gts_by_image[gt['image_id']].append(gt)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Get all unique image_ids
    all_image_ids = set(list(preds_by_image.keys()) + list(gts_by_image.keys()))

    for image_id in all_image_ids:
        img_preds = preds_by_image.get(image_id, [])
        img_gts = gts_by_image.get(image_id, [])

        matched_gts = set()

        # Sort predictions by score (highest first)
        img_preds = sorted(img_preds, key=lambda x: x.get('score', 1.0), reverse=True)

        for pred in img_preds:
            pred_class = pred['category_id']
            pred_bbox = pred['bbox']

            best_match_idx = None
            best_match_dist = center_threshold

            # Find best matching ground truth
            for idx, gt in enumerate(img_gts):
                if idx in matched_gts or gt['category_id'] != pred_class:
                    continue

                dist = center_distance(pred_bbox, gt['bbox'])
                if dist < best_match_dist:
                    best_match_dist = dist
                    best_match_idx = idx

            if best_match_idx is not None:
                true_positives += 1
                matched_gts.add(best_match_idx)
            else:
                false_positives += 1

        # Count unmatched ground truths
        false_negatives += len(img_gts) - len(matched_gts)

    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'f1': f1_score,
        'precision': precision,
        'recall': recall,
        'tp': true_positives,
        'fp': false_positives,
        'fn': false_negatives
    }
