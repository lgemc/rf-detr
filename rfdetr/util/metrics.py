from typing import Optional, List, Dict, Tuple
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

plt.ioff()

PLOT_FILE_NAME = "metrics_plot.png"


def calculate_box_center(bbox: List[float]) -> Tuple[float, float]:
    """
    Calculate center of a bounding box in COCO format [x, y, width, height].

    Args:
        bbox: Bounding box as [x, y, width, height]

    Returns:
        Tuple of (center_x, center_y)
    """
    x, y, w, h = bbox
    return (x + w / 2, y + h / 2)


def center_distance(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Euclidean distance between centers of two boxes.

    Args:
        bbox1: First bounding box [x, y, width, height]
        bbox2: Second bounding box [x, y, width, height]

    Returns:
        Euclidean distance between centers
    """
    cx1, cy1 = calculate_box_center(bbox1)
    cx2, cy2 = calculate_box_center(bbox2)
    return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def match_detections(
    predictions: List[Dict],
    ground_truths: List[Dict],
    center_threshold: float = 50.0,
    score_threshold: float = 0.5
) -> Tuple[int, int, int, Dict]:
    """
    Match predictions to ground truths based on class and center proximity.

    Args:
        predictions: List of predictions with keys: image_id, category_id, bbox, score
        ground_truths: List of ground truths with keys: image_id, category_id, bbox
        center_threshold: Maximum center distance (in pixels) to consider a match
        score_threshold: Minimum confidence score for predictions

    Returns:
        Tuple of (true_positives, false_positives, false_negatives, per_class_stats)
    """
    # Debug: Log initial counts
    print(f"\n[F1 DEBUG] match_detections called:")
    print(f"  Total predictions before filtering: {len(predictions)}")
    print(f"  Total ground truths: {len(ground_truths)}")
    print(f"  Score threshold: {score_threshold}")
    print(f"  Center threshold: {center_threshold}px")

    # Debug: Log score distribution
    if predictions:
        scores = [p.get('score', 1.0) for p in predictions]
        print(f"  Prediction scores - min: {min(scores):.4f}, max: {max(scores):.4f}, mean: {np.mean(scores):.4f}")

    # Filter predictions by score threshold
    predictions = [p for p in predictions if p.get('score', 1.0) >= score_threshold]

    # Debug: Log after filtering
    print(f"  Predictions after score filtering: {len(predictions)}")

    # Group by image_id for efficient matching
    preds_by_image = defaultdict(list)
    gts_by_image = defaultdict(list)

    for pred in predictions:
        preds_by_image[pred['image_id']].append(pred)

    for gt in ground_truths:
        gts_by_image[gt['image_id']].append(gt)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    per_class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    # Get all unique image_ids
    all_image_ids = set(list(preds_by_image.keys()) + list(gts_by_image.keys()))

    # Debug: Log image counts
    print(f"  Total unique images: {len(all_image_ids)}")
    print(f"  Images with predictions: {len(preds_by_image)}")
    print(f"  Images with ground truths: {len(gts_by_image)}")

    # Debug: Sample first few images
    sample_images = list(all_image_ids)[:3]
    print(f"\n[F1 DEBUG] Sample image statistics:")
    for img_id in sample_images:
        num_preds = len(preds_by_image.get(img_id, []))
        num_gts = len(gts_by_image.get(img_id, []))
        print(f"  Image {img_id}: {num_preds} predictions, {num_gts} ground truths")

    matches_found = 0
    for image_id in all_image_ids:
        img_preds = preds_by_image.get(image_id, [])
        img_gts = gts_by_image.get(image_id, [])

        matched_gts = set()

        # Sort predictions by score (highest first) for greedy matching
        img_preds = sorted(img_preds, key=lambda x: x.get('score', 1.0), reverse=True)

        # Debug: Log details for first few images
        log_details = image_id in sample_images

        for pred_idx, pred in enumerate(img_preds):
            pred_class = pred['category_id']
            pred_bbox = pred['bbox']
            pred_score = pred.get('score', 1.0)

            best_match_idx = None
            best_match_dist = center_threshold

            # Find best matching ground truth
            candidates = []
            for idx, gt in enumerate(img_gts):
                if idx in matched_gts:
                    continue

                if gt['category_id'] != pred_class:
                    continue

                dist = center_distance(pred_bbox, gt['bbox'])
                candidates.append((idx, dist))

                if dist < best_match_dist:
                    best_match_dist = dist
                    best_match_idx = idx

            # Debug: Log matching attempt for sample images
            if log_details and pred_idx < 3:  # Only first 3 predictions per sample image
                print(f"    Pred {pred_idx} (class={pred_class}, score={pred_score:.4f}, bbox={pred_bbox}):")
                if candidates:
                    print(f"      Candidates: {len(candidates)} GTs with same class")
                    for cand_idx, cand_dist in candidates[:3]:  # Show first 3 candidates
                        print(f"        GT {cand_idx}: distance={cand_dist:.2f}px")
                else:
                    print(f"      No candidates found (no GTs with class {pred_class} or all matched)")
                if best_match_idx is not None:
                    print(f"      ✓ Matched with GT {best_match_idx} (distance={best_match_dist:.2f}px)")
                else:
                    print(f"      ✗ No match found (best distance > {center_threshold}px)")

            if best_match_idx is not None:
                # True positive
                true_positives += 1
                per_class_stats[pred_class]['tp'] += 1
                matched_gts.add(best_match_idx)
                matches_found += 1
            else:
                # False positive
                false_positives += 1
                per_class_stats[pred_class]['fp'] += 1

        # Count unmatched ground truths as false negatives
        for idx, gt in enumerate(img_gts):
            if idx not in matched_gts:
                false_negatives += 1
                per_class_stats[gt['category_id']]['fn'] += 1

    # Debug: Log final counts
    print(f"\n[F1 DEBUG] Matching results:")
    print(f"  True Positives (TP): {true_positives}")
    print(f"  False Positives (FP): {false_positives}")
    print(f"  False Negatives (FN): {false_negatives}")
    print(f"  Total matches found: {matches_found}")
    print(f"  Match rate: {matches_found}/{len(predictions)} predictions = {100*matches_found/max(len(predictions),1):.1f}%")

    # Debug: Log per-class stats
    if per_class_stats:
        print(f"\n[F1 DEBUG] Per-class stats:")
        for class_id, stats in sorted(per_class_stats.items()):
            print(f"    Class {class_id}: TP={stats['tp']}, FP={stats['fp']}, FN={stats['fn']}")

    return true_positives, false_positives, false_negatives, dict(per_class_stats)


def calculate_f1_score(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score.

    Args:
        tp: True positives
        fp: False positives
        fn: False negatives

    Returns:
        Tuple of (precision, recall, f1_score)
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def evaluate_detection_f1(
    predictions: List[Dict],
    ground_truths: List[Dict],
    center_threshold: float = 50.0,
    score_threshold: float = 0.5,
    class_names: Dict[int, str] = None
) -> Dict:
    """
    Evaluate object detection predictions using F1 score with center-based matching.

    Args:
        predictions: List of predictions in COCO format
                    Each dict should have: image_id, category_id, bbox [x,y,w,h], score
        ground_truths: List of ground truths in COCO format
                      Each dict should have: image_id, category_id, bbox [x,y,w,h]
        center_threshold: Maximum center distance in pixels to consider a match
        score_threshold: Minimum confidence score for predictions
        class_names: Optional mapping of category_id to class names

    Returns:
        Dictionary with overall and per-class metrics
    """
    print(f"\n[F1 DEBUG] evaluate_detection_f1 called with:")
    print(f"  center_threshold={center_threshold}, score_threshold={score_threshold}")

    tp, fp, fn, per_class_stats = match_detections(
        predictions, ground_truths, center_threshold, score_threshold
    )

    overall_precision, overall_recall, overall_f1 = calculate_f1_score(tp, fp, fn)

    # Debug: Log final F1 calculation
    print(f"\n[F1 DEBUG] Final F1 Score Calculation:")
    print(f"  Precision = {tp} / ({tp} + {fp}) = {overall_precision:.4f}")
    print(f"  Recall = {tp} / ({tp} + {fn}) = {overall_recall:.4f}")
    print(f"  F1 = 2 * ({overall_precision:.4f} * {overall_recall:.4f}) / ({overall_precision:.4f} + {overall_recall:.4f}) = {overall_f1:.4f}")

    per_class_f1 = {}
    for class_id, stats in per_class_stats.items():
        precision, recall, f1 = calculate_f1_score(
            stats['tp'], stats['fp'], stats['fn']
        )
        class_label = class_names.get(class_id, f"class_{class_id}") if class_names else f"class_{class_id}"
        per_class_f1[class_label] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tp': stats['tp'],
            'fp': stats['fp'],
            'fn': stats['fn']
        }

    return {
        'overall': {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        },
        'per_class': per_class_f1,
        'config': {
            'center_threshold': center_threshold,
            'score_threshold': score_threshold
        }
    }


def safe_index(arr, idx):
    return arr[idx] if 0 <= idx < len(arr) else None


class MetricsPlotSink:
    """
    The MetricsPlotSink class records training metrics and saves them to a plot.

    Args:
        output_dir (str): Directory where the plot will be saved.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.history = []

    def update(self, values: dict):
        self.history.append(values)

    def save(self):
        if not self.history:
            print("No data to plot.")
            return

        def get_array(key):
            return np.array([h[key] for h in self.history if key in h])

        epochs = get_array('epoch')
        train_loss = get_array('train_loss')
        test_loss = get_array('test_loss')
        test_coco_eval = [h['test_coco_eval_bbox'] for h in self.history if 'test_coco_eval_bbox' in h]
        ap50_90 = np.array([safe_index(x, 0) for x in test_coco_eval if x is not None], dtype=np.float32)
        ap50 = np.array([safe_index(x, 1) for x in test_coco_eval if x is not None], dtype=np.float32)
        ar50_90 = np.array([safe_index(x, 8) for x in test_coco_eval if x is not None], dtype=np.float32)

        ema_coco_eval = [h['ema_test_coco_eval_bbox'] for h in self.history if 'ema_test_coco_eval_bbox' in h]
        ema_ap50_90 = np.array([safe_index(x, 0) for x in ema_coco_eval if x is not None], dtype=np.float32)
        ema_ap50 = np.array([safe_index(x, 1) for x in ema_coco_eval if x is not None], dtype=np.float32)
        ema_ar50_90 = np.array([safe_index(x, 8) for x in ema_coco_eval if x is not None], dtype=np.float32)

        # F1 metrics
        test_f1 = [h.get('test_f1_metrics', {}).get('overall', {}).get('f1_score') for h in self.history]
        test_f1 = np.array([x for x in test_f1 if x is not None], dtype=np.float32)
        test_precision = [h.get('test_f1_metrics', {}).get('overall', {}).get('precision') for h in self.history]
        test_precision = np.array([x for x in test_precision if x is not None], dtype=np.float32)
        test_recall = [h.get('test_f1_metrics', {}).get('overall', {}).get('recall') for h in self.history]
        test_recall = np.array([x for x in test_recall if x is not None], dtype=np.float32)

        ema_f1 = [h.get('ema_test_f1_metrics', {}).get('overall', {}).get('f1_score') for h in self.history]
        ema_f1 = np.array([x for x in ema_f1 if x is not None], dtype=np.float32)
        ema_precision = [h.get('ema_test_f1_metrics', {}).get('overall', {}).get('precision') for h in self.history]
        ema_precision = np.array([x for x in ema_precision if x is not None], dtype=np.float32)
        ema_recall = [h.get('ema_test_f1_metrics', {}).get('overall', {}).get('recall') for h in self.history]
        ema_recall = np.array([x for x in ema_recall if x is not None], dtype=np.float32)

        fig, axes = plt.subplots(3, 2, figsize=(18, 18))

        # Subplot (0,0): Training and Validation Loss
        if len(epochs) > 0:
            if len(train_loss):
                axes[0][0].plot(epochs, train_loss, label='Training Loss', marker='o', linestyle='-')
            if len(test_loss):
                axes[0][0].plot(epochs, test_loss, label='Validation Loss', marker='o', linestyle='--')
            axes[0][0].set_title('Training and Validation Loss')
            axes[0][0].set_xlabel('Epoch Number')
            axes[0][0].set_ylabel('Loss Value')
            axes[0][0].legend()
            axes[0][0].grid(True)

        # Subplot (0,1): Average Precision @0.50
        if ap50.size > 0 or ema_ap50.size > 0:
            if ap50.size > 0:
                axes[0][1].plot(epochs[:len(ap50)], ap50, marker='o', linestyle='-', label='Base Model')
            if ema_ap50.size > 0:
                axes[0][1].plot(epochs[:len(ema_ap50)], ema_ap50, marker='o', linestyle='--', label='EMA Model')
            axes[0][1].set_title('Average Precision @0.50')
            axes[0][1].set_xlabel('Epoch Number')
            axes[0][1].set_ylabel('AP50')
            axes[0][1].legend()
            axes[0][1].grid(True)

        # Subplot (1,0): Average Precision @0.50:0.95
        if ap50_90.size > 0 or ema_ap50_90.size > 0:
            if ap50_90.size > 0:
                axes[1][0].plot(epochs[:len(ap50_90)], ap50_90, marker='o', linestyle='-', label='Base Model')
            if ema_ap50_90.size > 0:
                axes[1][0].plot(epochs[:len(ema_ap50_90)], ema_ap50_90, marker='o', linestyle='--', label='EMA Model')
            axes[1][0].set_title('Average Precision @0.50:0.95')
            axes[1][0].set_xlabel('Epoch Number')
            axes[1][0].set_ylabel('AP')
            axes[1][0].legend()
            axes[1][0].grid(True)

        # Subplot (1,1): Average Recall @0.50:0.95
        if ar50_90.size > 0 or ema_ar50_90.size > 0:
            if ar50_90.size > 0:
                axes[1][1].plot(epochs[:len(ar50_90)], ar50_90, marker='o', linestyle='-', label='Base Model')
            if ema_ar50_90.size > 0:
                axes[1][1].plot(epochs[:len(ema_ar50_90)], ema_ar50_90, marker='o', linestyle='--', label='EMA Model')
            axes[1][1].set_title('Average Recall @0.50:0.95')
            axes[1][1].set_xlabel('Epoch Number')
            axes[1][1].set_ylabel('AR')
            axes[1][1].legend()
            axes[1][1].grid(True)

        # Subplot (2,0): Center-based F1 Score
        if test_f1.size > 0 or ema_f1.size > 0:
            if test_f1.size > 0:
                axes[2][0].plot(epochs[:len(test_f1)], test_f1, marker='o', linestyle='-', label='Base Model')
            if ema_f1.size > 0:
                axes[2][0].plot(epochs[:len(ema_f1)], ema_f1, marker='o', linestyle='--', label='EMA Model')
            axes[2][0].set_title('Center-based F1 Score (50px threshold)')
            axes[2][0].set_xlabel('Epoch Number')
            axes[2][0].set_ylabel('F1 Score')
            axes[2][0].legend()
            axes[2][0].grid(True)

        # Subplot (2,1): Center-based Precision and Recall
        if test_precision.size > 0 or test_recall.size > 0 or ema_precision.size > 0 or ema_recall.size > 0:
            if test_precision.size > 0:
                axes[2][1].plot(epochs[:len(test_precision)], test_precision, marker='o', linestyle='-', label='Base Precision')
            if test_recall.size > 0:
                axes[2][1].plot(epochs[:len(test_recall)], test_recall, marker='s', linestyle='-', label='Base Recall')
            if ema_precision.size > 0:
                axes[2][1].plot(epochs[:len(ema_precision)], ema_precision, marker='o', linestyle='--', label='EMA Precision')
            if ema_recall.size > 0:
                axes[2][1].plot(epochs[:len(ema_recall)], ema_recall, marker='s', linestyle='--', label='EMA Recall')
            axes[2][1].set_title('Center-based Precision & Recall (50px threshold)')
            axes[2][1].set_xlabel('Epoch Number')
            axes[2][1].set_ylabel('Score')
            axes[2][1].legend()
            axes[2][1].grid(True)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{PLOT_FILE_NAME}")
        plt.close(fig)
        print(f"Results saved to {self.output_dir}/{PLOT_FILE_NAME}")


class MetricsTensorBoardSink:
    """
    Training metrics via TensorBoard.

    Args:
        output_dir (str): Directory where TensorBoard logs will be written.
    """

    def __init__(self, output_dir: str):
        if SummaryWriter:
            self.writer = SummaryWriter(log_dir=output_dir)
            print(f"TensorBoard logging initialized. To monitor logs, use 'tensorboard --logdir {output_dir}' and open http://localhost:6006/ in browser.")
        else:
            self.writer = None
            print("Unable to initialize TensorBoard. Logging is turned off for this session.  Run 'pip install tensorboard' to enable logging.")

    def update(self, values: dict):
        if not self.writer:
            return

        epoch = values['epoch']

        if 'train_loss' in values:
            self.writer.add_scalar("Loss/Train", values['train_loss'], epoch)
        if 'test_loss' in values:
            self.writer.add_scalar("Loss/Test", values['test_loss'], epoch)

        if 'test_coco_eval_bbox' in values:
            coco_eval = values['test_coco_eval_bbox']
            ap50_90 = safe_index(coco_eval, 0)
            ap50 = safe_index(coco_eval, 1)
            ar50_90 = safe_index(coco_eval, 8)
            if ap50_90 is not None:
                self.writer.add_scalar("Metrics/Base/AP50_90", ap50_90, epoch)
            if ap50 is not None:
                self.writer.add_scalar("Metrics/Base/AP50", ap50, epoch)
            if ar50_90 is not None:
                self.writer.add_scalar("Metrics/Base/AR50_90", ar50_90, epoch)

        if 'ema_test_coco_eval_bbox' in values:
            ema_coco_eval = values['ema_test_coco_eval_bbox']
            ema_ap50_90 = safe_index(ema_coco_eval, 0)
            ema_ap50 = safe_index(ema_coco_eval, 1)
            ema_ar50_90 = safe_index(ema_coco_eval, 8)
            if ema_ap50_90 is not None:
                self.writer.add_scalar("Metrics/EMA/AP50_90", ema_ap50_90, epoch)
            if ema_ap50 is not None:
                self.writer.add_scalar("Metrics/EMA/AP50", ema_ap50, epoch)
            if ema_ar50_90 is not None:
                self.writer.add_scalar("Metrics/EMA/AR50_90", ema_ar50_90, epoch)

        # F1 metrics
        if 'test_f1_metrics' in values:
            f1_data = values['test_f1_metrics']['overall']
            self.writer.add_scalar("Metrics/Base/F1", f1_data['f1_score'], epoch)
            self.writer.add_scalar("Metrics/Base/F1_Precision", f1_data['precision'], epoch)
            self.writer.add_scalar("Metrics/Base/F1_Recall", f1_data['recall'], epoch)

        if 'ema_test_f1_metrics' in values:
            ema_f1_data = values['ema_test_f1_metrics']['overall']
            self.writer.add_scalar("Metrics/EMA/F1", ema_f1_data['f1_score'], epoch)
            self.writer.add_scalar("Metrics/EMA/F1_Precision", ema_f1_data['precision'], epoch)
            self.writer.add_scalar("Metrics/EMA/F1_Recall", ema_f1_data['recall'], epoch)

        self.writer.flush()

    def close(self):
        if not self.writer:
            return
        
        self.writer.close()

class MetricsWandBSink:
    """
    Training metrics via W&B.

    Args:
        output_dir (str): Directory where W&B logs will be written locally.
        project (str, optional): Associate this training run with a W&B project. If None, W&B will generate a name based on the git repo name.
        run (str, optional): W&B run name. If None, W&B will generate a random name.
        config (dict, optional): Input parameters, like hyperparameters or data preprocessing settings for the run for later comparison.
    """

    def __init__(self, output_dir: str, project: Optional[str] = None, run: Optional[str] = None, config: Optional[dict] = None):
        self.output_dir = output_dir
        if wandb:
            self.run = wandb.init(
                project=project,
                name=run,
                config=config,
                dir=output_dir
            )
            print(f"W&B logging initialized. To monitor logs, open {wandb.run.url}.")
        else:
            self.run = None
            print("Unable to initialize W&B. Logging is turned off for this session. Run 'pip install wandb' to enable logging.")

    def update(self, values: dict):
        if not wandb or not self.run:
            return

        epoch = values['epoch']
        log_dict = {"epoch": epoch}

        if 'train_loss' in values:
            log_dict["Loss/Train"] = values['train_loss']
        if 'test_loss' in values:
            log_dict["Loss/Test"] = values['test_loss']

        if 'test_coco_eval_bbox' in values:
            coco_eval = values['test_coco_eval_bbox']
            ap50_90 = safe_index(coco_eval, 0)
            ap50 = safe_index(coco_eval, 1)
            ar50_90 = safe_index(coco_eval, 8)
            if ap50_90 is not None:
                log_dict["Metrics/Base/AP50_90"] = ap50_90
            if ap50 is not None:
                log_dict["Metrics/Base/AP50"] = ap50
            if ar50_90 is not None:
                log_dict["Metrics/Base/AR50_90"] = ar50_90

        if 'ema_test_coco_eval_bbox' in values:
            ema_coco_eval = values['ema_test_coco_eval_bbox']
            ema_ap50_90 = safe_index(ema_coco_eval, 0)
            ema_ap50 = safe_index(ema_coco_eval, 1)
            ema_ar50_90 = safe_index(ema_coco_eval, 8)
            if ema_ap50_90 is not None:
                log_dict["Metrics/EMA/AP50_90"] = ema_ap50_90
            if ema_ap50 is not None:
                log_dict["Metrics/EMA/AP50"] = ema_ap50
            if ema_ar50_90 is not None:
                log_dict["Metrics/EMA/AR50_90"] = ema_ar50_90

        # F1 metrics
        if 'test_f1_metrics' in values:
            f1_data = values['test_f1_metrics']['overall']
            log_dict["Metrics/Base/F1"] = f1_data['f1_score']
            log_dict["Metrics/Base/F1_Precision"] = f1_data['precision']
            log_dict["Metrics/Base/F1_Recall"] = f1_data['recall']

        if 'ema_test_f1_metrics' in values:
            ema_f1_data = values['ema_test_f1_metrics']['overall']
            log_dict["Metrics/EMA/F1"] = ema_f1_data['f1_score']
            log_dict["Metrics/EMA/F1_Precision"] = ema_f1_data['precision']
            log_dict["Metrics/EMA/F1_Recall"] = ema_f1_data['recall']

        wandb.log(log_dict)

    def close(self):
        if not wandb or not self.run:
            return
            
        self.run.finish()