"""
Simple standalone script to calculate F1 score for object detection.
Adapted from animaldet evaluation.
"""

from rfdetr.f1_metric import calculate_f1


# Example usage:
if __name__ == "__main__":
    # Example predictions in COCO format
    # Each prediction should have: image_id, category_id, bbox [x,y,w,h], score
    predictions = [
        {"image_id": 1, "category_id": 1, "bbox": [100, 100, 50, 50], "score": 0.9},
        {"image_id": 1, "category_id": 1, "bbox": [200, 200, 50, 50], "score": 0.8},
        {"image_id": 2, "category_id": 2, "bbox": [150, 150, 60, 60], "score": 0.7},
    ]

    # Example ground truths in COCO format
    # Each GT should have: image_id, category_id, bbox [x,y,w,h]
    ground_truths = [
        {"image_id": 1, "category_id": 1, "bbox": [95, 105, 50, 50]},
        {"image_id": 1, "category_id": 1, "bbox": [205, 195, 50, 50]},
        {"image_id": 2, "category_id": 2, "bbox": [155, 145, 60, 60]},
    ]

    # Calculate F1 metrics
    metrics = calculate_f1(
        predictions=predictions,
        ground_truths=ground_truths,
        center_threshold=50.0,  # Max center distance in pixels
        score_threshold=0.5,    # Min confidence score
    )

    print("F1 Metrics:")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  True Positives: {metrics['tp']}")
    print(f"  False Positives: {metrics['fp']}")
    print(f"  False Negatives: {metrics['fn']}")
