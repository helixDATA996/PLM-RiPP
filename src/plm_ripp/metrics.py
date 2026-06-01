"""
Evaluation metrics for classification
"""
from typing import List
from collections import Counter

import numpy as np
import torch


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """
    Compute inverse-frequency class weights, normalized to mean=1.

    Args:
        labels: List of integer labels
        num_classes: Total number of classes

    Returns:
        Tensor of class weights [num_classes]
    """
    counter = Counter(labels)
    total = len(labels)
    w = torch.zeros(num_classes, dtype=torch.float32)

    for c in range(num_classes):
        freq = counter.get(c, 1) / total
        w[c] = 1.0 / max(freq, 1e-12)

    w = w / w.mean()
    return w


def compute_macro_f1(
    true_labels: List[int],
    pred_labels: List[int],
    num_classes: int
) -> float:
    """
    Compute macro-averaged F1 score.

    Macro-F1 computes F1 for each class independently and averages them,
    giving equal weight to each class regardless of support.

    Args:
        true_labels: Ground truth labels
        pred_labels: Predicted labels
        num_classes: Total number of classes

    Returns:
        Macro-averaged F1 score
    """
    f1_scores = []

    for c in range(num_classes):
        tp = sum((t == c and p == c) for t, p in zip(true_labels, pred_labels))
        fp = sum((t != c and p == c) for t, p in zip(true_labels, pred_labels))
        fn = sum((t == c and p != c) for t, p in zip(true_labels, pred_labels))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        f1_scores.append(f1)

    return float(np.mean(f1_scores))
