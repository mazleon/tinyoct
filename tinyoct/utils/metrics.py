"""
Metric computation utilities.
All metrics reported in the paper are computed here.
"""

import numpy as np
from typing import List

try:
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score,
        confusion_matrix, classification_report,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("scikit-learn not installed. pip install scikit-learn")

CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]


def compute_metrics(
    labels: List[int],
    preds: List[int],
    probs: List[List[float]],
) -> dict:
    """
    Compute all paper metrics from lists of labels, predictions, probabilities.

    Returns:
        dict with keys:
          accuracy, macro_f1, per_class_f1, macro_auc,
          confusion_matrix, dme_cnv_confusion (computed in Evaluator)
    """
    if not SKLEARN_AVAILABLE:
        return {"accuracy": sum(p == l for p, l in zip(preds, labels)) / len(labels)}

    labels_np = np.array(labels)
    preds_np  = np.array(preds)
    probs_np  = np.array(probs)

    acc    = accuracy_score(labels_np, preds_np)
    mf1    = f1_score(labels_np, preds_np, average="macro", zero_division=0)
    pf1    = f1_score(labels_np, preds_np, average=None, zero_division=0)
    cm     = confusion_matrix(labels_np, preds_np)

    try:
        mauc = roc_auc_score(
            labels_np, probs_np,
            multi_class="ovr", average="macro"
        )
    except Exception:
        mauc = 0.0

    per_class = {CLASS_NAMES[i]: float(pf1[i]) for i in range(min(4, len(pf1)))}

    # Flat per-class keys so checkpoint monitor / save_checkpoint can access them
    # directly via metrics.get("drusen_f1") without nested dict lookup
    flat_per_class = {f"{cls.lower()}_f1": v for cls, v in per_class.items()}

    return {
        "accuracy":        float(acc),
        "macro_f1":        float(mf1),
        "per_class_f1":    per_class,
        "macro_auc":       float(mauc),
        "confusion_matrix": cm.tolist(),
        # Flat aliases — enables checkpoint.monitor = "drusen_f1"
        **flat_per_class,   # cnv_f1, dme_f1, drusen_f1, normal_f1
    }
