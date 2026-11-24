"""Image-level metrics for anomaly scores: higher score = more likely a defect."""

import numpy as np


def auroc(labels, scores) -> float:
    """Area under the ROC curve: the chance that a random defect image scores higher than
    a random good one (ties count half). 1.0 = perfect, 0.5 = coin flip.

    Uses the rank formula (Mann-Whitney U), so no threshold has to be picked.
    """
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        raise ValueError("need at least one good and one defect image")
    # Rank of each score (1 = lowest); tied scores share their average rank.
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def f1_max(labels, scores) -> tuple[float, float]:
    """Best F1 over all thresholds ("score >= threshold" means defect), and that threshold.

    Optimistic: the threshold is chosen on the same images it is scored on.
    """
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")
    s, y = scores[order], labels[order]
    tp, fp = np.cumsum(y), np.cumsum(~y)
    # Only cut after the last image of each group of tied scores.
    last = np.r_[s[1:] != s[:-1], True]
    tp, fp, thresholds = tp[last], fp[last], s[last]
    f1 = 2 * tp / (tp + fp + labels.sum())  # = 2PR / (P + R) with P = tp/(tp+fp), R = tp/positives
    best = int(np.argmax(f1))
    return float(f1[best]), float(thresholds[best])
