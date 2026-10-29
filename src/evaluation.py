import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)

from .config import DEFAULT_THRESHOLD, TARGET_PRECISION


def tune_threshold(y_true, proba, target_precision: float = TARGET_PRECISION) -> float:
    """Highest-recall threshold whose out-of-fold precision >= target (falls back to 0.5)."""
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    prec, rec, thr = precision_recall_curve(y_true, proba)
    ok = np.where(prec[:-1] >= target_precision)[0]
    if len(ok) == 0:
        return DEFAULT_THRESHOLD
    best_recall = rec[ok].max()
    cands = [i for i in ok if rec[i] == best_recall]
    return float(thr[max(cands)])


def summarize(y_true, pred, proba=None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out = {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "false_alarms": int(fp), "missed_scams": int(fn), "tp": int(tp), "tn": int(tn),
    }
    if proba is not None:
        out["roc_auc"] = float(roc_auc_score(y_true, proba))
        out["avg_precision"] = float(average_precision_score(y_true, proba))
    return out


def fmt(m: dict) -> str:
    return (f"precision {m['precision']:.4f} | recall {m['recall']:.4f} | F1 {m['f1']:.4f} | "
            f"false alarms {m['false_alarms']} | missed scams {m['missed_scams']}")
