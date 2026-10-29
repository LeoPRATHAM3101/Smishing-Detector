import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from .config import MODEL_PATH, RANDOM_STATE, REPORTS_DIR, URL_BLOCK_RISK
from .data import load_data
from .evaluation import fmt, summarize, tune_threshold
from .pipeline import build_pipeline, feature_names
from .url_analysis import analyze_text


def url_flags(texts) -> np.ndarray:
    return np.array([analyze_text(t)["max_risk"] >= URL_BLOCK_RISK for t in texts])


def make_bundle(X, y, folds: int = 5):
    """Tune threshold on out-of-fold probabilities, then fit on all of X."""
    pipe = build_pipeline()
    cv = StratifiedKFold(folds, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    thr = tune_threshold(y, oof)
    pipe.fit(X, y)
    return {"pipeline": pipe, "threshold": thr}, oof


def main():
    df = load_data()
    print(f"Dataset: {len(df)} messages | scam share {df.label.mean():.1%} | "
          f"{(df.source == 'india').sum()} India samples")

    tr, te = train_test_split(df.index, test_size=0.2, stratify=df.label, random_state=RANDOM_STATE)
    X_tr, y_tr, X_te, y_te = df.text[tr], df.label[tr], df.text[te], df.label[te]

    print("Cross-validating to tune the decision threshold...")
    bundle, oof = make_bundle(X_tr, y_tr)
    pipe, thr = bundle["pipeline"], bundle["threshold"]
    print(f"Tuned threshold: {thr:.3f} (max recall at OOF precision >= 0.99)")

    proba = pipe.predict_proba(X_te)[:, 1]
    uf = url_flags(X_te)
    results = {
        "ml_only_0.50": summarize(y_te, proba >= 0.5, proba),
        "ml_only_tuned": summarize(y_te, proba >= thr, proba),
        "hybrid_tuned": summarize(y_te, (proba >= thr) | uf, proba),
    }
    for k, v in results.items():
        print(f"{k:>14}: {fmt(v)}")
    print(f"ROC-AUC {results['ml_only_tuned']['roc_auc']:.4f} | "
          f"avg precision {results['ml_only_tuned']['avg_precision']:.4f}")

    mask = (df.source[te] == "india").values
    if mask.sum() and y_te[mask].sum():
        results["india_slice"] = {
            "n": int(mask.sum()),
            "ml_only": summarize(y_te[mask], (proba >= thr)[mask]),
            "hybrid": summarize(y_te[mask], ((proba >= thr) | uf)[mask]),
        }
        s = results["india_slice"]
        print(f"India slice (n={s['n']}, too small to be a benchmark): "
              f"recall ML-only {s['ml_only']['recall']:.2f} -> hybrid {s['hybrid']['recall']:.2f}")

    REPORTS_DIR.mkdir(exist_ok=True)
    results["threshold"] = thr
    results["n_train"], results["n_test"] = int(len(tr)), int(len(te))
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2))

    ConfusionMatrixDisplay.from_predictions(
        y_te, (proba >= thr) | uf, display_labels=["legit", "scam"], cmap="Blues")
    plt.title("Confusion matrix (test set, hybrid)")
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight"); plt.close()

    PrecisionRecallDisplay.from_predictions(y_te, proba)
    plt.axhline(0.99, ls="--", c="grey", lw=1)
    plt.title("Precision-recall (ML only)")
    plt.savefig(REPORTS_DIR / "precision_recall.png", dpi=150, bbox_inches="tight"); plt.close()

    coefs, names = pipe.named_steps["clf"].coef_[0], np.array(feature_names(pipe))
    top = np.argsort(coefs)[-15:]
    plt.figure(figsize=(7, 5)); plt.barh(names[top], coefs[top])
    plt.title("Top features pushing towards SCAM"); plt.tight_layout()
    plt.savefig(REPORTS_DIR / "top_features.png", dpi=150); plt.close()

    # final model: refit on ALL data for deployment
    final, _ = make_bundle(df.text, df.label)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH} (threshold {final['threshold']:.3f})")


if __name__ == "__main__":
    main()
