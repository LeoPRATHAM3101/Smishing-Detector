import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import RANDOM_STATE, REPORTS_DIR
from .data import load_data
from .evaluation import summarize
from .features import normalize
from .pipeline import build_pipeline


def text_only(clf):
    return Pipeline([("tfidf", TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2),
                                               min_df=2, sublinear_tf=True)), ("clf", clf)])


def main():
    df = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(
        df.text, df.label, test_size=0.2, stratify=df.label, random_state=RANDOM_STATE)
    models = {
        "Complement Naive Bayes (word TF-IDF)": text_only(ComplementNB(alpha=0.3)),
        "Linear SVM (word TF-IDF)": text_only(LinearSVC(class_weight="balanced")),
        "SGD / modified Huber (word TF-IDF)": text_only(
            SGDClassifier(loss="modified_huber", class_weight="balanced", random_state=RANDOM_STATE)),
        "Logistic Regression (word TF-IDF)": text_only(
            LogisticRegression(C=10, class_weight="balanced", max_iter=3000)),
        "Logistic Regression (word+char+hand features) [ours]": build_pipeline(),
    }
    rows = []
    for name, m in models.items():
        t = time.time()
        cv = cross_val_score(m, X_tr, y_tr, cv=5, scoring="f1")
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        score = m.predict_proba(X_te)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X_te)
        s = summarize(y_te, pred, score)
        rows.append({"model": name, "cv_f1": round(cv.mean(), 4), "precision": round(s["precision"], 4),
                     "recall": round(s["recall"], 4), "f1": round(s["f1"], 4),
                     "roc_auc": round(s["roc_auc"], 4), "false_alarms": s["false_alarms"],
                     "missed_scams": s["missed_scams"], "seconds": round(time.time() - t, 1)})
        print(rows[-1])
    out = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    out.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    cols = list(out.columns)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in out.values.tolist()]
    (REPORTS_DIR / "model_comparison.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
