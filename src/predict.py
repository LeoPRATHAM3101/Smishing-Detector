import sys

import joblib

from .config import MODEL_PATH, URL_BLOCK_RISK, URL_WARN_RISK
from .features import explain
from .url_analysis import analyze_text

_bundle = None


def load_bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"No model at {MODEL_PATH}. Run: python -m src.train")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def analyze(text: str) -> dict:
    """Hybrid verdict: ML probability + URL rules. Labels: SCAM / SUSPICIOUS / LEGIT."""
    b = load_bundle()
    prob = float(b["pipeline"].predict_proba([text])[0, 1])
    thr = b["threshold"]
    url = analyze_text(text)

    ml_flag = prob >= thr
    url_flag = url["max_risk"] >= URL_BLOCK_RISK
    if ml_flag or url_flag:
        label = "SCAM"
    elif prob >= 0.5 or url["max_risk"] >= URL_WARN_RISK:
        label = "SUSPICIOUS"
    else:
        label = "LEGIT"

    flags = list(explain(text))
    for u in url["urls"]:
        flags += [f"{u['host']}: {r}" for r in u["reasons"] if r != "Official domain"]
    return {
        "label": label,
        "ml_probability": round(prob, 4),
        "ml_threshold": round(thr, 4),
        "ml_flag": bool(ml_flag),
        "url_flag": bool(url_flag),
        "url_risk": url["max_risk"],
        "urls": url["urls"],
        "red_flags": flags,
    }


if __name__ == "__main__":
    msg = " ".join(sys.argv[1:]) or "Your SBI KYC expired. Update now: bit.ly/abc123"
    import json
    print(json.dumps(analyze(msg), indent=2, ensure_ascii=False))
