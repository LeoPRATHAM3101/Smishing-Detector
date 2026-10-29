from src.predict import analyze


def test_obvious_scam_flagged(small_bundle):
    r = analyze("Dear customer your SBI account will be blocked today. Update KYC: http://sbi-kyc-update.xyz/login")
    assert r["label"] == "SCAM" and r["url_flag"]
    assert r["red_flags"]


def test_normal_chat_is_legit(small_bundle):
    r = analyze("Hey, are we still meeting at 6 for chai near the station?")
    assert r["label"] == "LEGIT"


def test_output_schema(small_bundle):
    r = analyze("hello")
    assert {"label", "ml_probability", "ml_threshold", "url_risk", "urls", "red_flags"} <= r.keys()
    assert 0.0 <= r["ml_probability"] <= 1.0
