from src.features import FEATURE_NAMES, explain, extract_features, normalize


def test_feature_shape():
    X = extract_features(["hello", "Update KYC at bit.ly/x now!"])
    assert X.shape == (2, len(FEATURE_NAMES))


def test_normalize_masks_values():
    out = normalize("Call 9876543210 or visit http://evil.xyz/a Rs 5000")
    assert "phonetoken" in out and "urltoken" in out and "5000" not in out


def test_explain_flags():
    flags = explain("Your SBI account is blocked. Update KYC: bit.ly/abc OTP")
    assert any("Urgency" in f for f in flags)
    assert any("OTP" in f for f in flags)
