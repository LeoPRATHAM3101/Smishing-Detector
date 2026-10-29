import pytest

from src.url_analysis import analyze_text


@pytest.mark.parametrize("text", [
    "Update KYC: http://sbi-kyc-update.xyz/login",
    "https://secure.sbi.co.in.evil-site.top/login",
    "Your account: http://hdfcbamk-secure.com/card",
    "Redeem: http://tinyurl.com/sbi-rewards.apk",
    "go to http://192.168.4.5/login now",
])
def test_risky_urls_flagged(text):
    assert analyze_text(text)["max_risk"] >= 60


@pytest.mark.parametrize("text", [
    "see https://www.sbi.co.in/web/personal-banking",
    "Pay at hdfcbank.com",
    "https://onlinesbi.sbi/",
    "Track at https://www.amazon.in/orders",
    "Meeting at 5pm. Call 9876543210",
])
def test_legit_urls_not_flagged(text):
    assert analyze_text(text)["max_risk"] < 35


def test_trailing_punctuation_stripped():
    r = analyze_text("Visit hdfcbank.com.")
    assert r["urls"][0]["host"] == "hdfcbank.com"
