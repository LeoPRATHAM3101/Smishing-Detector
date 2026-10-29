import re

import numpy as np

URL_RE = re.compile(
    r"(?:https?://|www\.)[^\s]+|"
    r"\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|in|net|org|co|io|ly|gl|me|xyz|top|cc|club|online|"
    r"site|link|info|icu|buzz|live|apk|app|biz|store|shop|vip|bank|sbi|gov|click|tk|ml|ga|cf|gq|"
    r"tech|work|fun|pw|ws)\b(?:/[^\s]*)?",
    re.I,
)
PHONE_RE = re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)")
CURRENCY_RE = re.compile(r"₹|\brs\.?\s?\d|\binr\b|\$|£")
CREDENTIAL_RE = re.compile(r"\b(?:otp|pin|cvv|password)\b")
URGENCY_RE = re.compile(
    r"\b(?:urgent|immediately|expire[sd]?|blocked|suspend(?:ed)?|last chance|act now|"
    r"final notice|deactivated|within 24|today only|frozen|disconnected|turant|abhi)\b")
FINANCE_RE = re.compile(
    r"\b(?:kyc|pan|aadhaar|aadhar|upi|sbi|hdfc|icici|axis|paytm|phonepe|gpay|bank|account|"
    r"credit card|loan|refund|reward|points|cashback|electricity|bill|fastag|income tax|"
    r"customs|parcel|courier|india post)\b")

FEATURE_NAMES = [
    "n_urls", "has_apk", "n_urgency", "n_finance", "has_phone", "digit_ratio",
    "upper_ratio", "log_len", "has_currency", "n_exclaim", "asks_credentials",
]


def normalize(text: str) -> str:
    """Mask URLs/phones/numbers so the model learns patterns, not specific values."""
    text = str(text).lower()
    text = URL_RE.sub(" urltoken ", text)
    text = PHONE_RE.sub(" phonetoken ", text)
    return re.sub(r"\d+", " 0 ", text)


def _row(t: str):
    t = str(t)
    low = t.lower()
    n = max(len(t), 1)
    return [
        len(URL_RE.findall(t)),
        int(".apk" in low),
        len(URGENCY_RE.findall(low)),
        len(FINANCE_RE.findall(low)),
        int(bool(PHONE_RE.search(t))),
        sum(c.isdigit() for c in t) / n,
        sum(c.isupper() for c in t) / n,
        float(np.log1p(len(t))),
        int(bool(CURRENCY_RE.search(low))),
        t.count("!"),
        int(bool(CREDENTIAL_RE.search(low))),
    ]


def extract_features(texts) -> np.ndarray:
    return np.array([_row(t) for t in texts], dtype=float)


def explain(text: str) -> list:
    """Human-readable text-level red flags (URL-level flags come from url_analysis)."""
    f = dict(zip(FEATURE_NAMES, _row(text)))
    out = []
    if f["n_urgency"]:
        out.append("Urgency / threat language")
    if f["n_finance"] and f["n_urls"]:
        out.append("Banking / KYC / government theme combined with a link")
    if f["asks_credentials"]:
        out.append("Mentions OTP / PIN / password")
    if f["has_phone"]:
        out.append("Asks you to contact a phone number")
    if f["has_currency"]:
        out.append("Mentions money / prizes")
    return out
