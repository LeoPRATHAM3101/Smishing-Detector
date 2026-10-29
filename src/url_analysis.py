"""Rule-based URL risk scoring: shorteners, raw IPs, odd TLDs, punycode, brand look-alikes."""
import difflib
import re

from .config import URL_BLOCK_RISK, URL_WARN_RISK
from .features import URL_RE

SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly", "rb.gy",
              "shorturl.at", "tiny.cc", "ow.ly", "bitly.com", "s.id", "v.gd"}
MESSAGING = {"wa.me", "t.me", "chat.whatsapp.com", "telegram.me"}
ODD_TLDS = {"xyz", "top", "cc", "club", "online", "site", "link", "icu", "buzz", "live",
            "click", "tk", "ml", "ga", "cf", "gq", "pw", "vip", "fun", "work"}
SLD = {"co", "com", "org", "net", "gov", "nic", "ac", "edu"}

# brand -> (name tokens to look for, official domains)
BRANDS = {
    "SBI": (["sbi", "sbicard"], ["sbi.co.in", "onlinesbi.sbi", "sbi.bank.in", "sbicard.com"]),
    "HDFC Bank": (["hdfc", "hdfcbank"], ["hdfcbank.com", "hdfc.com"]),
    "ICICI Bank": (["icici", "icicibank"], ["icicibank.com"]),
    "Axis Bank": (["axisbank", "axis"], ["axisbank.com"]),
    "Kotak": (["kotak"], ["kotak.com"]),
    "Paytm": (["paytm"], ["paytm.com"]),
    "PhonePe": (["phonepe"], ["phonepe.com"]),
    "Google Pay": (["gpay", "googlepay"], ["pay.google.com"]),
    "Amazon": (["amazon"], ["amazon.in", "amazon.com"]),
    "Flipkart": (["flipkart"], ["flipkart.com"]),
    "India Post": (["indiapost"], ["indiapost.gov.in"]),
    "Income Tax": (["incometax", "incometaxindia"], ["incometax.gov.in", "incometaxindia.gov.in"]),
    "UIDAI/Aadhaar": (["uidai", "aadhaar"], ["uidai.gov.in"]),
    "FASTag/NHAI": (["fastag", "nhai"], ["npci.org.in", "nhai.gov.in", "ihmcl.com"]),
    "Jio": (["jio"], ["jio.com"]),
    "Airtel": (["airtel"], ["airtel.in"]),
}


def _parse_host(url: str) -> str:
    u = re.sub(r"^[a-z][a-z0-9+.-]*://", "", url, flags=re.I)
    u = re.split(r"[/?#]", u, maxsplit=1)[0].rsplit("@", 1)[-1]
    return u.split(":")[0].lower().strip(".")


def _is_official(host: str, domains) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def _clean(url: str) -> str:
    return url.rstrip(".,;:!?)\"'")


def _brand_hit(host: str):
    """Return (brand, kind) if host imitates a brand without being official."""
    tokens = [t for t in re.split(r"[.-]", host) if t]
    for brand, (names, official) in BRANDS.items():
        if _is_official(host, official):
            return brand, "official"
    for brand, (names, _) in BRANDS.items():
        for n in names:
            for t in tokens:
                if t == n or (len(n) >= 4 and n in t):
                    return brand, "lookalike"
                if len(t) >= 5 and len(n) >= 4 and difflib.SequenceMatcher(None, t, n).ratio() >= 0.8:
                    return brand, "typosquat"
    return None, None


def analyze_url(url: str) -> dict:
    url = _clean(url)
    host = _parse_host(url)
    low = url.lower()
    reasons, risk = [], 0

    brand, kind = _brand_hit(host)
    if kind == "official":
        return {"url": url, "host": host, "risk": 0, "brand": brand, "reasons": ["Official domain"]}

    labels = host.split(".")
    tld = labels[-1] if labels else ""
    if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", host):
        risk += 60; reasons.append("Link uses a raw IP address")
    if any(host == s or host.endswith("." + s) for s in SHORTENERS):
        risk += 40; reasons.append("URL shortener hides the real destination")
    if host in MESSAGING:
        risk += 20; reasons.append("Link moves you to a chat app (common in job/lottery scams)")
    if tld in ODD_TLDS:
        risk += 30; reasons.append(f"Suspicious domain ending (.{tld})")
    if ".apk" in low:
        risk += 45; reasons.append("Links to an .apk file (Android malware install risk)")
    if "xn--" in host:
        risk += 40; reasons.append("Punycode domain (can disguise look-alike characters)")
    if brand and kind in {"lookalike", "typosquat"}:
        risk += 60
        reasons.append(f"Imitates {brand} but is not its real domain"
                       + (" (misspelt brand)" if kind == "typosquat" else ""))
    if host.count("-") >= 2:
        risk += 10; reasons.append("Many hyphens in the domain")
    if len(labels) >= 5:
        risk += 15; reasons.append("Unusually deep sub-domain chain")

    return {"url": url, "host": host, "risk": min(risk, 100), "brand": brand, "reasons": reasons}


def level(risk: int) -> str:
    return "high" if risk >= URL_BLOCK_RISK else "medium" if risk >= URL_WARN_RISK else \
           "low" if risk > 0 else "none"


def analyze_text(text: str) -> dict:
    seen, urls = set(), []
    for m in URL_RE.finditer(str(text)):
        u = _clean(m.group(0))
        if u and u.lower() not in seen:
            seen.add(u.lower())
            urls.append(analyze_url(u))
    max_risk = max((u["risk"] for u in urls), default=0)
    return {"urls": urls, "max_risk": max_risk, "level": level(max_risk)}
