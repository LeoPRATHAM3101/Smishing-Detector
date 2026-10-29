import io
import urllib.request
import zipfile

import pandas as pd

from .config import INDIA_FILE, RAW_DIR, RAW_FILE

UCI_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def download_uci() -> None:
    if RAW_FILE.exists():
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading UCI SMS Spam Collection...")
    with urllib.request.urlopen(UCI_URL, timeout=60) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extract("SMSSpamCollection", RAW_FILE.parent)


def load_data(include_uci: bool = True) -> pd.DataFrame:
    """Return DataFrame[text, label, source]. label: 1 = spam/smishing, 0 = legit."""
    frames = []
    if include_uci:
        download_uci()
        uci = pd.read_csv(RAW_FILE, sep="\t", header=None, names=["label", "text"],
                          quoting=3, encoding="latin-1")
        uci["label"] = uci["label"].map({"ham": 0, "spam": 1})
        uci["source"] = "uci"
        frames.append(uci)
    india = pd.read_csv(INDIA_FILE)
    india["source"] = "india"
    frames.append(india[["label", "text", "source"]])

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna().drop_duplicates(subset="text").reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df
