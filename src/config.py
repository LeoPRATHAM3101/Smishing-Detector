import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_FILE = Path(os.environ.get("SMS_RAW_FILE", RAW_DIR / "SMSSpamCollection"))
INDIA_FILE = ROOT / "data" / "india_samples.csv"
MODEL_PATH = Path(os.environ.get("SMISHING_MODEL_PATH", ROOT / "models" / "smishing_model.joblib"))
REPORTS_DIR = ROOT / "reports"

RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.5
TARGET_PRECISION = 0.99   # threshold is tuned for max recall subject to this precision
URL_BLOCK_RISK = 70       # URL risk score at/above which a message is flagged outright
URL_WARN_RISK = 35        # at/above which it is marked suspicious
