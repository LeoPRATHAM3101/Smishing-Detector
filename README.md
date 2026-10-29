# 🛡️ Smishing / Spam SMS Detector (India-focused)

Hybrid detector for scam SMS: **TF-IDF (word + char n-grams) + hand-crafted features + Logistic Regression**,
combined with **rule-based URL analysis** (shorteners, raw IPs, odd TLDs, `.apk` links, punycode, brand look-alikes
such as `hdfcbamk-secure.com`). Served via **Streamlit** (demo) and **FastAPI** (API), with Docker and CI.

## Quick start (Mac / VS Code)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m src.train            # downloads UCI data, tunes threshold, saves model + reports/
python -m src.compare_models   # model comparison table -> reports/model_comparison.md
pytest -q
streamlit run app.py           # demo UI
uvicorn src.api:app --reload   # API docs at http://127.0.0.1:8000/docs
```
Always run from the project root with `python -m ...`.

## How the verdict works
- **ML score** from the pipeline; the decision threshold is tuned on out-of-fold predictions for maximum recall
  subject to precision >= 0.99 (a false alarm on a real bank OTP is costly).
- **URL risk (0-100)**; >= 70 flags the message even if the ML score is low.
- Labels: `SCAM`, `SUSPICIOUS`, `LEGIT`, plus a list of human-readable red flags.

## API
```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"text":"Your SBI KYC expired. Update: bit.ly/abc"}'
```
Endpoints: `GET /health`, `POST /predict`, `POST /predict/batch` (max 100 messages).

## Docker
```bash
docker build -t smishing-detector .
docker run -p 8000:8000 smishing-detector
```

## Results
Run `python -m src.train` and `python -m src.compare_models`, then paste the numbers from `reports/metrics.json` and
`reports/model_comparison.md` here, along with `reports/confusion_matrix.png`.

## Limitations (be upfront about these in your write-up)
- UCI SMS Spam Collection is old and UK/Singapore-centred; `data/india_samples.csv` is a small, author-written
  illustrative set (70 messages), **not** a benchmark. Grow it with real, anonymised messages.
- URL rules use a hand-written brand list; no live domain-age or reputation lookups.
- Not a replacement for operator-level filtering. Report scams at cybercrime.gov.in or call 1930.
