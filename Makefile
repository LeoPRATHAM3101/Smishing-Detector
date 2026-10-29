setup:   ; python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt
train:   ; python -m src.train
compare: ; python -m src.compare_models
test:    ; pytest -q
app:     ; streamlit run app.py
api:     ; uvicorn src.api:app --reload
