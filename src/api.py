from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .predict import analyze, load_bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_bundle()  # fail fast if the model is missing
    yield


app = FastAPI(title="Smishing Detector API", version="1.0.0", lifespan=lifespan)


class Message(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


class Batch(BaseModel):
    messages: list[str] = Field(min_length=1, max_length=100)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_one(msg: Message):
    return analyze(msg.text)


@app.post("/predict/batch")
def predict_batch(batch: Batch):
    return {"results": [analyze(m[:1000]) for m in batch.messages]}
