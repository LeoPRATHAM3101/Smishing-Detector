import pytest

from src import predict
from src.data import load_data
from src.train import make_bundle


@pytest.fixture(scope="session")
def small_bundle():
    """Tiny model trained on india_samples.csv only (no network needed)."""
    df = load_data(include_uci=False)
    bundle, _ = make_bundle(df.text, df.label, folds=3)
    predict._bundle = bundle
    return bundle
