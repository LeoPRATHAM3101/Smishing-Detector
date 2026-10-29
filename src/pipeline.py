from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .features import extract_features, normalize


def word_tfidf():
    return TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def char_tfidf():
    return TfidfVectorizer(preprocessor=normalize, analyzer="char_wb", ngram_range=(2, 5),
                           min_df=3, sublinear_tf=True)


def hand_features():
    return Pipeline([("extract", FunctionTransformer(extract_features)),
                     ("scale", StandardScaler(with_mean=False))])


def build_pipeline(clf=None) -> Pipeline:
    """Word + char TF-IDF + hand-crafted features -> classifier (default: Logistic Regression)."""
    features = FeatureUnion([("word", word_tfidf()), ("char", char_tfidf()), ("hand", hand_features())])
    clf = clf or LogisticRegression(C=10, class_weight="balanced", max_iter=3000)
    return Pipeline([("features", features), ("clf", clf)])


def feature_names(pipe: Pipeline):
    from .features import FEATURE_NAMES
    names = []
    for name, tr in pipe.named_steps["features"].transformer_list:
        if name == "hand":
            names += [f"hand:{n}" for n in FEATURE_NAMES]
        else:
            names += [f"{name}:{n}" for n in tr.get_feature_names_out()]
    return names
