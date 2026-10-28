from pathlib import Path
import joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

CAT_FEATURES = ["language", "issue_type", "sentiment", "severity", "verified", "has_media", "hashtags", "tags_media_or_regulator", "hour_posted"]
NUM_FEATURES = ["author_followers", "account_age_days"]


def train(train_file: str | Path, model_file: str | Path, vectorizer_file: str | Path) -> dict:
    df = pd.read_csv(train_file)
    target = "spread" if "spread" in df else "spread_label"
    if target not in df:
        raise ValueError("Training data must contain spread or spread_label")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=15000)
    text_x = vectorizer.fit_transform(df["text"].fillna(""))
    encoder = OneHotEncoder(handle_unknown="ignore")
    cat_x = encoder.fit_transform(df[CAT_FEATURES].fillna("unknown").astype(str))
    scaler = StandardScaler()
    num_x = scaler.fit_transform(df[NUM_FEATURES].fillna(0).astype(float))
    x = hstack([text_x, cat_x, num_x])
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(x, df[target].astype(int))
    Path(model_file).parent.mkdir(exist_ok=True)
    joblib.dump(model, model_file)
    joblib.dump({"text": vectorizer, "categorical": encoder, "numeric": scaler}, vectorizer_file)
    return {"rows": len(df), "positive_rate": float(df[target].mean()), "target": target}


def predict_probability(text: str, model_file: str | Path, vectorizer_file: str | Path) -> float:
    model = joblib.load(model_file)
    bundle = joblib.load(vectorizer_file)
    row = pd.DataFrame([{"text": text, **{f: "unknown" for f in CAT_FEATURES}, **{f: 0 for f in NUM_FEATURES}}])
    text_x = bundle["text"].transform(row["text"])
    cat_x = bundle["categorical"].transform(row[CAT_FEATURES].astype(str))
    num_x = bundle["numeric"].transform(row[NUM_FEATURES].astype(float))
    return float(model.predict_proba(hstack([text_x, cat_x, num_x]))[0, 1])

