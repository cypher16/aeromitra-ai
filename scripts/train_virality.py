from pathlib import Path
import sys
import json
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.virality import CAT_FEATURES, NUM_FEATURES, train

root = Path(__file__).resolve().parents[1]
train_file = root / "data" / "virality_train.csv"
test_file = root / "data" / "virality_test.csv"
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
target = "spread"

text_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=15000)
text_train = text_vectorizer.fit_transform(train_df["text"].fillna(""))
text_test = text_vectorizer.transform(test_df["text"].fillna(""))
encoder = OneHotEncoder(handle_unknown="ignore")
cat_train = encoder.fit_transform(train_df[CAT_FEATURES].fillna("unknown").astype(str))
cat_test = encoder.transform(test_df[CAT_FEATURES].fillna("unknown").astype(str))
scaler = StandardScaler()
num_train = scaler.fit_transform(train_df[NUM_FEATURES].fillna(0).astype(float))
num_test = scaler.transform(test_df[NUM_FEATURES].fillna(0).astype(float))
x_train = hstack([text_train, cat_train, num_train])
x_test = hstack([text_test, cat_test, num_test])
y_train = train_df[target].astype(int)
y_test = test_df[target].astype(int)

models = {
    "majority_baseline": DummyClassifier(strategy="most_frequent"),
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
}
metrics = {"train_rows": len(train_df), "test_rows": len(test_df), "target": target, "models": {}}
for name, model in models.items():
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    p, r, f, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
    metrics["models"][name] = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }

train(train_file, root / "models" / "virality_model.joblib", root / "models" / "virality_features.joblib")
(root / "reports").mkdir(exist_ok=True)
(root / "reports" / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2))

