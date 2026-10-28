from pathlib import Path
import json
import sys
import joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from src.virality import CAT_FEATURES, NUM_FEATURES

train_df = pd.read_csv(root / "data" / "virality_train.csv")
test_df = pd.read_csv(root / "data" / "virality_test.csv")
y_train = train_df["spread"].astype(int)
y_test = test_df["spread"].astype(int)

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

models = {
    "majority_baseline": DummyClassifier(strategy="most_frequent"),
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "linear_svm": CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=42), cv=3),
    "sgd_log_loss": SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=1000, random_state=42),
}
metrics = {"train_rows": len(train_df), "test_rows": len(test_df), "target": "spread", "selection_rule": "highest recall among models with precision >= 0.25; tie-break by F1", "models": {}}
fitted = {}
for name, model in models.items():
    model.fit(x_train, y_train)
    fitted[name] = model
    pred = model.predict(x_test)
    probability = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None
    p, r, f, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
    metrics["models"][name] = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f), 4),
        "average_precision": round(float(average_precision_score(y_test, probability)), 4) if probability is not None else None,
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }

candidates = [(name, row) for name, row in metrics["models"].items() if name != "majority_baseline" and row["precision"] >= 0.25]
winner, _ = max(candidates or [("logistic_regression", metrics["models"]["logistic_regression"])], key=lambda item: (item[1]["recall"], item[1]["f1"]))
metrics["selected_model"] = winner
model_dir = root / "models"
model_dir.mkdir(exist_ok=True)
joblib.dump(fitted[winner], model_dir / "virality_model.joblib")
joblib.dump({"text": text_vectorizer, "categorical": encoder, "numeric": scaler}, model_dir / "virality_features.joblib")
(root / "reports").mkdir(exist_ok=True)
(root / "reports" / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2))

