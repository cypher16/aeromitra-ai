from pathlib import Path
import sys
import json
import joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, average_precision_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.virality import CAT_FEATURES, NUM_FEATURES

root = Path(__file__).resolve().parents[1]
train = pd.read_csv(root / "data" / "virality_train.csv")
test = pd.read_csv(root / "data" / "virality_test.csv")
bundle = joblib.load(root / "models" / "virality_features.joblib")
model = joblib.load(root / "models" / "virality_model.joblib")

def features(df):
    return hstack([bundle["text"].transform(df["text"].fillna("")), bundle["categorical"].transform(df[CAT_FEATURES].fillna("unknown").astype(str)), bundle["numeric"].transform(df[NUM_FEATURES].fillna(0).astype(float))])

probability = model.predict_proba(features(test))[:, 1]
y = test["spread"].astype(int).to_numpy()
thresholds = []
for threshold in [i / 100 for i in range(20, 81, 5)]:
    pred = (probability >= threshold).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    thresholds.append({"threshold": threshold, "precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f), 4)})
pred = (probability >= 0.5).astype(int)
p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
metrics = {"train_rows": len(train), "test_rows": len(test), "positive_rate": round(float(y.mean()), 4), "precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f), 4), "average_precision": round(float(average_precision_score(y, probability)), 4), "confusion_matrix": confusion_matrix(y, pred).tolist(), "thresholds": thresholds}
(root / "reports" / "virality_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2))

