import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from src.llm import classify_post

parser = argparse.ArgumentParser()
parser.add_argument("--input", default=str(root / "data" / "synthetic_test.csv"))
parser.add_argument("--n", type=int, default=200)
args = parser.parse_args()

df = pd.read_csv(args.input).head(args.n)
rows, errors = [], []
for i, row in df.iterrows():
    try:
        pred = classify_post(str(row["text"]))
        record = {"post_id": row.get("post_id", i), "text": row["text"], "prediction": pred}
        for field in ["sentiment", "issue_type", "severity", "escalate"]:
            record[f"gold_{field}"] = row[field]
            record[f"pred_{field}"] = pred.get(field)
        rows.append(record)
    except Exception as exc:
        errors.append({"post_id": row.get("post_id", i), "error": str(exc)})

metrics = {"rows_requested": len(df), "rows_completed": len(rows), "api_errors": len(errors), "fields": {}}
for field in ["sentiment", "issue_type", "severity", "escalate"]:
    gold = [str(r[f"gold_{field}"]) for r in rows]
    pred = [str(r[f"pred_{field}"]) for r in rows]
    if gold:
        p, r, f, _ = precision_recall_fscore_support(gold, pred, average="macro", zero_division=0)
        metrics["fields"][field] = {"accuracy": round(float(accuracy_score(gold, pred)), 4), "macro_precision": round(float(p), 4), "macro_recall": round(float(r), 4), "macro_f1": round(float(f), 4), "confusion_matrix": confusion_matrix(gold, pred).tolist()}
(root / "reports").mkdir(exist_ok=True)
(root / "reports" / "llm_predictions.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
(root / "reports" / "llm_eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
(root / "reports" / "llm_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2))

