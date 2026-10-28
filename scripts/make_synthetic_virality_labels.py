from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
for name in ["synthetic_train.csv", "synthetic_test.csv"]:
    path = root / "data" / name
    df = pd.read_csv(path)
    negative = df["sentiment"].eq("negative").astype(int)
    high_risk_issue = df["issue_type"].isin(["discrimination", "safety_medical", "cancellation", "refund"]).astype(int)
    hashtag = df["text"].str.contains(r"#|viral|RT|boycott", case=False, regex=True).astype(int)
    # This is synthetic and intentionally transparent; it is not real virality evidence.
    df["spread_label"] = ((negative + high_risk_issue + hashtag) >= 2).astype(int)
    df.to_csv(path, index=False)
print("Added transparent synthetic spread labels.")

