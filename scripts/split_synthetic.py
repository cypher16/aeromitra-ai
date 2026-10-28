from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root / "data" / "synthetic_hinglish_posts.csv")
parts = []
for _, group in df.groupby("sentiment", sort=True):
    parts.append(group.sample(frac=0.70, random_state=42))
train = pd.concat(parts).sort_index()
test = df.drop(train.index).sort_index()
train.to_csv(root / "data" / "synthetic_train.csv", index=False)
test.to_csv(root / "data" / "synthetic_test.csv", index=False)
print({"train": len(train), "test": len(test)})

