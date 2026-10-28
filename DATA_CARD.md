# Data card

`synthetic_hinglish_posts.csv` is a fictional, human-defined dataset for an educational prototype. It contains English, Hindi, and Hinglish airline posts with issue, sentiment, severity, and escalation labels.

`synthetic_train.csv` and `synthetic_test.csv` are separate deterministic splits. The test split must remain untouched while prompts and thresholds are developed. Synthetic labels may not represent real customer behavior.

The official virality files are `virality_train.csv` and `virality_test.csv`, supplied separately. They contain 4,840 training rows and 1,159 test rows. The target is `spread`, and `retweets_24h` is outcome information, so it must not be used as a model feature.

