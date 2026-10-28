# AeroMitra Project Brief

AeroMitra is a fictional Indian airline. The prototype classifies social posts, predicts spread, calculates a crisis score, retrieves relevant policy text, and drafts a reply. Nothing is published without human approval.

Stack: Python, Streamlit, Groq with the current `openai/gpt-oss-120b` model, scikit-learn, and ChromaDB.

The supplied `virality_train.csv` and `virality_test.csv` are the official virality-model datasets. The supplied `synthetic_hinglish_posts.csv` remains the LLM evaluation dataset. Results must be labelled synthetic and must not be presented as production performance. The `spread` column is the virality target; `retweets_24h` must not be used as an input feature because it is outcome information.

