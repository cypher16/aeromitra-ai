# AeroMitra AI

Educational prototype for social listening and crisis response for a fictional airline.
Nothing is published automatically. Every response requires human approval.

## Local run

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Create `.streamlit/secrets.toml` locally from `.streamlit/secrets.toml.example` and add a rotated Groq key. Never commit it.

## Build and test

```powershell
python scripts\train_virality.py
python scripts\evaluate_virality.py
pytest -q
```

The app trains the virality model on first use if model artifacts are absent. ChromaDB is the default in-process policy vector store. Qdrant is optional and requires a Qdrant URL and API key.

