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

