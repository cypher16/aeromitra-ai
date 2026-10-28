# Prompt log

| Date | Prompt or change | Result | What went wrong |
|---|---|---|---|
| 2026-10-28 | Codex scaffolded the AeroMitra prototype from the build guide. | Created the project structure, guardrails, scoring logic, policy retrieval, Streamlit shell, and separate synthetic train/test files. | The bundled runtime lacked scikit-learn, so the split script was changed to use pandas only. |

