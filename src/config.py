from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
POLICY_FILE = DATA_DIR / "aeromitra_policies.txt"
MODEL_NAME = "llama-3.3-70b-versatile"

ALLOWED_SENTIMENTS = ["positive", "neutral", "negative"]
ALLOWED_LANGUAGES = ["English", "Hindi", "Hinglish", "Other"]
ALLOWED_ISSUES = [
    "delay", "cancellation", "refund", "baggage", "staff",
    "safety_medical", "discrimination", "app_booking", "comfort_service",
    "praise", "query", "legal_media_threat", "other",
]
ALLOWED_SEVERITIES = ["none", "low", "medium", "high", "critical"]
ALLOWED_URGENCY = ["routine", "reply_needed", "urgent", "crisis"]

