SEVERITY_POINTS = {"none": 0, "low": 25, "medium": 60, "high": 100, "critical": 100}
NEGATIVITY_POINTS = {"positive": 0, "neutral": 30, "negative": 100}


def crisis_score(severity: str, spread_probability: float, sentiment: str, sarcasm: bool = False) -> float:
    negativity = "negative" if sarcasm else sentiment
    return round(
        0.5 * SEVERITY_POINTS.get(severity, 0)
        + 0.3 * spread_probability * 100
        + 0.2 * NEGATIVITY_POINTS.get(negativity, 0),
        1,
    )


def priority(score: float, issue_type: str, legal_media_threat: bool, escalate: bool) -> str:
    if score >= 70 or issue_type in {"safety_medical", "discrimination"} or legal_media_threat:
        return "P1"
    if score >= 40 or escalate:
        return "P2"
    return "P3"

