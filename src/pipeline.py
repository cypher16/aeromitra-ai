from .guardrails import validate_reply
from .rag import format_policy_context, retrieve_policies
from .scoring import crisis_score, priority


def analyze(text: str, classifier: dict, spread_probability: float = 0.5) -> dict:
    score = crisis_score(
        classifier.get("severity", "none"), spread_probability,
        classifier.get("sentiment", "neutral"), classifier.get("sarcasm", False),
    )
    prio = priority(score, classifier.get("issue_type", "other"), classifier.get("legal_media_threat", False), classifier.get("escalate", False))
    policies = retrieve_policies(text)
    return {
        "classification": classifier,
        "spread_probability": spread_probability,
        "crisis_score": score,
        "priority": prio,
        "policies": policies,
        "policy_context": format_policy_context(policies),
    }


def validate_draft(result: dict, priority_value: str) -> dict:
    draft = result.get("draft_reply", "") if priority_value != "P1" else ""
    warnings = validate_reply(draft)
    return {
        "draft_reply": draft,
        "reply_allowed": priority_value != "P1" and not warnings,
        "human_review_required": True,
        "policy_sources": result.get("policy_sources", []),
        "validation_warnings": warnings,
    }

