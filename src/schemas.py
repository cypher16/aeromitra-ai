from typing import Literal
from pydantic import BaseModel, Field


class ClassificationResult(BaseModel):
    language: Literal["English", "Hindi", "Hinglish", "Other"]
    sentiment: Literal["positive", "neutral", "negative"]
    issue_type: str
    severity: Literal["none", "low", "medium", "high", "critical"]
    urgency: Literal["routine", "reply_needed", "urgent", "crisis"]
    escalate: bool
    legal_media_threat: bool = False
    sarcasm: bool = False
    confidence: float = Field(ge=0, le=1)
    reason: str


class ReplyResult(BaseModel):
    draft_reply: str = ""
    reply_allowed: bool
    human_review_required: bool
    policy_sources: list[str] = []
    validation_warnings: list[str] = []

