from src.scoring import crisis_score, priority

def test_p1_safety_override():
    assert priority(10, "safety_medical", False, False) == "P1"

def test_crisis_score_increases_with_spread():
    assert crisis_score("high", 0.9, "negative") > crisis_score("high", 0.1, "negative")

