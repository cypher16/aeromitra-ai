import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.llm import classify_post, draft_reply
from src.pipeline import analyze, validate_draft
from src.virality import predict_probability, train as train_virality

st.set_page_config(page_title="AeroMitra AI", page_icon="✈️", layout="wide")
st.warning("Educational prototype. Not legal, financial, medical or PR advice. AeroMitra is a fictional airline. Nothing is published automatically.")
st.title("AeroMitra Social Listening")
st.caption("LLM classification, spread prediction, policy retrieval, and human-reviewed reply drafting")

demo_cases = {
    "Routine query": "What is the cabin baggage limit on AM480?",
    "Refund complaint": "Cancelled flight ka refund 30 din se pending hai. Har baar bolte hain 7 din me aayega.",
    "Crisis discrimination": "Wheelchair maangi toh staff ne bola pehle bolna chahiye tha, phir 3 ghante wait karaya. Disabled logon ke saath aisa?",
    "Sarcastic Hinglish": "Meeting miss ho gayi because of AM364 delay. Thanks AeroMitra, bahut bada kaam kiya 👏 #AeroMitraFail",
}

def get_spread_probability(text: str) -> float:
    model_file = ROOT / "models" / "virality_model.joblib"
    features_file = ROOT / "models" / "virality_features.joblib"
    if not model_file.exists() or not features_file.exists():
        train_virality(ROOT / "data" / "virality_train.csv", model_file, features_file)
    return predict_probability(text, model_file, features_file)

tab_inbox, tab_queue, tab_report = st.tabs(["Inbox", "Approval Queue", "Model Report"])

with tab_inbox:
    selected = st.selectbox("Demo case", ["Custom post", *demo_cases.keys()])
    text = st.text_area("Social media post", value="" if selected == "Custom post" else demo_cases[selected], height=120)
    if st.button("Analyze post", type="primary", disabled=not text.strip()):
        try:
            with st.spinner("Classifying and retrieving policy..."):
                classification = classify_post(text)
                spread_probability = get_spread_probability(text)
                result = analyze(text, classification, spread_probability)
            st.session_state["analysis"] = result
            st.session_state["post"] = text
        except Exception as exc:
            st.error(str(exc))

    result = st.session_state.get("analysis")
    if result:
        c = result["classification"]
        cols = st.columns(5)
        cols[0].metric("Sentiment", c.get("sentiment", "unknown"))
        cols[1].metric("Issue", c.get("issue_type", "unknown"))
        cols[2].metric("Spread probability", f"{result['spread_probability']:.0%}")
        cols[3].metric("Crisis score", result["crisis_score"])
        cols[4].metric("Priority", result["priority"])
        st.json(c)
        st.subheader("Retrieved policy")
        for p in result["policies"]:
            st.write(f"**{p['title']}**")
            st.caption(p["text"])
        if result["priority"] == "P1":
            st.error("CRISIS ALERT: human review required. No automatic reply is allowed.")
        elif st.button("Draft policy-grounded reply"):
            try:
                raw = draft_reply(st.session_state["post"], c, result["policy_context"], result["priority"])
                st.session_state["reply"] = validate_draft(raw, result["priority"])
            except Exception as exc:
                st.error(str(exc))
        if st.session_state.get("reply"):
            st.subheader("Draft reply")
            st.text_area("Human review", st.session_state["reply"]["draft_reply"], height=100)
            st.write(st.session_state["reply"])

with tab_queue:
    st.subheader("Approval queue")
    st.info("This prototype never publishes. A human must Approve, Edit, or Reject every draft.")
    if st.session_state.get("reply"):
        st.write(st.session_state["reply"]["draft_reply"] or "No automatic draft for this case.")
        st.button("Approve for simulated posting", disabled=not st.session_state["reply"]["reply_allowed"])
        st.button("Reject")
    else:
        st.write("No draft in the queue.")

with tab_report:
    st.subheader("Evaluation and limitations")
    metrics_path = ROOT / "reports" / "model_metrics.json"
    if metrics_path.exists():
        st.json(json.loads(metrics_path.read_text(encoding="utf-8")))
    else:
        st.info("Run the training and evaluation scripts to populate metrics.")
    for title, filename in [("Virality metrics", "virality_metrics.json"), ("LLM evaluation metrics", "llm_eval_metrics.json")]:
        st.subheader(title)
        path = ROOT / "reports" / filename
        if path.exists():
            st.json(json.loads(path.read_text(encoding="utf-8")))
        else:
            st.info(f"{filename} is not available yet.")
    st.markdown("Synthetic labels and LLM results are educational measurements, not production performance.")

