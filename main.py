import streamlit as st
import json
import re
from prompt_engine import generate_experiment_plan

# --- Utility Functions ---
def sanitize_text(text):
    if not text or not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.replace("\n", " ").replace("\r", " ").replace("\t", " ")).strip()

def safe_display(text, method=st.info):
    method(sanitize_text(text))

def extract_json(text):
    try:
        match = re.search(r"\{[\s\S]+\}", text)
        return match.group(0) if match else text
    except:
        return text

def remove_units_from_text(text, unit):
    if not text or not unit.strip():
        return text
    return re.sub(rf"(\d+\.?\d*)\s*{re.escape(unit.strip())}", r"\1", text)

# --- Page Setup ---
st.set_page_config(page_title="A/B Test Architect", layout="wide")
st.title("🧪 AI-Powered A/B Test Architect")
st.markdown("Use Groq + LLMs to design smarter experiments from fuzzy product goals.")

# --- Input: Product Context ---
st.header("🧠 Product Context")
product_type = st.radio("Product Type *", ["SaaS", "Consumer App", "E-commerce", "Marketplace", "Gaming", "Other"], horizontal=True)
user_base = st.radio("User Base Size (DAU) *", ["< 10K", "10K–100K", "100K–1M", "> 1M"], horizontal=True)
metric_focus = st.radio("Primary Metric Focus *", ["Activation", "Retention", "Monetization", "Engagement", "Virality"], horizontal=True)
product_notes = st.text_area("Anything unique about your product or users?", placeholder="e.g. drop-off at pricing, seasonality, power users...")

# --- Input: Metric Goal ---
st.markdown("## 🎯 Metric Improvement Objective")
exact_metric = st.text_input("🎯 Metric to Improve * (e.g. Activation Rate, ARPU, DAU/MAU)")
metric_unit = st.text_input("📐 Metric Unit (e.g. %, $, secs, count)", value="%")
current_value_raw = st.text_input("📉 Current Metric Value * (numerical only)")
target_value_raw = st.text_input("🚀 Target Metric Value * (numerical only)")

# --- Generate Plan ---
if st.button("Generate Plan") or "output" not in st.session_state:
    required = [product_type, user_base, metric_focus, exact_metric, current_value_raw, target_value_raw, metric_unit]
    if not all(map(str.strip, required)):
        st.warning("Please fill all required fields.")
        st.stop()

    try:
        current = float(current_value_raw)
        target = float(target_value_raw)
        expected_lift = round(target - current, 4)
        mde = round(abs((target - current) / current), 4) if current != 0 else 0.0
    except ValueError:
        st.error("Metric values must be numeric.")
        st.stop()

    # Save State
    st.session_state.current = current
    st.session_state.target = target
    st.session_state.auto_goal = f"I want to improve {exact_metric} from {current} to {target}."
    st.session_state.context = {
        "type": product_type,
        "users": user_base,
        "metric": metric_focus,
        "notes": product_notes,
        "exact_metric": exact_metric,
        "current_value": current,
        "target_value": target,
        "expected_lift": expected_lift,
        "minimum_detectable_effect": round(mde * 100, 2),
        "metric_unit": metric_unit.strip()
    }

    st.session_state.output = generate_experiment_plan(st.session_state.auto_goal, st.session_state.context)
    st.session_state.hypothesis_confirmed = False
    st.session_state.selected_index = None

# --- Display Plan Output ---
if "output" in st.session_state:
    raw_output = extract_json(st.session_state.output)

    try:
        plan = json.loads(raw_output)
    except Exception as e:
        st.error(f"❌ Could not parse JSON: {e}")
        st.code(raw_output)
        st.stop()

    unit = " " + st.session_state.context.get("metric_unit", "").strip()
    safe_display(st.session_state.auto_goal)

    st.subheader("🧩 Problem Statement")
    problem_statement = plan.get("problem_statement", "")
    safe_display(remove_units_from_text(problem_statement, unit) or "⚠️ Problem statement not generated.")

    st.subheader("🧪 Choose a Hypothesis")
    hypotheses = plan.get("hypotheses", [])
    if not hypotheses:
        st.warning("No hypotheses found.")
    else:
        for i, h in enumerate(hypotheses):
            hypo = h.get("hypothesis", str(h))
            with st.expander(f"H{i+1}: {hypo}", expanded=(st.session_state.get("selected_index") == i)):
                if st.button(f"✅ Select H{i+1}", key=f"select_{i}"):
                    st.session_state.selected_index = i
                    st.session_state.hypothesis_confirmed = True
                    st.rerun()

# --- Show Detailed Output After Selection ---
if st.session_state.get("hypothesis_confirmed") and st.session_state.selected_index is not None:
    i = st.session_state.selected_index
    selected_hypo_obj = hypotheses[i] if i < len(hypotheses) else {}
    selected_hypo = selected_hypo_obj.get("hypothesis", "N/A")

    effort = plan.get("effort", [{}])[i].get("effort", "N/A")
    variant = plan.get("variants", [{}])[i] if i < len(plan.get("variants", [])) else {}
    control = variant.get("control", "Not specified")
    variation = variant.get("variation", "Not specified")

    rationale = plan.get("hypothesis_rationale", [{}])[i]
    rationale = rationale.get("rationale", rationale) if isinstance(rationale, dict) else rationale
    rationale = sanitize_text(rationale)

    teams = plan.get("team_involved", [])
    criteria = plan.get("success_criteria", {})
    metrics = plan.get("metrics", [])
    segments = plan.get("segments", [])
    risks = [sanitize_text(r) for r in plan.get("risks_and_assumptions", []) if isinstance(r, str) and r.strip()]
    steps = [sanitize_text(s) for s in plan.get("next_steps", []) if s.strip()]

    try:
        conf_level = float(criteria.get("confidence_level", 0))
        conf_display = f"{round(conf_level * 100)}%" if conf_level <= 1 else f"{round(conf_level)}%"
    except:
        conf_display = "N/A"

    try:
        expected_lift = float(criteria.get("expected_lift", 0))
        expected_lift_str = f"{expected_lift}{unit}"
    except:
        expected_lift_str = "N/A"

    try:
        mde = float(criteria.get("MDE", 0))
        mde_display = f"{round(mde * 100)}%" if mde <= 1 else f"{round(mde)}%"
    except:
        mde_display = "N/A"

    
