import streamlit as st
import json
import base64
import re
import os
from prompt_engine import generate_experiment_plan

def sanitize_text(text):
    if not text or not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def safe_display(text, method=st.info):
    clean_text = sanitize_text(text)
    method(clean_text)

def extract_json(text):
    try:
        match = re.search(r"\{[\s\S]+\}", text)
        return match.group(0) if match else text
    except:
        return text

def remove_units_from_text(text, unit):
    if not text or not unit.strip():
        return text
    escaped_unit = re.escape(unit.strip())
    return re.sub(rf"(\d+\.?\d*)\s*{escaped_unit}", r"\1", text)

def insert_units_in_goal(text, unit):
    if not text or not unit.strip():
        return text

    unit = unit.strip()
    output = []
    tokens = re.split(r'(\s+|\d+\.?\d*)', text)  # keep whitespace

    for i, token in enumerate(tokens):
        if re.match(r'^\d+\.?\d*$', token):
            if unit in ["$", "₹", "€", "£"]:
                output.append(unit + token)
            else:
                output.append(token + " " + unit)
        else:
            output.append(token)
    return ''.join(output)

def section_with_info(title, key, explanation):
    col1, col2 = st.columns([10, 1])
    with col1:
        st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    with col2:
        toggle = st.toggle("ℹ️", key=f"explain_{key}", label_visibility="collapsed")
    if toggle:
        st.caption(explanation)

# --- Page Setup ---
st.set_page_config(page_title="A/B Test Architect", layout="wide")

# --- Onboarding Additions ---
st.markdown("""
<div style='padding: 20px 0;'>
  <h2 style='margin-bottom: 0;'>📊 Smarter A/B Test Planning in 2 Minutes</h2>
  <p style='font-size: 16px; color: #444;'>
    Struggling to write strong hypotheses or align on success criteria? This tool helps you instantly generate a full experiment PRD — from your product goal to clear variants, metrics, and risks — powered by LLMs.
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
**Steps:**
1. Provide product + metric context →  
2. Get AI-generated experiment plan →  
3. Select best hypothesis →  
4. Download PRD
""")

st.markdown("⏳ This will take **under 2 minutes** to fill.")

with st.expander("🔍 What will I get?"):
    st.markdown("""
    - ✅ Inferred Product Goal
    - 🧪 2–3 Actionable Hypotheses
    - 🔁 Clear Control vs Variant
    - 📊 MDE, Confidence, Sample Size
    - 📏 Custom Metrics
    - ⚠️ Risks & Segments
    - ✅ Downloadable PRD
    > Looks like this:
    """)
    st.code("## 🧪 Hypothesis: Showing price upfront\n...\n- MDE: 3.2%\n- Users per Variant: 5,000")

st.title("\U0001F9EA AI-Powered A/B Test Architect")
st.markdown("Use Groq + LLMs to design smarter experiments from fuzzy product goals.")

if st.button("\U0001F504 Start Over"):
    st.session_state.clear()
    st.rerun()


# --- Product Context ---
st.markdown("<div class='blue-section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>\U0001F9E0 Product Context</div>", unsafe_allow_html=True)
product_type = st.radio("Product Type *", ["SaaS", "Consumer App", "E-commerce", "Marketplace", "Gaming", "Other"], horizontal=True, help="What kind of product are you testing?")
user_base = st.radio("User Base Size (DAU) *", ["< 10K", "10K–100K", "100K–1M", "> 1M"], horizontal=True, help="Your product's average daily active users")
metric_focus = st.radio("Primary Metric Focus *", ["Activation", "Retention", "Monetization", "Engagement", "Virality"], horizontal=True, help="The key area you want to improve")
product_notes = st.text_area("Anything unique about your product or users?", placeholder="e.g. drop-off at pricing, seasonality, power users...", help="Optional context to inform better suggestions")
st.markdown("</div>", unsafe_allow_html=True)

# --- Metric Objective ---
st.markdown("<div class='blue-section'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>\U0001F3AF Metric Improvement Objective</div>", unsafe_allow_html=True)
exact_metric = st.text_input("\U0001F3AF Metric to Improve * (e.g. Activation Rate, ARPU, DAU/MAU)", help="Be specific — name the metric you want to shift")
metric_unit = st.text_input("\U0001F4C0 Metric Unit (e.g. %, $, secs, count)", value="%", help="How is the metric measured?")
col1, col2 = st.columns(2)
with col1:
    current_value_raw = st.text_input("\U0001F4C9 Current Metric Value *", help="Current observed value of the metric")
with col2:
    target_value_raw = st.text_input("\U0001F680 Target Metric Value *", help="What do you want the metric to reach?")
st.markdown("</div>", unsafe_allow_html=True)

# --- Generate Plan ---
st.markdown("<div class='green-section'>", unsafe_allow_html=True)
if st.button("Generate Plan") or "output" not in st.session_state:
    missing = []
    if not product_type: missing.append("Product Type")
    if not user_base: missing.append("User Base Size")
    if not metric_focus: missing.append("Primary Metric Focus")
    if not exact_metric.strip(): missing.append("Metric to Improve")
    if not current_value_raw.strip(): missing.append("Current Value")
    if not target_value_raw.strip(): missing.append("Target Value")
    if not metric_unit.strip(): missing.append("Metric Unit")

    if missing:
        st.warning("Please fill all required fields: " + ", ".join(missing))
        st.stop()

    try:
        current = float(current_value_raw)
        target = float(target_value_raw)
        expected_lift = round(target - current, 4)
        mde = round(abs((target - current) / current), 4) if current != 0 else 0.0
    except ValueError:
        st.error("Metric values must be numeric.")
        st.stop()

    goal_text = f"I want to improve {exact_metric} from {current} to {target}."
    goal_with_units = insert_units_in_goal(goal_text, metric_unit).strip()

    st.session_state.current = current
    st.session_state.target = target
    st.session_state.auto_goal = goal_with_units
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

    with st.spinner("\U0001F300 Generating your plan..."):
        output = generate_experiment_plan(goal_with_units, st.session_state.context)
    st.session_state.output = output
    st.session_state.hypothesis_confirmed = False
    st.session_state.selected_index = None
st.markdown("</div>", unsafe_allow_html=True)

# --- Display Output ---
if "output" in st.session_state:
    raw_output = extract_json(st.session_state.output)
    try:
        plan = json.loads(raw_output)
    except Exception as e:
        st.error(f"❌ Could not parse JSON: {e}")
        st.code(raw_output)
        st.stop()

    unit = " " + st.session_state.context.get("metric_unit", "").strip()
    st.markdown("<a name='output'></a>", unsafe_allow_html=True)
    st.markdown("<div class='green-section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📌 Inferred Product Goal</div>", unsafe_allow_html=True)
    safe_display(st.session_state.auto_goal)

    section_with_info("🧩 Problem Statement", "problem", 
        "Explains the gap between current and target metric values, and why this improvement matters.")
    problem_statement = plan.get("problem_statement", "")
    problem_statement = remove_units_from_text(problem_statement, unit)
    safe_display(problem_statement or "⚠️ Problem statement not generated by the model.")

    section_with_info("🧪 Hypotheses", "hypotheses", 
    "These are actionable, testable ideas likely to improve the metric. They’re short, specific, and informed by your product context.")
    hypotheses = plan.get("hypotheses", [])
    if not hypotheses:
        st.warning("No hypotheses found in the generated plan.")
    else:
        for i, h in enumerate(hypotheses):
            hypo = h.get("hypothesis") if isinstance(h, dict) else str(h)
            col1, col2 = st.columns([8, 1])
            with col1:
                st.markdown(f"**H{i+1}:** {hypo}")
            with col2:
                if st.button("Select", key=f"select_{i}"):
                    st.session_state.selected_index = i
                    st.session_state.hypothesis_confirmed = True
                    st.rerun()

    if st.session_state.get("hypothesis_confirmed") and st.session_state.selected_index is not None:
        i = st.session_state.selected_index
        selected_hypo = hypotheses[i].get("hypothesis", "N/A")
        rationale = sanitize_text(plan.get("hypothesis_rationale", [{}])[i].get("rationale", "N/A"))
        variant = plan.get("variants", [{}])[i]
        control = variant.get("control", "Not specified")
        variation = variant.get("variation", "Not specified")
        effort = plan.get("effort", [{}])[i].get("effort", "N/A")

        criteria = plan.get("success_criteria", {})
        try:
            confidence = float(criteria.get("confidence_level", 0))
            confidence_str = f"{round(confidence)}%" if confidence > 1 else f"{round(confidence * 100)}%"
        except:
            confidence_str = "N/A"
        sample_size = criteria.get("sample_size_required", "N/A")
        users_per_variant = criteria.get("users_per_variant", "N/A")
        duration = criteria.get("estimated_test_duration", "N/A")
        try:
            mde = float(criteria.get("MDE", 0))
            mde_display = f"{round(mde)}%" if mde > 1 else f"{round(mde * 100)}%"
        except:
            mde_display = "N/A"

        st.markdown("<div class='section-title'>✅ Selected Hypothesis</div>", unsafe_allow_html=True)
        st.code(selected_hypo)

        section_with_info("🔁 Variants", "variants", 
            "Shows what users in the control group vs. the variant group will see or experience.")
        st.markdown(f"- Control: {control}\n- Variation: {variation}")

        section_with_info("💡 Rationale", "rationale", 
            "Explains *why* the selected hypothesis is worth testing, based on user behavior or insight.")
        st.markdown(rationale)

        section_with_info("📊 Experiment Stats", "stats", 
            "Details like sample size, confidence level, MDE, and duration help determine how trustworthy and actionable your results will be.")
        st.markdown(f"""
- Confidence Level: {confidence_str}
- Minimum Detectable Effect (MDE): {mde_display}
- Sample Size Required: {sample_size}
- Users per Variant: {users_per_variant}
- Estimated Duration: {duration} days
- Estimated Effort: {effort}
""")

        metrics = plan.get("metrics", [])
        if metrics:
            st.markdown("<div class='section-title'>📏 Metrics</div>", unsafe_allow_html=True)
            for m in metrics:
                st.markdown(f"- **{m.get('name', 'Unnamed')}**: {m.get('formula', 'N/A')}")

        segments = plan.get("segments", [])
        if segments:
            st.markdown("<div class='section-title'>👥 Segments</div>", unsafe_allow_html=True)
            for s in segments:
                st.markdown(f"- {s}")

        risks = plan.get("risks", [])
        if risks:
            st.markdown("<div class='section-title'>⚠️ Risks</div>", unsafe_allow_html=True)
            for r in risks:
                st.markdown(f"- {r}")

        next_steps = plan.get("next_steps", [])
        if next_steps:
            st.markdown("<div class='section-title'>✅ Next Steps</div>", unsafe_allow_html=True)
            for step in next_steps:
                st.markdown(f"- {step}")

        prd = ""
        prd += "# 🧪 Experiment PRD\n\n"
        prd += "## 🎯 Goal\n"
        prd += st.session_state.auto_goal + "\n\n"

        prd += "## 🧩 Problem\n"
        prd += problem_statement + "\n\n"

        prd += "## 🧪 Hypothesis\n"
        prd += selected_hypo + "\n\n"

        prd += "## 🔁 Variants\n"
        prd += f"- Control: {control}\n- Variation: {variation}\n\n"

        prd += "## 💡 Rationale\n"
        prd += rationale + "\n\n"

        prd += "## 📊 Experiment Stats\n"
        prd += f"- Confidence Level: {confidence_str}\n"
        prd += f"- MDE: {mde_display}\n"
        prd += f"- Sample Size: {sample_size}\n"
        prd += f"- Users/Variant: {users_per_variant}\n"
        prd += f"- Duration: {duration} days\n"
        prd += f"- Effort: {effort}\n\n"

        metrics = plan.get("metrics", [])
        if metrics:
            prd += "## 📏 Metrics\n"
            for m in metrics:
                prd += f"- {m.get('name', 'Unnamed')}: {m.get('formula', 'N/A')}\n"

        segments = plan.get("segments", [])
        if segments:
            prd += "\n## 👥 Segments\n"
            for s in segments:
                prd += f"- {s}\n"

        risks = plan.get("risks_and_assumptions", [])  # ✅ Correct key
        if risks:
            prd += "\n## ⚠️ Risks\n"
            for r in risks:
                prd += f"- {r}\n"

        next_steps = plan.get("next_steps", [])
        if next_steps:
            prd += "\n## ✅ Next Steps\n"
            for step in next_steps:
                prd += f"- {step}\n"

        st.download_button("📄 Download PRD", prd, file_name="experiment_prd.txt")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<script>
    const params = new URLSearchParams(window.location.search);
    if (params.get("scroll") === "output") {
        const el = document.querySelector("a[name='output']");
        if (el) {
            el.scrollIntoView({ behavior: "smooth", block: "start" });
        }
    }
</script>
""", unsafe_allow_html=True)
