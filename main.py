# main.py — AI A/B Test Architect (polished UI + robust parsing)
import streamlit as st
import json
import re
import os
from typing import Any, Dict, Optional, Tuple
from prompt_engine import generate_experiment_plan
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize
import numpy as np
import hashlib

# -------------------------
# Helpers / Utilities
# -------------------------
def create_header_with_help(header_text: str, help_text: str, icon: str = "🔗"):
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="font-size: 1.5rem;">{icon}</div>
                <div class="section-title" style="margin-bottom: 0;">{header_text}</div>
            </div>
            <span style="font-size: 0.95rem; color: #666; cursor: help; float: right;" title="{help_text}">❓</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sanitize_text(text: Any) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_display(text: Any, method=st.info):
    method(sanitize_text(text))


def _extract_json_first_braces(text: str) -> Optional[str]:
    """
    Attempt to extract the most plausible JSON object substring from text by finding first '{' and matching '}'.
    This handles cases where the model returns extra commentary before/after JSON.
    """
    if not isinstance(text, str):
        return None
    # Try <json> ... </json> tags first
    tag_match = re.search(r"<json>([\s\S]+?)</json>", text, re.IGNORECASE)
    if tag_match:
        return tag_match.group(1).strip()

    # Try to find the first `{` and find a matching `}` by scanning.
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json(text: Any) -> Optional[Dict]:
    """
    Robust attempt to parse JSON from a variety of LLM outputs.
    Returns a dict on success, otherwise None and shows errors in the UI.
    """
    if text is None:
        st.error("No output returned from LLM.")
        return None

    if isinstance(text, dict):
        return text

    if isinstance(text, list):
        st.error("LLM returned a JSON list when an object was expected.")
        return None

    # Convert to string
    try:
        raw = str(text)
    except Exception as e:
        st.error(f"Unexpected LLM output type: {e}")
        return None

    # Try direct json.loads first
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
        else:
            st.error("Parsed JSON is not an object.")
            return None
    except Exception:
        pass

    # Try to extract a JSON substring heuristically
    candidate = _extract_json_first_braces(raw)
    if candidate:
        # Clean common artifacts
        candidate_clean = re.sub(r',\s*,', ',', candidate)  # double commas
        candidate_clean = re.sub(r',\s*\}', '}', candidate_clean)
        candidate_clean = re.sub(r',\s*\]', ']', candidate_clean)
        # Remove leading/trailing code fences
        candidate_clean = re.sub(r"^```(?:json)?", "", candidate_clean).strip()
        candidate_clean = re.sub(r"```$", "", candidate_clean).strip()

        try:
            parsed = json.loads(candidate_clean)
            if isinstance(parsed, dict):
                return parsed
            else:
                st.error("Extracted JSON is not an object.")
                st.code(candidate_clean[:1000] + ("..." if len(candidate_clean) > 1000 else ""))
                return None
        except json.JSONDecodeError as e:
            st.error(f"Could not parse JSON from LLM output: {e}")
            st.code(candidate_clean[:1000] + ("..." if len(candidate_clean) > 1000 else ""))
            return None

    # Last resort: show raw output and let the user edit
    st.error("LLM output could not be parsed as JSON. Please inspect or edit the raw output below.")
    try:
        st.code(raw[:2000] + ("..." if len(raw) > 2000 else ""))
    except Exception:
        st.write("LLM output could not be displayed.")
    return None


def post_process_llm_text(text: Any, unit: str) -> str:
    if text is None:
        return ""
    s = sanitize_text(text)
    if unit == "%":
        s = s.replace("%%", "%")
        s = re.sub(r"\s+%", "%", s)
    return s


def format_value_with_unit(value: Any, unit: str) -> str:
    try:
        # keep numeric formatting tidy
        if isinstance(value, (int, float)):
            if float(value).is_integer():
                v_str = str(int(value))
            else:
                v_str = str(round(float(value), 4)).rstrip("0").rstrip(".")
        else:
            v_str = str(value)
    except Exception:
        v_str = str(value)

    units_with_space = ["USD", "count", "minutes", "hours", "days"]
    if unit in units_with_space:
        return f"{v_str} {unit}"
    else:
        return f"{v_str}{unit}"


def calculate_sample_size(
    baseline, mde, alpha, power, num_variants, metric_type, std_dev=None
) -> Tuple[Optional[int], Optional[int]]:
    """
    Returns (sample_size_per_variant, total_sample_size) or (None, None) on error.
    baseline: numeric baseline (if percent metric, pass percent value, e.g. 5 for 5%)
    mde: percent (e.g. 5 for 5%)
    alpha: significance level (e.g. 0.05)
    power: desired power as fraction (e.g. 0.8)
    num_variants: integer (number of variants including control)
    metric_type: 'Conversion Rate' or 'Numeric Value'
    std_dev: required for numeric
    """
    try:
        if baseline is None or mde is None:
            return None, None

        mde_relative = float(mde) / 100.0

        if metric_type == "Conversion Rate":
            baseline_prop = float(baseline) / 100.0
            expected_prop = baseline_prop * (1 + mde_relative)
            if baseline_prop <= 0:
                st.warning("Baseline conversion must be > 0 for reliable calculations.")
                return None, None
            if expected_prop >= 1:
                expected_prop = 0.999

            effect_size = proportion_effectsize(baseline_prop, expected_prop)
            if effect_size == 0:
                return None, None

            analysis = NormalIndPower()
            sample_size_per_variant = analysis.solve_power(
                effect_size=effect_size, alpha=alpha, power=power, alternative="two-sided"
            )

        elif metric_type == "Numeric Value":
            if std_dev is None or float(std_dev) == 0:
                st.error("Standard deviation is required and must be non-zero for numeric metrics.")
                return None, None
            mde_absolute = float(baseline) * mde_relative
            effect_size = mde_absolute / float(std_dev)
            if effect_size == 0:
                return None, None
            analysis = TTestIndPower()
            sample_size_per_variant = analysis.solve_power(
                effect_size=effect_size, alpha=alpha, power=power, alternative="two-sided"
            )
        else:
            st.error("Unknown metric type for sample size calculation.")
            return None, None

        if sample_size_per_variant is None or sample_size_per_variant <= 0 or not np.isfinite(sample_size_per_variant):
            return None, None

        total = sample_size_per_variant * num_variants
        return int(np.ceil(sample_size_per_variant)), int(np.ceil(total))
    except Exception as e:
        st.error(f"Error calculating sample size: {e}")
        return None, None


# -------------------------
# Page Setup & Styling
# -------------------------
st.set_page_config(page_title="A/B Test Architect", layout="wide")
st.markdown(
    """
<style>
.blue-section {background-color: #f6f9ff; padding: 18px; border-radius: 8px; margin-bottom: 18px;}
.green-section {background-color: #f7fff7; padding: 18px; border-radius: 8px; margin-bottom: 18px;}
.section-title {font-size: 1.1rem; font-weight: 600; color: #1E90FF; margin-bottom: 8px;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("💡 A/B Test Architect — AI-assisted experiment PRD generator")
st.markdown("Create experiment PRDs, hypotheses, stats, and sample-size guidance — faster and with guardrails.")

# -------------------------
# Initialize session state defaults
# -------------------------
if "output" not in st.session_state:
    st.session_state.output = None
if "ai_parsed" not in st.session_state:
    st.session_state.ai_parsed = None
if "calc_locked" not in st.session_state:
    st.session_state.calc_locked = False
if "locked_stats" not in st.session_state:
    st.session_state.locked_stats = {}
if "selected_index" not in st.session_state:
    st.session_state.selected_index = None
if "hypothesis_confirmed" not in st.session_state:
    st.session_state.hypothesis_confirmed = False
if "last_llm_hash" not in st.session_state:
    st.session_state.last_llm_hash = None

# -------------------------
# INPUTS: Business Context (left column priority)
# -------------------------
st.markdown("<div class='blue-section'>", unsafe_allow_html=True)
create_header_with_help(
    "Product Context",
    "Provide the product context and business goal so the AI can produce a focused experiment plan.",
    icon="💡",
)
col_a, col_b = st.columns([2, 1])
with col_a:
    product_type = st.selectbox(
        "Product Type",
        ["SaaS", "Consumer App", "E-commerce", "Marketplace", "Gaming", "Other"],
        index=0,
        help="What kind of product are you testing?",
    )
    user_base_choice = st.selectbox(
        "User Base Size (DAU)",
        ["< 10K", "10K–100K", "100K–1M", "> 1M"],
        index=0,
        help="Average daily active users for the product.",
    )
    metric_focus = st.selectbox(
        "Primary Metric Focus",
        ["Activation", "Retention", "Monetization", "Engagement", "Virality"],
        index=0,
        help="The general category of metrics you're trying to move.",
    )
    product_notes = st.text_area(
        "Anything unique about your product or users? (optional)",
        placeholder="e.g. seasonality, power users, drop-off at pricing",
        help="Optional context to inform better suggestions.",
    )
with col_b:
    strategic_goal = st.text_area(
        "High-Level Business Goal *",
        placeholder="e.g., Increase overall revenue from our premium tier",
        help="This is the broader business goal the experiment supports.",
    )
    user_persona = st.text_input(
        "Target User Persona (optional)",
        placeholder="e.g., First-time users from India, iOS users, power users",
        help="Focus the plan on a specific user segment.",
    )

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# INPUTS: Metric Details
# -------------------------
st.markdown("<div class='blue-section'>", unsafe_allow_html=True)
create_header_with_help(
    "Metric Improvement Objective",
    "Provide the exact metric and current vs target values. Use the proper units.",
    icon="🎯",
)
col_m1, col_m2 = st.columns([2, 2])
with col_m1:
    exact_metric = st.text_input(
        "Metric to Improve * (e.g. Activation Rate, ARPU, DAU/MAU)",
        help="Be specific — name the metric you want to shift.",
    )
with col_m2:
    metric_type = st.radio("Metric Type", ["Conversion Rate", "Numeric Value"], horizontal=True)
col_unit, col_values = st.columns([1, 2])
with col_unit:
    metric_unit = st.selectbox(
        "Metric Unit", ["%", "USD", "minutes", "count", "other"], index=0, help="Choose the unit for clarity."
    )
with col_values:
    # Use number inputs to avoid string parsing errors
    if metric_unit == "%":
        current_value = st.number_input("Current Metric Value *", min_value=0.0, step=0.01, format="%.2f")
        target_value = st.number_input("Target Metric Value *", min_value=0.0, step=0.01, format="%.2f")
    else:
        current_value = st.number_input("Current Metric Value *", min_value=0.0, step=0.01, format="%.2f")
        target_value = st.number_input("Target Metric Value *", min_value=0.0, step=0.01, format="%.2f")

    std_dev = None
    if metric_type == "Numeric Value":
        std_dev = st.number_input(
            "Standard Deviation of Metric * (required for numeric metrics)",
            min_value=0.0,
            step=0.01,
            format="%.4f",
            help="The standard deviation is required for numeric metrics to compute sample sizes.",
        )

# Validation: prevent equal current and target
metric_inputs_valid = True
if current_value == target_value:
    st.warning("The target metric must be different from the current metric to measure change. Please adjust one or the other.")
    metric_inputs_valid = False

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# GENERATE PLAN AREA — green section
# -------------------------
st.markdown("<div class='green-section'>", unsafe_allow_html=True)
create_header_with_help("Generate Experiment Plan", "When ready, click Generate to call the LLM and create a plan.", icon="🧠")

# Build goal string (human readable)
formatted_current = format_value_with_unit(current_value, metric_unit)
formatted_target = format_value_with_unit(target_value, metric_unit)
sanitized_metric_name = sanitize_text(exact_metric)
goal_with_units = f"I want to improve {sanitized_metric_name} from {formatted_current} to {formatted_target}."

# Generate button is disabled until required inputs are present and valid
required_ok = all(
    [
        product_type,
        user_base_choice,
        metric_focus,
        sanitized_metric_name,
        metric_inputs_valid,
        strategic_goal,
    ]
)
generate_btn = st.button("Generate Plan", disabled=not required_ok)

if generate_btn:
    # Compose the context dict the LLM expects
    context = {
        "type": product_type,
        "users": user_base_choice,
        "metric": metric_focus,
        "notes": product_notes,
        "exact_metric": sanitized_metric_name,
        "current_value": current_value,
        "target_value": target_value,
        "expected_lift": round(((target_value - current_value) / current_value) * 100, 2) if current_value != 0 else 0.0,
        "minimum_detectable_effect": round(abs((target_value - current_value) / current_value) * 100, 2) if current_value != 0 else 0.0,
        "metric_unit": metric_unit,
        "strategic_goal": strategic_goal,
        "user_persona": user_persona,
        "metric_type": metric_type,
        "std_dev": std_dev,
    }

    # Call the LLM and parse safely
    with st.spinner("Generating your plan..."):
        try:
            raw_llm = generate_experiment_plan(goal_with_units, context)
            # store raw string for inspection
            st.session_state.output = raw_llm if raw_llm is not None else ""
            parsed = extract_json(raw_llm)
            st.session_state.ai_parsed = parsed
            # compute hash of the LLM output for caching/traceability
            try:
                h = hashlib.sha256((str(raw_llm) or "").encode("utf-8")).hexdigest()
                st.session_state.last_llm_hash = h
            except Exception:
                st.session_state.last_llm_hash = None
            if parsed:
                st.success("Plan generated successfully — review and edit below.")
            else:
                st.warning("Plan generated but parsing failed — edit the raw output to correct JSON or try regenerate.")
        except Exception as e:
            st.error(f"LLM generation failed: {e}")
            st.session_state.output = ""
            st.session_state.ai_parsed = None

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Calculator (Sample size) Section — shown when there's a plan or context
# -------------------------
if st.session_state.get("ai_parsed") is not None or st.session_state.get("output"):
    st.markdown("<hr>", unsafe_allow_html=True)
    with st.expander("🔢 A/B Test Calculator: Fine-tune sample size", expanded=True):
        # Prefill calculator values from session or defaults
        calc_mde = st.session_state.get("calc_mde", st.session_state.get("ai_parsed", {}).get("success_criteria", {}).get("MDE", 5.0) if st.session_state.get("ai_parsed") else 5.0)
        calc_conf = st.session_state.get("calc_confidence", 95)
        calc_power = st.session_state.get("calc_power", 80)
        calc_variants = st.session_state.get("calc_variants", 2)

        col1, col2 = st.columns(2)
        with col1:
            calc_mde = st.number_input("Minimum Detectable Effect (MDE) %", min_value=0.1, max_value=50.0, value=float(calc_mde), step=0.1)
            calc_conf = st.number_input("Confidence Level (%)", min_value=80, max_value=99, value=int(calc_conf), step=1)
        with col2:
            calc_power = st.number_input("Statistical Power (%)", min_value=70, max_value=95, value=int(calc_power), step=1)
            calc_variants = st.number_input("Number of Variants (Control + Variations)", min_value=2, max_value=5, value=int(calc_variants), step=1)

        if metric_type == "Numeric Value":
            if std_dev is not None:
                st.info(f"Standard Deviation pre-filled: {std_dev}")

        refresh_btn = st.button("Refresh Calculator")
        lock_btn = st.button("Lock Values for Plan")

        if refresh_btn or st.session_state.get("calculate_now", False):
            st.session_state.calculate_now = False
            st.session_state.last_calc_mde = float(calc_mde)
            st.session_state.last_calc_confidence = int(calc_conf)
            st.session_state.last_calc_power = int(calc_power)
            st.session_state.last_calc_variants = int(calc_variants)

            alpha_calc = 1 - (st.session_state.last_calc_confidence / 100.0)
            power_calc = st.session_state.last_calc_power / 100.0

            sample_per_variant, total_sample = calculate_sample_size(
                baseline=current_value,
                mde=st.session_state.last_calc_mde,
                alpha=alpha_calc,
                power=power_calc,
                num_variants=st.session_state.last_calc_variants,
                metric_type=metric_type,
                std_dev=std_dev,
            )

            st.session_state.calculated_sample_size_per_variant = sample_per_variant
            st.session_state.calculated_total_sample_size = total_sample

            # map user_base_choice to numeric DAU estimate
            dau_map = {"< 10K": 5000, "10K–100K": 50000, "100K–1M": 500000, "> 1M": 2000000}
            dau = dau_map.get(user_base_choice, 10000)
            users_to_test = st.session_state.calculated_total_sample_size or 0
            st.session_state.calculated_duration_days = (users_to_test / dau) if dau > 0 else float("inf")

        # Display results if present
        if st.session_state.get("calculated_sample_size_per_variant") and st.session_state.get("calculated_total_sample_size"):
            st.markdown("---")
            st.metric("Users Per Variant", f"{st.session_state.calculated_sample_size_per_variant:,} users")
            st.metric("Total Sample Size", f"{st.session_state.calculated_total_sample_size:,} users")
            st.metric("Estimated Test Duration", f"{st.session_state.calculated_duration_days:,.0f} days")
            st.caption("Assumes all DAU are eligible and evenly split across variants.")
        else:
            st.info("Click 'Refresh Calculator' to compute sample size with current inputs.")

        if lock_btn:
            if st.session_state.get("calculated_sample_size_per_variant") is not None:
                st.session_state.calc_locked = True
                st.session_state.locked_stats = {
                    "confidence_level": st.session_state.last_calc_confidence,
                    "MDE": st.session_state.last_calc_mde,
                    "sample_size_required": f"{st.session_state.calculated_total_sample_size:,} users",
                    "users_per_variant": f"{st.session_state.calculated_sample_size_per_variant:,} users",
                    "estimated_test_duration": f"{st.session_state.calculated_duration_days:,.0f} days",
                }
                st.success("Calculator values locked and will be used in the final plan.")
            else:
                st.error("Cannot lock values. Run the calculator first.")

st.markdown("<hr>", unsafe_allow_html=True)

# -------------------------
# DISPLAY AI-GENERATED PLAN (editable + final view)
# -------------------------
if st.session_state.get("ai_parsed") is None and st.session_state.get("output"):
    st.info("AI returned output but parsing failed. Edit raw output in the Raw JSON tab to fix the parse or try regenerating.")

if st.session_state.get("ai_parsed"):
    plan = st.session_state.ai_parsed
    unit = st.session_state.get("context", {}).get("metric_unit", metric_unit) if st.session_state.get("context") else metric_unit

    st.markdown("<div class='green-section'>", unsafe_allow_html=True)
    create_header_with_help("Inferred Product Goal", "The AI's interpretation of your goal. Edit if needed.", icon="🎯")
    safe_display(post_process_llm_text(goal_with_units, unit))

    create_header_with_help("Problem Statement", "Clear description of the gap and why it matters.", icon="🧩")
    problem_statement = post_process_llm_text(plan.get("problem_statement", ""), unit)
    st.text_area("Problem Statement", value=problem_statement, key="editable_problem", height=120)

    create_header_with_help("Hypotheses", "Editable list of testable hypotheses.", icon="🧪")
    hypotheses = plan.get("hypotheses", [])
    if not hypotheses or not isinstance(hypotheses, list):
        st.warning("No hypotheses found in the generated plan.")
        hypotheses = []

    # show hypotheses with Select buttons
    for i, h in enumerate(hypotheses):
        if isinstance(h, dict):
            hypo_text = sanitize_text(h.get("hypothesis") or h.get("description") or json.dumps(h))
        else:
            hypo_text = sanitize_text(h)
        cols = st.columns([9, 1])
        cols[0].write(f"**H{i+1}:** {hypo_text}")
        if cols[1].button("Select", key=f"select_{i}"):
            st.session_state.selected_index = i
            st.session_state.hypothesis_confirmed = True
            st.session_state.calc_locked = False
            # try to pull MDE suggested by LLM
            try:
                llm_mde = plan.get("success_criteria", {}).get("MDE", st.session_state.get("calc_mde", 5.0))
                st.session_state.calc_mde = float(llm_mde)
            except Exception:
                pass
            st.rerun()

    # If a hypothesis has been selected, present detailed editable fields
    if st.session_state.get("hypothesis_confirmed") and st.session_state.get("selected_index") is not None:
        idx = st.session_state.selected_index
        if idx < 0 or idx >= len(hypotheses):
            st.error("Selected hypothesis index out of range.")
        else:
            # Normalize objects
            h_obj = hypotheses[idx] if isinstance(hypotheses[idx], dict) else {"hypothesis": hypotheses[idx]}
            selected_hypo = sanitize_text(h_obj.get("hypothesis", "N/A"))

            # Rationale
            raw_rationales = plan.get("hypothesis_rationale", [])
            rationale = "N/A"
            if isinstance(raw_rationales, list) and idx < len(raw_rationales):
                r_item = raw_rationales[idx]
                if isinstance(r_item, dict):
                    rationale = sanitize_text(r_item.get("rationale", "N/A"))
                else:
                    rationale = sanitize_text(r_item)

            # variants
            raw_variants = plan.get("variants", [])
            control = "Not specified"
            variation = "Not specified"
            if isinstance(raw_variants, list) and idx < len(raw_variants):
                v = raw_variants[idx]
                if isinstance(v, dict):
                    control = sanitize_text(v.get("control", "Not specified"))
                    variation = sanitize_text(v.get("variation", "Not specified"))
                else:
                    variation = sanitize_text(v)

            # effort
            raw_efforts = plan.get("effort", [])
            effort_display = "N/A"
            if isinstance(raw_efforts, list) and idx < len(raw_efforts):
                e = raw_efforts[idx]
                if isinstance(e, dict):
                    effort_display = sanitize_text(e.get("effort", "N/A"))
                else:
                    effort_display = sanitize_text(e)

            # Criteria (may use locked stats)
            if st.session_state.get("calc_locked", False):
                criteria_display = st.session_state.get("locked_stats", {})
            else:
                criteria_display = plan.get("success_criteria", {})

            if not criteria_display:
                criteria_display = {}

            statistical_rationale_display = (
                criteria_display.get("statistical_rationale")
                or plan.get("statistical_rationale")
                or plan.get("success_criteria", {}).get("statistical_rationale")
                or "The experiment is designed with a specified MDE, confidence level, and power to detect meaningful changes."
            )

            # Confidence formatting
            try:
                confidence_raw = criteria_display.get("confidence_level", 0)
                confidence = float(confidence_raw)
                confidence_str = f"{round(confidence)}%" if confidence > 1 else f"{round(confidence * 100)}%"
            except Exception:
                confidence_str = "N/A"

            sample_size = criteria_display.get("sample_size_required", "N/A")
            users_per_variant = criteria_display.get("users_per_variant", "N/A")
            duration = criteria_display.get("estimated_test_duration", "N/A")

            try:
                mde_val = float(criteria_display.get("MDE", 0))
                mde_display = f"{mde_val}%"
            except Exception:
                mde_display = "N/A"

            # Show details and allow edits for these fields
            create_header_with_help("Selected Hypothesis", "This is the hypothesis you selected from the generated options.", icon="🧪")
            st.text_area("Hypothesis", value=selected_hypo, key="editable_hypothesis", height=100)

            create_header_with_help("Variants", "Control vs Variation", icon="🔁")
            st.text_input("Control (editable)", value=control, key="editable_control")
            st.text_input("Variation (editable)", value=variation, key="editable_variation")

            create_header_with_help("Rationale", "Why this hypothesis is worth testing", icon="💡")
            st.text_area("Rationale (editable)", value=rationale, key="editable_rationale", height=120)

            create_header_with_help("Experiment Stats", "Review/lock the final experiment stats used in the PRD", icon="📊")
            if not st.session_state.get("calc_locked", False):
                st.warning("Adjust the calculator above and Lock values for the plan to finalize experiment stats.")
            st.markdown(
                f"""
- Confidence Level: **{confidence_str}**
- Minimum Detectable Effect (MDE): **{mde_display}**
- Sample Size Required: **{sample_size}**
- Users per Variant: **{users_per_variant}**
- Estimated Duration: **{duration}**
- Estimated Effort: **{effort_display}**
**Statistical Rationale:** {statistical_rationale_display}
"""
            )

            # Metrics / segments / risks / next steps — editable areas
            metrics = plan.get("metrics", [])
            if metrics:
                create_header_with_help("Metrics", "Primary and secondary metrics", icon="📏")
                # show each metric as editable JSON text area
                for mi, m in enumerate(metrics):
                    st.text_area(f"Metric {mi+1}", value=json.dumps(m, ensure_ascii=False), key=f"metric_{mi}", height=80)

            segments = plan.get("segments", [])
            if segments:
                create_header_with_help("Segments", "User segments for analysis", icon="👥")
                st.text_area("Segments (one per line)", value="\n".join(segments), key="editable_segments", height=80)

            risks = plan.get("risks_and_assumptions", [])
            if risks:
                create_header_with_help("Risks & Assumptions", "What could impact test outcomes", icon="⚠️")
                st.text_area("Risks (one per line)", value="\n".join(risks), key="editable_risks", height=80)

            next_steps = plan.get("next_steps", [])
            if next_steps:
                create_header_with_help("Next Steps", "Actionable tasks to start the experiment", icon="✅")
                st.text_area("Next Steps (one per line)", value="\n".join(next_steps), key="editable_next_steps", height=80)

            # Build final PRD string
            prd_parts = []
            prd_parts.append("# 🧪 Experiment PRD\n")
            prd_parts.append("## 🎯 Goal\n")
            prd_parts.append(goal_with_units + "\n\n")
            prd_parts.append("## 🧩 Problem\n")
            prd_parts.append(st.session_state.get("editable_problem", problem_statement) + "\n\n")
            prd_parts.append("## 🧪 Hypothesis\n")
            prd_parts.append(st.session_state.get("editable_hypothesis", selected_hypo) + "\n\n")
            prd_parts.append("## 🔁 Variants\n")
            prd_parts.append(f"- Control: {st.session_state.get('editable_control', control)}\n- Variation: {st.session_state.get('editable_variation', variation)}\n\n")
            prd_parts.append("## 💡 Rationale\n")
            prd_parts.append(st.session_state.get("editable_rationale", rationale) + "\n\n")
            prd_parts.append("## 📊 Experiment Stats\n")
            prd_parts.append(f"- Confidence Level: {confidence_str}\n")
            prd_parts.append(f"- MDE: {mde_display}\n")
            prd_parts.append(f"- Sample Size: {sample_size}\n")
            prd_parts.append(f"- Users/Variant: {users_per_variant}\n")
            prd_parts.append(f"- Duration: {duration}\n")
            prd_parts.append(f"- Effort: {effort_display}\n")
            prd_parts.append(f"- Statistical Rationale: {statistical_rationale_display}\n\n")

            # Append metrics, segments, risks, next steps if present (allow edited values)
            if metrics:
                prd_parts.append("## 📏 Metrics\n")
                for mi, m in enumerate(metrics):
                    # try to use edited value if present
                    edited = st.session_state.get(f"metric_{mi}")
                    if edited:
                        try:
                            parsed_m = json.loads(edited)
                            prd_parts.append(f"- {parsed_m.get('name', 'Unnamed')}: {parsed_m.get('formula','N/A')}\n")
                        except:
                            prd_parts.append(f"- {sanitize_text(edited)}\n")
                    else:
                        if isinstance(m, dict):
                            prd_parts.append(f"- {m.get('name','Unnamed')}: {m.get('formula','N/A')}\n")
                        else:
                            prd_parts.append(f"- {sanitize_text(m)}\n")

            if st.session_state.get("editable_segments"):
                prd_parts.append("\n## 👥 Segments\n")
                for s in st.session_state.get("editable_segments", "").splitlines():
                    prd_parts.append(f"- {s}\n")

            if st.session_state.get("editable_risks"):
                prd_parts.append("\n## ⚠️ Risks\n")
                for r in st.session_state.get("editable_risks", "").splitlines():
                    prd_parts.append(f"- {r}\n")

            if st.session_state.get("editable_next_steps"):
                prd_parts.append("\n## ✅ Next Steps\n")
                for ns in st.session_state.get("editable_next_steps", "").splitlines():
                    prd_parts.append(f"- {ns}\n")

            prd_text = "\n".join(prd_parts)

            st.download_button("📄 Download PRD (.txt)", prd_text, file_name="experiment_prd.txt")
            st.download_button("📥 Download PRD (.json)", json.dumps(plan, indent=2, ensure_ascii=False), file_name="experiment_plan.json")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Raw LLM Output / Manual Repair Tab (if parsing failed)
# -------------------------
if st.session_state.get("output") and not st.session_state.get("ai_parsed"):
    st.markdown("<div class='blue-section'>", unsafe_allow_html=True)
    create_header_with_help("Raw LLM Output (fix JSON here)", "When parsing fails you'll see the raw LLM output — edit it then click Parse JSON.", icon="🛠️")
    raw_edit = st.text_area("Raw LLM output / edit here", value=st.session_state.get("output", ""), height=400, key="raw_llm_edit")
    if st.button("Parse JSON"):
        parsed_try = extract_json(st.session_state.get("raw_llm_edit", raw_edit))
        if parsed_try:
            st.session_state.ai_parsed = parsed_try
            st.success("Manual parse succeeded — plan is now usable.")
            st.rerun()
        else:
            st.error("Manual parse failed — edit the text and try again.")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Footer / Debug info
# -------------------------
st.markdown("<hr>", unsafe_allow_html=True)
with st.expander("⚙️ Debug & Trace"):
    st.write("Last LLM hash:", st.session_state.get("last_llm_hash"))
    st.write("AI parsed present:", bool(st.session_state.get("ai_parsed")))
    st.write("Raw output length:", len(st.session_state.get("output") or ""))
    if st.button("Clear session state"):
        keys_to_clear = ["output", "ai_parsed", "calculated_sample_size_per_variant", "calculated_total_sample_size", "calculated_duration_days", "locked_stats", "calc_locked", "selected_index", "hypothesis_confirmed", "last_llm_hash"]
        for k in keys_to_clear:
            if k in st.session_state:
                del st.session_state[k]
        st.success("Session cleared. Reloading...")
        st.rerun()
