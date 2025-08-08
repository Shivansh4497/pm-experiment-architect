import streamlit as st
import json
import base64
import re
import os
from prompt_engine import generate_experiment_plan
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize
import numpy as np
from typing import Tuple, Optional, Any, Dict

# --- Helper Functions ---
def create_header_with_help(header_text, help_text, icon="🔗"):
    """
    Creates a consistent header with a custom icon, title, and a help tooltip.
    """
    st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="font-size: 1.5rem;">{icon}</div>
                <div class="section-title" style="margin-bottom: 0;">{header_text}</div>
            </div>
            <span style="font-size: 1rem; color: #888; cursor: help; float: right;" title="{help_text}">❓</span>
        </div>
    """, unsafe_allow_html=True)

def sanitize_text(text: Any) -> str:
    """Sanitizes text to prevent injection issues and clean up whitespace."""
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            text = str(text)
        except:
            return ""
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def safe_display(text: Any, method=st.info):
    """Displays sanitized text using a Streamlit method."""
    clean_text = sanitize_text(text)
    method(clean_text)

def extract_json(text: Any) -> Optional[Dict]:
    """
    Extracts and attempts to fix a JSON object from a string that may contain other text.
    Accepts:
      - dict (returns as-is)
      - json string (attempts to find {...} within and parse)
    Returns parsed dict or None (and shows error in UI).
    """
    if text is None:
        st.error("No output from plan generator.")
        return None

    # if it's already a dict, return
    if isinstance(text, dict):
        return text

    # if it's a list and contains a dict at top-level, wrap into dict if possible
    if isinstance(text, list):
        st.error("Plan generator returned a JSON list when a JSON object was expected.")
        return None

    if not isinstance(text, str):
        # try converting to string then parse
        try:
            text = str(text)
        except Exception as e:
            st.error(f"Unexpected output type from plan generator: {e}")
            return None

    # attempt to pull a JSON object from the string
    try:
        match = re.search(r"\{[\s\S]+\}$", text) or re.search(r"\{[\s\S]+\}", text)
        json_str = match.group(0) if match else text

        # remove common LLM artifacts
        json_str = re.sub(r'[\r\n]+', ' ', json_str)
        json_str = re.sub(r',\s*\}', '}', json_str)
        json_str = re.sub(r',\s*\]', ']', json_str)

        parsed = json.loads(json_str)
        if not isinstance(parsed, dict):
            st.error("Parsed JSON is not an object.")
            return None
        return parsed
    except Exception as e:
        st.error(f"❌ Could not parse JSON from LLM output: {e}")
        # show a short preview for debugging (avoid huge dumps)
        try:
            st.code(text[:1000] + ("..." if len(text) > 1000 else ""))
        except:
            st.write("LLM output could not be displayed.")
        return None

def post_process_llm_text(text: Any, unit: str) -> str:
    """Removes double units and ensures proper spacing."""
    if text is None:
        return ""
    text = sanitize_text(text)
    if not unit:
        return text
    # Fix double percentage signs
    if unit == "%":
        text = text.replace("%%", "%")
    # remove double spaces before unit patterns like "50  %"
    text = re.sub(r'\s+%','%', text)
    return text

# NEW: Helper function to format value and unit correctly
def format_value_with_unit(value: Any, unit: str) -> str:
    """Adds a space for certain units, but not for others like % or currency symbols."""
    try:
        # preserve original formatting if already a string with unit
        if isinstance(value, str) and unit in value:
            return value
        # keep numeric values with at most 4 significant digits for readability
        if isinstance(value, (int, float)):
            # If it's effectively an integer, show without decimal
            if float(value).is_integer():
                v_str = str(int(value))
            else:
                v_str = str(round(float(value), 4)).rstrip('0').rstrip('.') if '.' in str(value) else str(value)
        else:
            v_str = str(value)
    except:
        v_str = str(value)

    # List of units that should have a space before them
    units_with_space = ["USD", "count", "minutes", "hours", "days"]

    if unit in units_with_space:
        return f"{v_str} {unit}"
    else:
        # Assumes units like %, $, etc. should be directly attached
        return f"{v_str}{unit}"

def calculate_sample_size(baseline, mde, alpha, power, num_variants, metric_type, std_dev=None) -> Tuple[Optional[int], Optional[int]]:
    """
    Calculates the required sample size for an A/B test based on metric type.
    Returns: (sample_size_per_variant, total_sample_size) as integers or (None, None) on error.
    """
    try:
        # basic validation
        if baseline is None:
            st.error("Baseline must be provided for sample size calculation.")
            return None, None
        if mde is None:
            st.error("MDE must be provided for sample size calculation.")
            return None, None

        mde_relative = float(mde) / 100.0

        if metric_type == 'Conversion Rate':
            baseline_prop = float(baseline) / 100.0
            expected_prop = baseline_prop * (1 + mde_relative)
            if baseline_prop <= 0 or expected_prop <= 0:
                st.warning("Baseline conversion rate must be > 0.")
                return None, None
            if expected_prop >= 1.0:
                expected_prop = 0.999

            effect_size = proportion_effectsize(baseline_prop, expected_prop)
            if effect_size == 0:
                st.warning("MDE must be > 0 for a meaningful calculation.")
                return None, None

            analysis = NormalIndPower()
            sample_size_per_variant = analysis.solve_power(
                effect_size=effect_size,
                alpha=alpha,
                power=power,
                alternative='two-sided'
            )
        elif metric_type == 'Numeric Value':
            if std_dev is None or float(std_dev) == 0:
                st.error("Standard deviation is required and must be non-zero for numeric metrics.")
                return None, None
            mde_absolute = float(baseline) * mde_relative
            effect_size = mde_absolute / float(std_dev)

            if effect_size == 0:
                st.warning("MDE must be > 0 for a meaningful calculation.")
                return None, None

            analysis = TTestIndPower()
            sample_size_per_variant = analysis.solve_power(
                effect_size=effect_size,
                alpha=alpha,
                power=power,
                alternative='two-sided'
            )
        else:
            st.error("Invalid metric type selected.")
            return None, None

        if sample_size_per_variant is None or sample_size_per_variant <= 0 or not np.isfinite(sample_size_per_variant):
            return None, None

        total_sample_size = sample_size_per_variant * num_variants
        return int(np.ceil(sample_size_per_variant)), int(np.ceil(total_sample_size))

    except Exception as e:
        st.error(f"Error in sample size calculation: {e}")
        return None, None

# --- Page Setup ---
st.set_page_config(page_title="A/B Test Architect", layout="wide")

st.markdown("""
<style>
.blue-section {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}
.green-section {
    background-color: #f0fff4;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}
.section-title {
    font-size: 1.5rem;
    font-weight: bold;
    color: #1E90FF;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)


# --- Onboarding Additions ---
st.markdown("""
<div style='padding: 20px 0;'>
  <h2 style='margin-bottom: 0;'>📊 Smarter A/B Test Planning in 2 Minutes</h2>
  <p style='font-size: 16px; color: #444;'>
    Struggling to write strong hypotheses or align on success criteria? This tool helps you instantly generate a full experiment PRD — from your product goal to clear variants, metrics, and risks — powered by LLMs.
  </p></div>
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

st.title("💡 A/B Test Architect")
st.markdown("Use Groq + LLMs to design smarter experiments from fuzzy product goals.")

if st.button("🔄 Start Over"):
    st.session_state.clear()
    st.rerun()

# --- Product Context ---
st.markdown("<div class='blue-section'>", unsafe_allow_html=True)
create_header_with_help("Product Context", "This section provides the LLM with information about your product to generate a more relevant and accurate plan.", icon="💡")
product_type = st.radio("Product Type *", ["SaaS", "Consumer App", "E-commerce", "Marketplace", "Gaming", "Other"], horizontal=True, help="What kind of product are you testing?")
user_base = st.radio("User Base Size (DAU) *", ["< 10K", "10K–100K", "100K–1M", "> 1M"], horizontal=True, help="Your product's average daily active users")
metric_focus = st.radio("Primary Metric Focus *", ["Activation", "Retention", "Monetization", "Engagement", "Virality"], horizontal=True, help="The key area you want to improve")
product_notes = st.text_area("Anything unique about your product or users?", placeholder="e.g. drop-off at pricing, seasonality, power users...", help="Optional context to inform better suggestions")
strategic_goal = st.text_area("High-Level Business Goal *", placeholder="e.g., Increase overall revenue from our premium tier", help="The broader business objective this experiment supports.")
user_persona = st.text_input("Target User Persona (optional)", placeholder="e.g., First-time users from India, iOS users, power users", help="Focus the plan on a specific user segment.")
st.markdown("</div>", unsafe_allow_html=True)

# --- Metric Objective ---
st.markdown("<div class='blue-section'>", unsafe_allow_html=True)
create_header_with_help("Metric Improvement Objective", "Provide the specific metric you want to impact and its current vs. target value.", icon="🎯")
exact_metric = st.text_input("Metric to Improve * (e.g. Activation Rate, ARPU, DAU/MAU)", help="Be specific — name the metric you want to shift")
col_metric_type, col_metric_unit = st.columns(2)
with col_metric_type:
    metric_type = st.radio("Metric Type", ["Conversion Rate", "Numeric Value"], horizontal=True, help="Is this a percentage/proportion or a continuous number?")
with col_metric_unit:
    metric_unit = st.text_input("Metric Unit", value="%", help="How is the metric measured? E.g., 'USD', 'minutes', 'count', or '%'")

col1, col2 = st.columns(2)
with col1:
    current_value_raw = st.text_input("Current Metric Value *", help="Current observed value of the metric")
with col2:
    target_value_raw = st.text_input("Target Metric Value *", help="What do you want the metric to reach?")

std_dev_raw = None
if metric_type == "Numeric Value":
    std_dev_raw = st.text_input("Standard Deviation of Metric *", placeholder="e.g., 2.5", help="The standard deviation is crucial for calculating sample size for numeric metrics.")

st.markdown("</div>", unsafe_allow_html=True)

# --- Generate Plan ---
st.markdown("<div class='green-section'>", unsafe_allow_html=True)
if st.button("Generate Plan"):
    missing = []
    if not product_type: missing.append("Product Type")
    if not user_base: missing.append("User Base Size")
    if not metric_focus: missing.append("Primary Metric Focus")
    if not exact_metric or not exact_metric.strip(): missing.append("Metric to Improve")
    if not current_value_raw or not current_value_raw.strip(): missing.append("Current Value")
    if not target_value_raw or not target_value_raw.strip(): missing.append("Target Value")
    if not metric_unit or not metric_unit.strip(): missing.append("Metric Unit")
    if metric_type == "Numeric Value" and (std_dev_raw is None or not std_dev_raw.strip()): missing.append("Standard Deviation")

    invalid_chars = ['"', '{', '}', '[', ']', '$', '₹', '€', '£']
    if any(char in metric_unit for char in invalid_chars):
        st.error("The 'Metric Unit' contains invalid characters. Please use plain text like 'USD' or 'count' instead of symbols like '$' or brackets.")
    elif missing:
        st.warning("Please fill all required fields: " + ", ".join(missing))
    else:
        try:
            # Sanitize inputs by removing any percentage signs before converting to float
            current = float(current_value_raw.replace('%', '').strip())
            target = float(target_value_raw.replace('%', '').strip())
            std_dev = float(std_dev_raw.replace('%', '').strip()) if std_dev_raw else None

            sanitized_metric_name = exact_metric.replace('%', '').strip()

            if current == 0 and metric_type == "Conversion Rate":
                st.error("Current value cannot be zero for conversion rate lift calculation.")
            elif current == 0 and metric_type == "Numeric Value" and (std_dev is None or std_dev == 0):
                st.error("Current value or standard deviation cannot be zero for numeric metric calculation.")
            else:
                expected_lift = round(((target - current) / current) * 100, 2) if current != 0 else 0.0
                mde_percent = round(abs((target - current) / current) * 100, 2) if current != 0 else 0.0

                # Use the helper to format values with units for human-readable goal
                formatted_current = format_value_with_unit(current, metric_unit)
                formatted_target = format_value_with_unit(target, metric_unit)
                goal_with_units = f"I want to improve {sanitized_metric_name} from {formatted_current} to {formatted_target}."

                # Build session context
                st.session_state.current = current
                st.session_state.target = target
                st.session_state.auto_goal = goal_with_units
                st.session_state.context = {
                    "type": product_type,
                    "users": user_base,
                    "metric": metric_focus,
                    "notes": product_notes,
                    "exact_metric": sanitized_metric_name,
                    "current_value": current,
                    "target_value": target,
                    "expected_lift": expected_lift,
                    "minimum_detectable_effect": mde_percent,
                    "metric_unit": metric_unit.strip(),
                    "strategic_goal": strategic_goal,
                    "user_persona": user_persona,
                    "metric_type": metric_type,
                    "std_dev": std_dev
                }
                # reset calculation/status flags
                st.session_state.stats_locked = False
                st.session_state.calculate_now = True
                st.session_state.hypothesis_confirmed = False
                st.session_state.selected_index = None

                # Call the plan generator in a safe way
                with st.spinner("🧠 Generating your plan..."):
                    try:
                        raw_output = generate_experiment_plan(goal_with_units, st.session_state.context)
                        # If the generator returns a dict, convert to string for storage consistency
                        if isinstance(raw_output, dict):
                            output_str = json.dumps(raw_output)
                        else:
                            output_str = str(raw_output) if raw_output is not None else ""
                    except Exception as e:
                        st.error(f"LLM generation failed: {e}")
                        output_str = ""

                        # fall back to a minimal safe plan dictionary
                    if not output_str:
                        fallback_plan = {
                            "problem_statement": f"The product aims to improve {sanitized_metric_name} from {current}{metric_unit} to {target}{metric_unit}, which supports the strategic goal. If unchanged, the product risks missing target outcomes.",
                            "hypotheses": [
                                {"hypothesis": f"Show explicit {sanitized_metric_name} benefit upfront to new users", "description": "Surface benefit to increase user intent"},
                                {"hypothesis": f"Simplify flow to reduce friction in the {sanitized_metric_name} funnel", "description": "Lower friction should increase conversions"}
                            ],
                            "variants": [
                                {"hypothesis": "Show explicit benefit", "control": "Current flow", "variation": "Add benefit messaging on signup"},
                                {"hypothesis": "Reduce friction", "control": "Current flow", "variation": "Remove one step in funnel"}
                            ],
                            "metrics": [
                                {"name": sanitized_metric_name, "formula": f"{sanitized_metric_name} / Eligible users"},
                                {"name": "Secondary metric", "formula": "Engagement / Users"}
                            ],
                            "segments": [user_persona or "All users"],
                            "success_criteria": {"confidence_level": 95, "expected_lift": expected_lift, "MDE": mde_percent, "estimated_test_duration": 7, "statistical_rationale": "Fallback statistical rationale."},
                            "effort": [{"hypothesis": "Show explicit benefit", "effort": "Low"}, {"hypothesis": "Reduce friction", "effort": "Medium"}],
                            "team_involved": ["Design", "Data", "Backend"],
                            "hypothesis_rationale": [{"rationale": "Based on user drop-off in funnel; surface benefit to improve intent."}, {"rationale": "Reduce steps to increase completion rate."}],
                            "risks_and_assumptions": ["Users may not notice messaging", "Traffic quality varies"],
                            "next_steps": ["Prioritize hypothesis", "Implement variation tracking"]
                        }
                        output_str = json.dumps(fallback_plan)

                    st.session_state.output = output_str
                    st.success("Plan generated (or fallback provided). Review below.")
        except ValueError:
            st.error("Metric values and standard deviation must be numeric.")
st.markdown("</div>", unsafe_allow_html=True)

# --- Calculator Section ---
if "output" in st.session_state:
    st.markdown("<a name='output'></a>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    with st.expander("🔢 A/B Test Calculator: Fine-tune Your Sample Size", expanded=True):
        st.markdown("Adjust the experiment parameters to understand the impact on test duration.")
        
        baseline_rate = st.session_state.get('current', 0)
        metric_unit = st.session_state.get('context', {}).get('metric_unit', '')
        metric_type = st.session_state.get('context', {}).get('metric_type', 'Conversion Rate')
        
        st.metric(f"Baseline {'Conversion Rate' if metric_type == 'Conversion Rate' else 'Value'}", format_value_with_unit(baseline_rate, metric_unit))

        if 'calc_mde' not in st.session_state:
            st.session_state.calc_mde = st.session_state.get('context', {}).get('minimum_detectable_effect', 5.0)
        if 'calc_confidence' not in st.session_state:
            st.session_state.calc_confidence = 95
        if 'calc_power' not in st.session_state:
            st.session_state.calc_power = 80
        if 'calc_variants' not in st.session_state:
            st.session_state.calc_variants = 2

        col_input1, col_input2 = st.columns(2)
        with col_input1:
            st.session_state.calc_mde = st.number_input("Minimum Detectable Effect (MDE) %", min_value=0.1, max_value=50.0, value=float(st.session_state.calc_mde), step=0.1, key="mde_input")
            st.session_state.calc_confidence = st.number_input("Confidence Level (%)", min_value=80, max_value=99, value=int(st.session_state.calc_confidence), step=1, key="confidence_input")
        with col_input2:
            st.session_state.calc_power = st.number_input("Statistical Power (%)", min_value=70, max_value=95, value=int(st.session_state.calc_power), step=1, key="power_input")
            st.session_state.calc_variants = st.number_input("Number of Variants (Control + Variations)", min_value=2, max_value=5, value=int(st.session_state.calc_variants), step=1, key="variants_input")

        std_dev_calc = st.session_state.get('context', {}).get('std_dev', None)
        if metric_type == "Numeric Value":
            st.info(f"Standard Deviation for this metric is pre-filled from your input: **{std_dev_calc}**")

        col_buttons = st.columns(2)
        with col_buttons[0]:
            refresh_button = st.button("Refresh Calculator", key="refresh_calc_btn")
        with col_buttons[1]:
            lock_button = st.button("Lock Values for Plan", key="lock_calc_btn")
        
        if refresh_button or st.session_state.get('calculate_now', False) or lock_button:
            st.session_state.calculate_now = False
            
            st.session_state.last_calc_mde = float(st.session_state.calc_mde)
            st.session_state.last_calc_confidence = int(st.session_state.calc_confidence)
            st.session_state.last_calc_power = int(st.session_state.calc_power)
            st.session_state.last_calc_variants = int(st.session_state.calc_variants)

            alpha_calc = 1 - (st.session_state.last_calc_confidence / 100)
            power_calc = st.session_state.last_calc_power / 100
            
            sample_size_per_variant, total_sample_size = calculate_sample_size(
                baseline=baseline_rate, 
                mde=st.session_state.last_calc_mde, 
                alpha=alpha_calc, 
                power=power_calc, 
                num_variants=st.session_state.last_calc_variants,
                metric_type=metric_type,
                std_dev=std_dev_calc
            )

            st.session_state.calculated_sample_size_per_variant = sample_size_per_variant
            st.session_state.calculated_total_sample_size = total_sample_size
            
            dau_raw = st.session_state.get('context', {}).get('users', '< 10K')
            try:
                if dau_raw == '< 10K': dau = 5000
                elif dau_raw == '10K–100K': dau = 50000
                elif dau_raw == '100K–1M': dau = 500000
                else: dau = 2000000
            except:
                dau = 10000

            users_to_test = st.session_state.calculated_total_sample_size if st.session_state.calculated_total_sample_size else 0
            st.session_state.calculated_duration_days = (users_to_test / dau) if dau > 0 else float('inf')

        if st.session_state.get('calculated_sample_size_per_variant') is not None and st.session_state.get('calculated_total_sample_size') is not None:
            st.markdown("---")
            st.subheader("Calculator Results")
            st.metric("Users Per Variant", f"{st.session_state.calculated_sample_size_per_variant:,} users")
            st.metric("Total Sample Size", f"{st.session_state.calculated_total_sample_size:,} users")
            st.metric("Estimated Test Duration", f"{st.session_state.calculated_duration_days:,.0f} days")
            st.caption("Note: This calculation assumes all DAU are eligible for the test and are split evenly.")
        else:
            st.warning("Please adjust inputs and click 'Refresh Calculator' for results.")

        if lock_button:
            if st.session_state.get('calculated_sample_size_per_variant') is not None:
                st.session_state.stats_locked = True
                st.session_state.locked_stats = {
                    "confidence_level": st.session_state.last_calc_confidence,
                    "MDE": st.session_state.last_calc_mde,
                    "sample_size_required": f"{st.session_state.calculated_total_sample_size:,} users",
                    "users_per_variant": f"{st.session_state.calculated_sample_size_per_variant:,} users",
                    "estimated_test_duration": f"{st.session_state.calculated_duration_days:,.0f} days",
                }
                st.success("Calculator values locked and will be used in the plan below!")
                st.rerun()
            else:
                st.error("Cannot lock values. Please ensure the calculator has successfully generated results.")


    st.markdown("</div>", unsafe_allow_html=True)

# --- Display AI-Generated Plan ---
if "output" in st.session_state:
    plan = extract_json(st.session_state.output)
    if plan is None:
        st.error("Plan content could not be parsed. Please re-run generation.")
    else:
        unit = st.session_state.context.get("metric_unit", "").strip() if st.session_state.get("context") else ""
        
        st.markdown("<div class='green-section'>", unsafe_allow_html=True)
        create_header_with_help("Inferred Product Goal", "This is the goal the AI inferred from your input. Review and confirm it aligns with your objective.", icon="🎯")
        safe_display(post_process_llm_text(st.session_state.get("auto_goal", ""), unit))

        create_header_with_help("Problem Statement", "Explains the gap between current and target metric values, and why this improvement matters.", icon="🧩")
        problem_statement = post_process_llm_text(plan.get("problem_statement", ""), unit)
        safe_display(problem_statement or "⚠️ Problem statement not generated by the model.")

        create_header_with_help("Hypotheses", "These are actionable, testable ideas likely to improve the metric. They’re short, specific, and informed by your product context.", icon="🧪")
        hypotheses = plan.get("hypotheses", [])
        if not hypotheses or not isinstance(hypotheses, list):
            st.warning("No hypotheses found in the generated plan.")
            hypotheses = []

        # Render hypotheses and allow select
        for i, h in enumerate(hypotheses):
            # normalize hypothesis object
            if isinstance(h, dict):
                hypo_text = sanitize_text(h.get("hypothesis", "") or h.get("title", "") or "")
            else:
                hypo_text = sanitize_text(h)
            col1, col2 = st.columns([8, 1])
            with col1:
                st.markdown(f"**H{i+1}:** {hypo_text}")
            with col2:
                # Use on_click to set session state without ambiguous rerun race conditions
                if st.button("Select", key=f"select_{i}"):
                    st.session_state.selected_index = i
                    st.session_state.hypothesis_confirmed = True
                    st.session_state.stats_locked = False
                    # try to pull MDE suggested by LLM
                    try:
                        llm_mde = plan.get("success_criteria", {}).get("MDE", st.session_state.get('calc_mde', 5.0))
                        st.session_state.calc_mde = float(llm_mde)
                    except:
                        # leave existing MDE
                        pass
                    st.session_state.calculate_now = True
                    st.rerun()

        # If user selected one hypothesis earlier, show details
        if st.session_state.get("hypothesis_confirmed") and st.session_state.get("selected_index") is not None:
            i = st.session_state.selected_index
            # Guard against index errors
            if i < 0 or i >= len(hypotheses):
                st.error("Selected hypothesis index is out of range. Please re-select.")
            else:
                h_obj = hypotheses[i] if isinstance(hypotheses[i], dict) else {"hypothesis": hypotheses[i]}
                selected_hypo = sanitize_text(h_obj.get("hypothesis", "N/A"))

                # Safe extract rationale
                raw_rationales = plan.get("hypothesis_rationale", [])
                rationale = "N/A"
                if isinstance(raw_rationales, list) and i < len(raw_rationales):
                    r_item = raw_rationales[i]
                    if isinstance(r_item, dict):
                        rationale = sanitize_text(r_item.get("rationale", "N/A"))
                    else:
                        rationale = sanitize_text(r_item)

                # Safe variants / control / variation
                raw_variants = plan.get("variants", [])
                control = "Not specified"
                variation = "Not specified"
                if isinstance(raw_variants, list) and i < len(raw_variants):
                    v = raw_variants[i]
                    if isinstance(v, dict):
                        control = sanitize_text(v.get("control", "Not specified"))
                        variation = sanitize_text(v.get("variation", "Not specified"))
                    else:
                        # If variant is a string, put into variation
                        variation = sanitize_text(v)

                # effort
                raw_efforts = plan.get("effort", [])
                effort_display = "N/A"
                if isinstance(raw_efforts, list) and i < len(raw_efforts):
                    e = raw_efforts[i]
                    if isinstance(e, dict):
                        effort_display = sanitize_text(e.get("effort", "N/A"))
                    else:
                        effort_display = sanitize_text(e)

                # success criteria (use locked stats if locked)
                if st.session_state.get("stats_locked", False):
                    criteria_display = st.session_state.get("locked_stats", {})
                else:
                    criteria_display = plan.get("success_criteria", {})

                statistical_rationale_display = criteria_display.get("statistical_rationale") or plan.get("statistical_rationale") or plan.get("success_criteria", {}).get("statistical_rationale")
                if not statistical_rationale_display:
                    statistical_rationale_display = "The experiment is designed to detect a minimum effect size with a specified confidence and power level to ensure that any observed changes are statistically significant."

                # Confidence formatting
                try:
                    confidence = float(criteria_display.get("confidence_level", 0))
                    confidence_str = f"{round(confidence)}%" if confidence > 1 else f"{round(confidence * 100)}%"
                except:
                    confidence_str = "N/A"

                sample_size = criteria_display.get("sample_size_required", "N/A")
                users_per_variant = criteria_display.get("users_per_variant", "N/A")
                duration = criteria_display.get("estimated_test_duration", "N/A")

                try:
                    mde = float(criteria_display.get("MDE", 0))
                    mde_display = f"{mde}%"
                except:
                    mde_display = "N/A"

                create_header_with_help("Selected Hypothesis", "This is the hypothesis you selected from the generated options.", icon="🧪")
                st.code(selected_hypo)

                create_header_with_help("Variants", "Shows what users in the control group vs. the variant group will see or experience.", icon="🔁")
                st.markdown(f"- Control: {control}\n- Variation: {variation}")

                create_header_with_help("Rationale", "Explains why the selected hypothesis is worth testing, based on user behavior or insight.", icon="💡")
                st.markdown(rationale)

                create_header_with_help("Experiment Stats", "Details like sample size, confidence level, MDE, and duration help determine how trustworthy and actionable your results will be.", icon="📊")
                
                if not st.session_state.get("stats_locked", False):
                    st.warning("Please adjust the A/B Test Calculator above and click 'Lock Values for Plan' to finalize your experiment stats.")

                st.markdown(f"""
- Confidence Level: {confidence_str}
- Minimum Detectable Effect (MDE): {mde_display}
- Sample Size Required: {sample_size}
- Users per Variant: {users_per_variant}
- Estimated Duration: {duration}
- Estimated Effort: {effort_display}
**Statistical Rationale:** {statistical_rationale_display}
""")

                metrics = plan.get("metrics", [])
                if metrics and isinstance(metrics, list):
                    create_header_with_help("Metrics", "The primary and secondary metrics to be measured during the experiment.", icon="📏")
                    for m in metrics:
                        if isinstance(m, dict):
                            st.markdown(f"- **{m.get('name', 'Unnamed')}:** {m.get('formula', 'N/A')}")
                        else:
                            st.markdown(f"- {sanitize_text(m)}")

                segments = plan.get("segments", [])
                if segments and isinstance(segments, list):
                    create_header_with_help("Segments", "The specific user groups that will be included in the test.", icon="👥")
                    for s in segments:
                        st.markdown(f"- {sanitize_text(s)}")

                risks = plan.get("risks_and_assumptions", [])
                if risks and isinstance(risks, list):
                    create_header_with_help("Risks", "Potential risks, assumptions, and negative outcomes to monitor during the experiment.", icon="⚠️")
                    for r in risks:
                        st.markdown(f"- {sanitize_text(r)}")

                next_steps = plan.get("next_steps", [])
                if next_steps and isinstance(next_steps, list):
                    create_header_with_help("Next Steps", "A plan for the actions to be taken after the experiment concludes.", icon="✅")
                    for step in next_steps:
                        st.markdown(f"- {sanitize_text(step)}")

                # Build PRD string for download
                prd = []
                prd.append("# 🧪 Experiment PRD\n")
                prd.append("## 🎯 Goal\n")
                prd.append(st.session_state.get("auto_goal", "") + "\n")
                prd.append("## 🧩 Problem\n")
                prd.append(problem_statement + "\n")
                prd.append("## 🧪 Hypothesis\n")
                prd.append(selected_hypo + "\n")
                prd.append("## 🔁 Variants\n")
                prd.append(f"- Control: {control}\n- Variation: {variation}\n")
                prd.append("## 💡 Rationale\n")
                prd.append(rationale + "\n")
                prd.append("## 📊 Experiment Stats\n")
                prd.append(f"- Confidence Level: {confidence_str}\n")
                prd.append(f"- MDE: {mde_display}\n")
                prd.append(f"- Sample Size: {sample_size}\n")
                prd.append(f"- Users/Variant: {users_per_variant}\n")
                prd.append(f"- Duration: {duration}\n")
                prd.append(f"- Effort: {effort_display}\n")
                prd.append(f"- Statistical Rationale: {statistical_rationale_display}\n")

                if metrics and isinstance(metrics, list):
                    prd.append("\n## 📏 Metrics\n")
                    for m in metrics:
                        if isinstance(m, dict):
                            prd.append(f"- {m.get('name','Unnamed')}: {m.get('formula','N/A')}\n")
                        else:
                            prd.append(f"- {sanitize_text(m)}\n")

                if segments and isinstance(segments, list):
                    prd.append("\n## 👥 Segments\n")
                    for s in segments:
                        prd.append(f"- {sanitize_text(s)}\n")

                if risks and isinstance(risks, list):
                    prd.append("\n## ⚠️ Risks\n")
                    for r in risks:
                        prd.append(f"- {sanitize_text(r)}\n")

                if next_steps and isinstance(next_steps, list):
                    prd.append("\n## ✅ Next Steps\n")
                    for step in next_steps:
                        prd.append(f"- {sanitize_text(step)}\n")

                prd_text = "\n".join(prd)
                st.download_button("📄 Download PRD", prd_text, file_name="experiment_prd.txt")
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
