import streamlit as st
import json
import base64
import re
import os
from prompt_engine import generate_experiment_plan
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize
import numpy as np

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

def sanitize_text(text):
    """Sanitizes text to prevent injection issues and clean up whitespace."""
    if not text or not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def safe_display(text, method=st.info):
    """Displays sanitized text using a Streamlit method."""
    clean_text = sanitize_text(text)
    method(clean_text)

def extract_json(text):
    """
    Extracts and attempts to fix a JSON object from a string that may contain other text.
    """
    try:
        match = re.search(r"\{[\s\S]+\}", text)
        json_str = match.group(0) if match else text
        
        # Simple fix for trailing commas, which can cause a JSONDecodeError
        json_str = re.sub(r',\s*\}', '}', json_str)
        json_str = re.sub(r',\s*\]', ']', json_str)
        
        return json.loads(json_str)
    except Exception as e:
        st.error(f"❌ Could not parse JSON: {e}")
        st.code(text)
        return None

def remove_units_from_text(text, unit):
    """Removes a unit from a number within a string."""
    if not text or not unit.strip():
        return text
    escaped_unit = re.escape(unit.strip())
    return re.sub(rf"(\d+\.?\d*)\s*{escaped_unit}", r"\1", text)

def insert_units_in_goal(text, unit):
    """
    Inserts a unit into numeric values within a string.
    Correctly places '%' after the number.
    """
    if not text or not unit.strip():
        return text

    unit = unit.strip()
    output = []
    tokens = re.split(r'(\s+|\d+\.?\d*)', text)
    for i, token in enumerate(tokens):
        if re.match(r'^\d+\.?\d*$', token):
            if unit in ["$", "₹", "€", "£"]:
                output.append(unit + token)
            else:
                output.append(token + " " + unit)
        else:
            output.append(token)
    
    # Correct the placement of the '%' symbol
    return ''.join(output).replace(" %", "%").replace(" %", "%")

def calculate_sample_size(baseline, mde, alpha, power, num_variants, metric_type, std_dev=None):
    """
    Calculates the required sample size for an A/B test based on metric type.
    """
    try:
        alpha = alpha
        power = power
        mde_relative = mde / 100.0

        if metric_type == 'Conversion Rate':
            baseline_prop = baseline / 100.0
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
            if std_dev is None or std_dev == 0:
                st.error("Standard deviation is required and must be non-zero.")
                return None, None
            mde_absolute = baseline * mde_relative
            effect_size = mde_absolute / std_dev
            
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
        
        if sample_size_per_variant <= 0 or not np.isfinite(sample_size_per_variant):
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
    if not exact_metric.strip(): missing.append("Metric to Improve")
    if not current_value_raw.strip(): missing.append("Current Value")
    if not target_value_raw.strip(): missing.append("Target Value")
    if not metric_unit.strip(): missing.append("Metric Unit")
    if metric_type == "Numeric Value" and not std_dev_raw: missing.append("Standard Deviation")

    invalid_chars = ['"', '{', '}', '[', ']', '$', '₹', '€', '£']
    if any(char in metric_unit for char in invalid_chars):
        st.error("The 'Metric Unit' contains invalid characters. Please use plain text like 'USD' or 'count' instead of symbols like '$' or brackets.")
        st.stop()

    if missing:
        st.warning("Please fill all required fields: " + ", ".join(missing))
        st.stop()

    try:
        # Sanitize inputs by removing any percentage signs before converting to float
        current = float(current_value_raw.replace('%', ''))
        target = float(target_value_raw.replace('%', ''))
        std_dev = float(std_dev_raw.replace('%', '')) if std_dev_raw else None
        
        # Also, remove any percentage signs from the metric name
        sanitized_metric_name = exact_metric.replace('%', '')

        if current == 0 and metric_type == "Conversion Rate":
            st.error("Current value cannot be zero for conversion rate lift calculation.")
            st.stop()
        if current == 0 and metric_type == "Numeric Value" and std_dev == 0:
             st.error("Current value or standard deviation cannot be zero for numeric metric calculation.")
             st.stop()

        expected_lift = round(((target - current) / current) * 100, 2) if current != 0 else 0.0
        mde_percent = round(abs((target - current) / current) * 100, 2) if current != 0 else 0.0
    except ValueError:
        st.error("Metric values and standard deviation must be numeric.")
        st.stop()

    goal_text = f"I want to improve {sanitized_metric_name} from {current} to {target}."
    goal_with_units = insert_units_in_goal(goal_text, metric_unit).strip()

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
    st.session_state.stats_locked = False
    st.session_state.calculate_now = True

    with st.spinner("🧠 Generating your plan..."):
        output = generate_experiment_plan(goal_with_units, st.session_state.context)
    st.session_state.output = output
    st.session_state.hypothesis_confirmed = False
    st.session_state.selected_index = None

# --- Calculator Section ---
if "output" in st.session_state:
    st.markdown("<a name='output'></a>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    with st.expander("🔢 A/B Test Calculator: Fine-tune Your Sample Size", expanded=True):
        st.markdown("Adjust the experiment parameters to understand the impact on test duration.")
        
        baseline_rate = st.session_state.get('current', 0)
        metric_unit = st.session_state.get('context', {}).get('metric_unit', '')
        metric_type = st.session_state.get('context', {}).get('metric_type', 'Conversion Rate')
        
        st.metric(f"Baseline {'Conversion Rate' if metric_type == 'Conversion Rate' else 'Value'}", f"{baseline_rate}{metric_unit}")

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
            st.session_state.calc_mde = st.number_input("Minimum Detectable Effect (MDE) %", min_value=0.1, max_value=50.0, value=st.session_state.calc_mde, step=0.1, key="mde_input")
            st.session_state.calc_confidence = st.number_input("Confidence Level (%)", min_value=80, max_value=99, value=st.session_state.calc_confidence, step=1, key="confidence_input")
        with col_input2:
            st.session_state.calc_power = st.number_input("Statistical Power (%)", min_value=70, max_value=95, value=st.session_state.calc_power, step=1, key="power_input")
            st.session_state.calc_variants = st.number_input("Number of Variants (Control + Variations)", min_value=2, max_value=5, value=st.session_state.calc_variants, step=1, key="variants_input")

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
            
            st.session_state.last_calc_mde = st.session_state.calc_mde
            st.session_state.last_calc_confidence = st.session_state.calc_confidence
            st.session_state.last_calc_power = st.session_state.calc_power
            st.session_state.last_calc_variants = st.session_state.calc_variants

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
        st.stop()

    unit = st.session_state.context.get("metric_unit", "").strip()
    
    st.markdown("<div class='green-section'>", unsafe_allow_html=True)
    create_header_with_help("Inferred Product Goal", "This is the goal the AI inferred from your input. Review and confirm it aligns with your objective.", icon="🎯")
    safe_display(st.session_state.auto_goal)

    create_header_with_help("Problem Statement", "Explains the gap between current and target metric values, and why this improvement matters.", icon="🧩")
    problem_statement = plan.get("problem_statement", "")
    problem_statement = insert_units_in_goal(problem_statement, unit)
    safe_display(problem_statement or "⚠️ Problem statement not generated by the model.")

    create_header_with_help("Hypotheses", "These are actionable, testable ideas likely to improve the metric. They’re short, specific, and informed by your product context.", icon="🧪")
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
                    st.session_state.stats_locked = False
                    llm_mde = plan.get("success_criteria", {}).get("MDE", 5.0)
                    st.session_state.calc_mde = llm_mde
                    st.session_state.calculate_now = True
                    st.rerun()

    if st.session_state.get("hypothesis_confirmed") and st.session_state.selected_index is not None:
        i = st.session_state.selected_index
        selected_hypo = hypotheses[i].get("hypothesis", "N/A")
        rationale = sanitize_text(plan.get("hypothesis_rationale", [{}])[i].get("rationale", "N/A"))
        variant = plan.get("variants", [{}])[i]
        control = variant.get("control", "Not specified")
        variation = variant.get("variation", "Not specified")
        
        effort_display = plan.get("effort", [{}])[i].get("effort", "N/A")
        
        statistical_rationale_display = plan.get("success_criteria", {}).get("statistical_rationale")
        if not statistical_rationale_display:
            statistical_rationale_display = "The experiment is designed to detect a minimum effect size with a specified confidence and power level to ensure that any observed changes are statistically significant."

        if st.session_state.get("stats_locked", False):
            criteria_display = st.session_state.locked_stats
        else:
            criteria_display = plan.get("success_criteria", {})

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
        if metrics:
            create_header_with_help("Metrics", "The primary and secondary metrics to be measured during the experiment.", icon="📏")
            for m in metrics:
                st.markdown(f"- **{m.get('name', 'Unnamed')}:** {m.get('formula', 'N/A')}")

        segments = plan.get("segments", [])
        if segments:
            create_header_with_help("Segments", "The specific user groups that will be included in the test.", icon="👥")
            for s in segments:
                st.markdown(f"- {s}")

        risks = plan.get("risks_and_assumptions", [])
        if risks:
            create_header_with_help("Risks", "Potential risks, assumptions, and negative outcomes to monitor during the experiment.", icon="⚠️")
            for r in risks:
                st.markdown(f"- {r}")

        next_steps = plan.get("next_steps", [])
        if next_steps:
            create_header_with_help("Next Steps", "A plan for the actions to be taken after the experiment concludes.", icon="✅")
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
        prd += f"- Duration: {duration}\n"
        prd += f"- Effort: {effort_display}\n"
        prd += f"- Statistical Rationale: {statistical_rationale_display}\n\n"

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

        risks = plan.get("risks_and_assumptions", [])
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
