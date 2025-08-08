# main.py — AI A/B Test Architect (polished UI + robust parsing + PDF export + structured metric editor)
import streamlit as st
import json
import re
import os
import pandas as pd
from typing import Any, Dict, Optional, Tuple, List
from prompt_engine import generate_experiment_plan
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize
import numpy as np
import hashlib
import io
from datetime import datetime
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter

# Try to import reportlab for PDF export; we'll gracefully fall back if missing.
REPORTLAB_AVAILABLE = False
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas

    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

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
    if not isinstance(text, str):
        return None
    tag_match = re.search(r"<json>([\s\S]+?)</json>", text, re.IGNORECASE)
    if tag_match:
        return tag_match.group(1).strip()

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
    if text is None:
        st.error("No output returned from LLM.")
        return None

    if isinstance(text, dict):
        return text

    if isinstance(text, list):
        st.error("LLM returned a JSON list when an object was expected.")
        return None

    try:
        raw = str(text)
    except Exception as e:
        st.error(f"Unexpected LLM output type: {e}")
        return None

    # Try direct parse
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
        else:
            st.error("Parsed JSON is not an object.")
            return None
    except Exception:
        pass

    candidate = _extract_json_first_braces(raw)
    if candidate:
        candidate_clean = re.sub(r',\s*,', ',', candidate)
        candidate_clean = re.sub(r',\s*\}', '}', candidate_clean)
        candidate_clean = re.sub(r',\s*\]', ']', candidate_clean)
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
# PDF / Export helpers
# -------------------------
def build_prd_html(title: str, prd_text: str, logo_svg: Optional[str] = None) -> str:
    # Basic HTML wrapper with nicer styling for preview/export
    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>{title}</title>
      <style>
        body {{ font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial; padding: 40px; color: #1b1b1b; }}
        .logo {{ width: 120px; }}
        h1 {{ font-size: 28px; margin-bottom: 6px; }}
        h2 {{ font-size: 20px; margin-top: 18px; }}
        p, li {{ font-size: 14px; line-height: 1.45; }}
        pre {{ white-space: pre-wrap; font-family: inherit; }}
        .meta {{ color: #666; margin-bottom: 20px; }}
        .section {{ margin-bottom: 18px; }}
      </style>
    </head>
    <body>
      <div class="header">
        {logo_svg or '<div style="width:120px;height:40px;background:#1E90FF;border-radius:6px;color:white;display:flex;align-items:center;justify-content:center;font-weight:700">A/B</div>'}
        <h1>{title}</h1>
        <div class="meta">Generated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}</div>
      </div>
      <div class="content">
        <pre>{prd_text}</pre>
      </div>
    </body>
    </html>
    """
    return html


def generate_pdf_bytes_from_text(prd_data, title="Experiment PRD"):
    """
    prd_data: dict containing the PRD content already parsed from AI output
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=40, leftMargin=40,
                            topMargin=50, bottomMargin=40)
    styles = getSampleStyleSheet()

    # Custom styles
    styles.add(ParagraphStyle(name="PRDTitle", fontSize=20, leading=24, spaceAfter=20, alignment=1))
    styles.add(ParagraphStyle(name="SectionHeading", fontSize=14, leading=18, spaceBefore=12, spaceAfter=6, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name="BodyTextCustom", fontSize=11, leading=14))
    styles.add(ParagraphStyle(name="BulletText", fontSize=11, leading=14, leftIndent=15, bulletIndent=5))

    story = []

    # Optional logo
    try:
        from reportlab.platypus import Image
        logo_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Google_2015_logo.svg/320px-Google_2015_logo.svg.png"
        story.append(Image(logo_url, width=120, height=40))
    except Exception:
        pass

    # Title
    story.append(Paragraph(title, styles["PRDTitle"]))

    # Sections
    def add_section(heading, content):
        story.append(Paragraph(heading, styles["SectionHeading"]))
        if isinstance(content, list):
            for item in content:
                story.append(Paragraph(item, styles["BulletText"], bulletText="•"))
        else:
            story.append(Paragraph(str(content), styles["BodyTextCustom"]))
        story.append(Spacer(1, 8))

    add_section("🎯 Goal", prd_data.get("goal", ""))
    add_section("🧩 Problem Statement", prd_data.get("problem_statement", ""))
    add_section("💡 Hypotheses", [f"{h['hypothesis']}: {h.get('description','')}" for h in prd_data.get("hypotheses", [])])
    add_section("📊 Metrics", [f"{m['name']} — {m['formula']}" for m in prd_data.get("metrics", [])])
    add_section("👥 Segments", prd_data.get("segments", []))
    add_section("✅ Success Criteria", prd_data.get("success_criteria", {}))
    add_section("⚙️ Effort", [f"{e['hypothesis']}: {e['effort']}" for e in prd_data.get("effort", [])])
    add_section("🛠 Team Involved", prd_data.get("team_involved", []))
    add_section("🔍 Hypothesis Rationale", [r['rationale'] for r in prd_data.get("hypothesis_rationale", [])])
    add_section("⚠️ Risks & Assumptions", prd_data.get("risks_and_assumptions", []))
    add_section("🚀 Next Steps", prd_data.get("next_steps", []))
    add_section("📈 Statistical Rationale", prd_data.get("statistical_rationale", ""))

    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf


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
.small-muted { color: #7a7a7a; font-size: 13px; }
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
# INPUTS: Business Context
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
# GENERATE PLAN AREA
# -------------------------
st.markdown("<div class='green-section'>", unsafe_allow_html=True)
create_header_with_help("Generate Experiment Plan", "When ready, click Generate to call the LLM and create a plan.", icon="🧠")

formatted_current = format_value_with_unit(current_value, metric_unit)
formatted_target = format_value_with_unit(target_value, metric_unit)
sanitized_metric_name = sanitize_text(exact_metric)
goal_with_units = f"I want to improve {sanitized_metric_name} from {formatted_current} to {formatted_target}."

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

    with st.spinner("Generating your plan..."):
        try:
            raw_llm = generate_experiment_plan(goal_with_units, context)
            st.session_state.output = raw_llm if raw_llm is not None else ""
            parsed = extract_json(raw_llm)
            st.session_state.ai_parsed = parsed
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
# Calculator (Sample size)
# -------------------------
if st.session_state.get("ai_parsed") is not None or st.session_state.get("output"):
    st.markdown("<hr>", unsafe_allow_html=True)
    with st.expander("🔢 A/B Test Calculator: Fine-tune sample size", expanded=True):
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

            dau_map = {"< 10K": 5000, "10K–100K": 50000, "100K–1M": 500000, "> 1M": 2000000}
            dau = dau_map.get(user_base_choice, 10000)
            users_to_test = st.session_state.calculated_total_sample_size or 0
            st.session_state.calculated_duration_days = (users_to_test / dau) if dau > 0 else float("inf")

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
# DISPLAY AI-GENERATED PLAN (editable + final view + structured metric editor)
# -------------------------
if st.session_state.get("ai_parsed") is None and st.session_state.get("output"):
    st.info("AI returned output but parsing failed. Edit raw output in the Raw JSON tab to fix the parse or try regenerating.")

if st.session_state.get("ai_parsed"):
    plan = st.session_state.ai_parsed
    # store context locally for easy access
    st.session_state.context = st.session_state.get("context", {
        "metric_unit": metric_unit
    })

    unit = st.session_state.context.get("metric_unit", metric_unit)

    st.markdown("<div class='green-section'>", unsafe_allow_html=True)
    create_header_with_help("Inferred Product Goal", "The AI's interpretation of your goal. Edit if needed.", icon="🎯")
    st.markdown(f"**Goal:** {post_process_llm_text(goal_with_units, unit)}")

    create_header_with_help("Problem Statement", "Clear description of the gap and why it matters.", icon="🧩")
    problem_statement = post_process_llm_text(plan.get("problem_statement", ""), unit)
    st.markdown(problem_statement or "⚠️ Problem statement not generated by the model.")
    with st.expander("Edit Problem Statement"):
        st.text_area("Problem Statement (edit)", value=problem_statement, key="editable_problem", height=160)

    create_header_with_help("Hypotheses", "Editable list of testable hypotheses.", icon="🧪")
    hypotheses = plan.get("hypotheses", [])
    if not hypotheses or not isinstance(hypotheses, list):
        st.warning("No hypotheses found in the generated plan.")
        hypotheses = []

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
            try:
                llm_mde = plan.get("success_criteria", {}).get("MDE", st.session_state.get("calc_mde", 5.0))
                st.session_state.calc_mde = float(llm_mde)
            except Exception:
                pass
            st.rerun()

    # If selected, show details
    if st.session_state.get("hypothesis_confirmed") and st.session_state.get("selected_index") is not None:
        idx = st.session_state.selected_index
        if idx < 0 or idx >= len(hypotheses):
            st.error("Selected hypothesis index out of range.")
        else:
            h_obj = hypotheses[idx] if isinstance(hypotheses[idx], dict) else {"hypothesis": hypotheses[idx]}
            selected_hypo = sanitize_text(h_obj.get("hypothesis", "N/A"))

            raw_rationales = plan.get("hypothesis_rationale", [])
            rationale = "N/A"
            if isinstance(raw_rationales, list) and idx < len(raw_rationales):
                r_item = raw_rationales[idx]
                if isinstance(r_item, dict):
                    rationale = sanitize_text(r_item.get("rationale", "N/A"))
                else:
                    rationale = sanitize_text(r_item)

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

            raw_efforts = plan.get("effort", [])
            effort_display = "N/A"
            if isinstance(raw_efforts, list) and idx < len(raw_efforts):
                e = raw_efforts[idx]
                if isinstance(e, dict):
                    effort_display = sanitize_text(e.get("effort", "N/A"))
                else:
                    effort_display = sanitize_text(e)

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

            # Display and edit sections
            create_header_with_help("Selected Hypothesis", "This is the hypothesis you selected from the generated options.", icon="🧪")
            st.markdown(f"**Hypothesis:** {selected_hypo}")
            with st.expander("Edit Hypothesis"):
                st.text_area("Hypothesis (edit)", value=selected_hypo, key="editable_hypothesis", height=100)

            create_header_with_help("Variants", "Control vs Variation", icon="🔁")
            st.markdown(f"- **Control:** {control}\n- **Variation:** {variation}")
            with st.expander("Edit Variants"):
                st.text_input("Control (editable)", value=control, key="editable_control")
                st.text_input("Variation (editable)", value=variation, key="editable_variation")

            create_header_with_help("Rationale", "Why this hypothesis is worth testing", icon="💡")
            st.markdown(rationale)
            with st.expander("Edit Rationale"):
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

            # Metrics — structured editor (name/formula) + nice table
            metrics = plan.get("metrics", [])
            if metrics and isinstance(metrics, list):
                create_header_with_help("Metrics", "Primary and secondary metrics (structured editor)", icon="📏")
                normalized: List[Dict[str, str]] = []
                for m in metrics:
                    if isinstance(m, dict):
                        normalized.append({"name": m.get("name", "Unnamed"), "formula": m.get("formula", "")})
                    else:
                        try:
                            parsed_m = json.loads(m)
                            normalized.append({"name": parsed_m.get("name", "Unnamed"), "formula": parsed_m.get("formula", "")})
                        except Exception:
                            normalized.append({"name": sanitize_text(m), "formula": ""})

                # Display table view
                try:
                    df_metrics = pd.DataFrame(normalized)
                    st.table(df_metrics)
                except Exception:
                    for nm in normalized:
                        st.markdown(f"- **{nm.get('name')}**: {nm.get('formula')}")

                with st.expander("Edit Metrics (structured)"):
                    for mi, m in enumerate(normalized):
                        st.text_input(f"Metric {mi+1} Name", value=m.get("name", ""), key=f"metric_name_{mi}")
                        st.text_input(f"Metric {mi+1} Formula", value=m.get("formula", ""), key=f"metric_formula_{mi}")

            # Segments (clean list)
            segments = plan.get("segments", [])
            if segments and isinstance(segments, list):
                create_header_with_help("Segments", "User segments for analysis (clean list)", icon="👥")
                for s in segments:
                    st.markdown(f"- {sanitize_text(s)}")
                with st.expander("Edit Segments"):
                    st.text_area("Segments (one per line)", value="\n".join(segments), key="editable_segments", height=120)

            # Risks
            risks = plan.get("risks_and_assumptions", [])
            if risks and isinstance(risks, list):
                create_header_with_help("Risks & Assumptions", "What could impact test outcomes (clean list)", icon="⚠️")
                for r in risks:
                    st.markdown(f"- {sanitize_text(r)}")
                with st.expander("Edit Risks & Assumptions"):
                    st.text_area("Risks (one per line)", value="\n".join(risks), key="editable_risks", height=120)

            # Next steps
            next_steps = plan.get("next_steps", [])
            if next_steps and isinstance(next_steps, list):
                create_header_with_help("Next Steps", "Actionable tasks to start the experiment (clean list)", icon="✅")
                for ns in next_steps:
                    st.markdown(f"- {sanitize_text(ns)}")
                with st.expander("Edit Next Steps"):
                    st.text_area("Next Steps (one per line)", value="\n".join(next_steps), key="editable_next_steps", height=120)

            # Build final PRD string (gather edited inputs)
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

            # Metrics: use structured edited fields if present
            if metrics and isinstance(metrics, list):
                prd_parts.append("## 📏 Metrics\n")
                for mi, orig_m in enumerate(metrics):
                    edited_name = st.session_state.get(f"metric_name_{mi}")
                    edited_formula = st.session_state.get(f"metric_formula_{mi}")
                    if edited_name or edited_formula:
                        name = edited_name or (orig_m.get("name") if isinstance(orig_m, dict) else sanitize_text(orig_m))
                        formula = edited_formula or (orig_m.get("formula") if isinstance(orig_m, dict) else "")
                        prd_parts.append(f"- {name}: {formula}\n")
                    else:
                        if isinstance(orig_m, dict):
                            prd_parts.append(f"- {orig_m.get('name','Unnamed')}: {orig_m.get('formula','N/A')}\n")
                        else:
                            prd_parts.append(f"- {sanitize_text(orig_m)}\n")

            if st.session_state.get("editable_segments"):
                prd_parts.append("\n## 👥 Segments\n")
                for s in st.session_state.get("editable_segments", "").splitlines():
                    if s.strip():
                        prd_parts.append(f"- {s.strip()}\n")

            if st.session_state.get("editable_risks"):
                prd_parts.append("\n## ⚠️ Risks\n")
                for r in st.session_state.get("editable_risks", "").splitlines():
                    if r.strip():
                        prd_parts.append(f"- {r.strip()}\n")

            if st.session_state.get("editable_next_steps"):
                prd_parts.append("\n## ✅ Next Steps\n")
                for ns in st.session_state.get("editable_next_steps", "").splitlines():
                    if ns.strip():
                        prd_parts.append(f"- {ns.strip()}\n")

            prd_text = "\n".join(prd_parts)

            # Show Final PRD preview (production-like)
            create_header_with_help("Final PRD Preview", "A clean, production-style preview suitable for interviews. Export to PDF or HTML.", icon="📄")
            # Polished preview area (larger font)
            st.markdown(
                f"""
                <div style="padding:18px;border-radius:8px;background:white;">
                    <div style="display:flex;align-items:center;gap:18px;">
                        <div style="width:72px;height:72px;background:#1E90FF;color:white;border-radius:10px;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:22px;">A/B</div>
                        <div>
                            <div style="font-size:18px;font-weight:700;">Experiment PRD</div>
                            <div style="color:#666;">{sanitize_text(goal_with_units)}</div>
                        </div>
                    </div>
                    <hr style="margin-top:12px;margin-bottom:12px;">
                    <div style="font-size:14px;line-height:1.5;">
                        <pre style="white-space:pre-wrap;font-family:inherit;">{sanitize_text(prd_text)}</pre>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Download buttons: TXT, JSON, HTML, PDF (if available)
            col_dl1, col_dl2, col_dl3, col_dl4 = st.columns([1,1,1,1])
            with col_dl1:
                st.download_button("📄 Download PRD (.txt)", prd_text, file_name="experiment_prd.txt")
            with col_dl2:
                st.download_button("📥 Download Plan (.json)", json.dumps(plan, indent=2, ensure_ascii=False), file_name="experiment_plan.json")
            with col_dl3:
                html_blob = build_prd_html("Experiment PRD", prd_text)
                st.download_button("🌐 Download PRD (.html)", html_blob, file_name="experiment_prd.html")
            with col_dl4:
                if REPORTLAB_AVAILABLE:
                    pdf_bytes = generate_pdf_bytes_from_text(prd_text, title="Experiment PRD")
                    if pdf_bytes:
                        st.download_button("📁 Download PRD (.pdf)", pdf_bytes, file_name="experiment_prd.pdf", mime="application/pdf")
                    else:
                        st.warning("PDF generation currently failed — try HTML download.")
                else:
                    st.info("PDF export requires 'reportlab'. Click HTML or TXT to download. To enable PDF, `pip install reportlab`.")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Raw LLM Output / Manual Repair Tab
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
        keys_to_clear = ["output", "ai_parsed", "calculated_sample_size_per_variant", "calculated_total_sample_size", "calculated_duration_days", "locked_stats", "calc_locked", "selected_index", "hypothesis_confirmed", "last_llm_hash", "context"]
        for k in keys_to_clear:
            if k in st.session_state:
                del st.session_state[k]
        st.success("Session cleared. Reloading...")
        st.rerun()
