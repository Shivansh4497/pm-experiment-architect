# main.py — Bug-fixed, optimized, polished UI for A/B Test Architect
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
from datetime import datetime
from io import BytesIO
import ast

# reportlab imports (optional) — used for PDF export
REPORTLAB_AVAILABLE = False
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.units import inch
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
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="font-size: 1.4rem;">{icon}</div>
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
    text = text.replace("\r", " ").replace("\t", " ")
    text = re.sub(r"[ \f\v]+", " ", text)
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

def _safe_single_to_double_quotes(s: str) -> str:
    s = re.sub(r"(?<=[:\{\[,]\s*)'([^']*?)'(?=\s*[,}\]])", r'"\1"', s)
    s = re.sub(r"'([A-Za-z0-9_ \-]+?)'\s*:", r'"\1":', s)
    return s

def extract_json(text: Any) -> Optional[Dict]:
    """
    Robust attempt to parse JSON from a variety of LLM outputs.
    Attempts:
      - json.loads(raw)
      - ast.literal_eval(raw)
      - extract first {...} and parse
      - safe single->double quote conversion attempts
    If top-level list is returned, we attempt to wrap into an object where reasonable.
    """
    if text is None:
        st.error("No output returned from LLM.")
        return None

    if isinstance(text, dict):
        return text
    if isinstance(text, list):
        # If model returned a list but we expected an object, try a safe wrap if list contains a dict
        if all(isinstance(i, dict) for i in text):
            return {"items": text}
        st.error("LLM returned a JSON list when an object was expected.")
        return None

    try:
        raw = str(text)
    except Exception as e:
        st.error(f"Unexpected LLM output type: {e}")
        return None

    # 1) direct JSON
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and all(isinstance(i, dict) for i in parsed):
            return {"items": parsed}
        st.error("Parsed JSON was not an object; expected an object.")
        return None
    except Exception:
        pass

    # 2) ast.literal_eval
    try:
        parsed_ast = ast.literal_eval(raw)
        if isinstance(parsed_ast, dict):
            return parsed_ast
        if isinstance(parsed_ast, list) and all(isinstance(i, dict) for i in parsed_ast):
            return {"items": parsed_ast}
    except Exception:
        pass

    # 3) extract first balanced braces substring
    candidate = _extract_json_first_braces(raw)
    if candidate:
        candidate_clean = candidate
        candidate_clean = re.sub(r"^```(?:json)?\s*", "", candidate_clean).strip()
        candidate_clean = re.sub(r"\s*```$", "", candidate_clean).strip()
        candidate_clean = re.sub(r',\s*,', ',', candidate_clean)
        candidate_clean = re.sub(r',\s*\}', '}', candidate_clean)
        candidate_clean = re.sub(r',\s*\]', ']', candidate_clean)
        try:
            parsed = json.loads(candidate_clean)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and all(isinstance(i, dict) for i in parsed):
                return {"items": parsed}
            st.error("Extracted JSON parsed but was not an object.")
            st.code(candidate_clean[:2000] + ("..." if len(candidate_clean) > 2000 else ""))
            return None
        except Exception:
            try:
                parsed_ast = ast.literal_eval(candidate_clean)
                if isinstance(parsed_ast, dict):
                    return parsed_ast
                if isinstance(parsed_ast, list) and all(isinstance(i, dict) for i in parsed_ast):
                    return {"items": parsed_ast}
            except Exception:
                try:
                    converted = _safe_single_to_double_quotes(candidate_clean)
                    parsed = json.loads(converted)
                    if isinstance(parsed, dict):
                        return parsed
                    if isinstance(parsed, list) and all(isinstance(i, dict) for i in parsed):
                        return {"items": parsed}
                except Exception:
                    st.error("Could not parse extracted JSON block. See snippet below.")
                    st.code(candidate_clean[:3000] + ("..." if len(candidate_clean) > 3000 else ""))
                    return None

    # 4) conservative single->double quotes on full raw
    try:
        converted_full = _safe_single_to_double_quotes(raw)
        parsed = json.loads(converted_full)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and all(isinstance(i, dict) for i in parsed):
            return {"items": parsed}
    except Exception:
        pass

    st.error("LLM output could not be parsed as JSON. Please inspect or edit the raw output below.")
    try:
        st.code(raw[:3000] + ("..." if len(raw) > 3000 else ""))
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

def calculate_sample_size(baseline, mde, alpha, power, num_variants, metric_type, std_dev=None) -> Tuple[Optional[int], Optional[int]]:
    try:
        if baseline is None or mde is None:
            return None, None

        # If baseline is zero, treat special-case to avoid division by zero.
        mde_relative = float(mde) / 100.0
        if metric_type == "Conversion Rate":
            try:
                baseline_prop = float(baseline) / 100.0
            except Exception:
                return None, None
            if baseline_prop <= 0:
                return None, None
            expected_prop = baseline_prop * (1 + mde_relative)
            expected_prop = min(expected_prop, 0.999)
            effect_size = proportion_effectsize(baseline_prop, expected_prop)
            if effect_size == 0:
                return None, None
            analysis = NormalIndPower()
            sample_size_per_variant = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative="two-sided")
        elif metric_type == "Numeric Value":
            if std_dev is None or float(std_dev) == 0:
                return None, None
            mde_absolute = float(baseline) * mde_relative
            effect_size = mde_absolute / float(std_dev)
            if effect_size == 0:
                return None, None
            analysis = TTestIndPower()
            sample_size_per_variant = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative="two-sided")
        else:
            return None, None

        if sample_size_per_variant is None or sample_size_per_variant <= 0 or not np.isfinite(sample_size_per_variant):
            return None, None

        total = sample_size_per_variant * num_variants
        return int(np.ceil(sample_size_per_variant)), int(np.ceil(total))
    except Exception:
        return None, None

# -------------------------
# PDF export
# -------------------------
def generate_pdf_bytes_from_prd_dict(prd: Dict, title: str = "Experiment PRD") -> Optional[bytes]:
    if not REPORTLAB_AVAILABLE:
        return None
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="PRDTitle", fontSize=20, leading=24, spaceAfter=12, alignment=1))
    styles.add(ParagraphStyle(name="SectionHeading", fontSize=13, leading=16, spaceBefore=12, spaceAfter=6, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name="BodyTextCustom", fontSize=10.5, leading=14))
    styles.add(ParagraphStyle(name="BulletText", fontSize=10.5, leading=14, leftIndent=12, bulletIndent=6))
    story: List[Any] = []
    story.append(Paragraph(title, styles["PRDTitle"]))
    def add_paragraph(heading: str, content: Any):
        story.append(Paragraph(heading, styles["SectionHeading"]))
        if content is None:
            return
        def pdf_sanitize(text: Any) -> str:
            if text is None: return ""
            return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if isinstance(content, str):
            story.append(Paragraph(pdf_sanitize(content).replace('\n', '<br/>'), styles["BodyTextCustom"]))
        elif isinstance(content, dict):
            for k, v in content.items():
                if v is None:
                    continue
                story.append(Paragraph(f"<b>{pdf_sanitize(str(k))}:</b> {pdf_sanitize(str(v))}", styles["BodyTextCustom"]))
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    story.append(Paragraph(json.dumps(item, ensure_ascii=False), styles["BulletText"], bulletText="•"))
                else:
                    story.append(Paragraph(pdf_sanitize(str(item)), styles["BulletText"], bulletText="•"))
        else:
            story.append(Paragraph(pdf_sanitize(str(content)), styles["BodyTextCustom"]))
        story.append(Spacer(1, 6))
    add_paragraph("🎯 Goal", prd.get("goal", ""))
    add_paragraph("🧩 Problem Statement", prd.get("problem_statement", ""))
    add_paragraph("💡 Hypotheses", prd.get("hypotheses", []))
    add_paragraph("📏 Metrics", prd.get("metrics", []))
    add_paragraph("👥 Segments", prd.get("segments", []))
    add_paragraph("✅ Success Criteria", [f"{k}: {v}" for k, v in (prd.get("success_criteria", {}) or {}).items()])
    add_paragraph("⚙️ Effort", prd.get("effort", []))
    add_paragraph("🧑‍💻 Team Involved", prd.get("team_involved", []))
    add_paragraph("🔍 Hypothesis Rationale", prd.get("hypothesis_rationale", []))
    add_paragraph("⚠️ Risks & Assumptions", prd.get("risks_and_assumptions", []))
    add_paragraph("🚀 Next Steps", prd.get("next_steps", []))
    add_paragraph("📈 Statistical Rationale", prd.get("statistical_rationale", ""))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", styles["BodyTextCustom"]))
    doc.build(story)
    buffer.seek(0)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# -------------------------
# Page setup & styling
# -------------------------
st.set_page_config(page_title="A/B Test Architect", layout="wide")
st.markdown(
    """
<style>
/* Page-level */
.blue-section {background-color: #f6f9ff; padding: 14px; border-radius: 10px; margin-bottom: 14px;}
.green-section {background-color: #f7fff7; padding: 14px; border-radius: 10px; margin-bottom: 14px;}
.section-title {font-size: 1.15rem; font-weight: 700; color: #0b63c6; margin-bottom: 6px;}
.small-muted { color: #7a7a7a; font-size: 13px; }

/* Final PRD preview styling — more polished */
.prd-card {
  background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
  border-radius: 14px;
  padding: 28px;
  box-shadow: 0 12px 36px rgba(13,60,120,0.08);
  border: 1px solid rgba(13,60,120,0.06);
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* Header */
.prd-header {
  display:flex;
  align-items:center;
  gap:18px;
  margin-bottom: 24px;
}
.prd-logo {
  width:84px;
  height:84px;
  background: linear-gradient(135deg,#0b63c6,#3ac2ff);
  color: white;
  border-radius: 14px;
  display:flex;
  align-items:center;
  justify-content:center;
  font-weight:800;
  font-size:28px;
  box-shadow: 0 10px 26px rgba(10,120,200,0.12);
}
.prd-title { font-size:24px; font-weight:800; color:#052a4a; margin-bottom:4px; }
.prd-subtitle { font-size:16px; color:#475569; margin-top:3px; }

/* Section headings inside preview */
.prd-section { margin-top:20px; margin-bottom:12px; }
.prd-section h3 {
  margin:0;
  font-size:18px;
  color:#0b63c6;
  font-weight:700;
  padding-bottom: 6px;
  border-bottom: 2px solid #e0e7ff;
}
.prd-body { font-size:15px; color:#334155; line-height:1.7; white-space: pre-wrap; margin-top:10px; }

/* bullets */
.prd-body ul { padding-left:20px; margin-top:8px; }
.prd-body li { margin-bottom: 6px; }

/* meta */
.prd-meta { color:#6b7280; font-size:13px; margin-top:8px; }

/* Footer */
.prd-footer {
  margin-top: 28px;
  padding-top: 16px;
  border-top: 1px solid #e2e8f0;
  text-align: center;
  font-size: 12px;
  color: #94a3b8;
}
</style>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800&display=swap" rel="stylesheet">
""",
    unsafe_allow_html=True,
)

st.title("💡 A/B Test Architect — AI-assisted experiment PRD generator")
st.markdown("Create experiment PRDs, hypotheses, stats, and sample-size guidance — faster and with guardrails.")

# -------------------------
# Session state defaults
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
if "calculate_now" not in st.session_state:
    st.session_state.calculate_now = False
if "metrics_table" not in st.session_state:
    st.session_state.metrics_table = None

# -------------------------
# INPUTS: Business Context (in an expander)
# -------------------------
with st.expander("💡 Product Context (click to expand)", expanded=True):
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

# -------------------------
# INPUTS: Metric Details (in an expander)
# -------------------------
with st.expander("🎯 Metric Improvement Objective (click to expand)", expanded=True):
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
    
    if metric_type == "Conversion Rate" and metric_unit != "%":
        st.warning("For 'Conversion Rate' metric type, the unit should be '%'.")
        metric_inputs_valid = False

# -------------------------
# GENERATE PLAN AREA
# -------------------------
with st.expander("🧠 Generate Experiment Plan", expanded=True):
    create_header_with_help("Generate Experiment Plan", "When ready, click Generate to call the LLM and create a plan.", icon="🧠")
    sanitized_metric_name = sanitize_text(exact_metric)
    # safe expected lift calculation
    try:
        if current_value and current_value != 0:
            expected_lift_val = round(((target_value - current_value) / current_value) * 100, 2)
            mde_default = round(abs((target_value - current_value) / current_value) * 100, 2)
        else:
            expected_lift_val = 0.0
            mde_default = 5.0
    except Exception:
        expected_lift_val = 0.0
        mde_default = 5.0

    formatted_current = format_value_with_unit(current_value, metric_unit) if sanitized_metric_name else ""
    formatted_target = format_value_with_unit(target_value, metric_unit) if sanitized_metric_name else ""
    goal_with_units = f"I want to improve {sanitized_metric_name} from {formatted_current} to {formatted_target}." if sanitized_metric_name else ""

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
        # Reset selection state on new generation
        st.session_state.selected_index = None
        st.session_state.hypothesis_confirmed = False
        st.session_state.calc_locked = False
        st.session_state.locked_stats = {}

        context = {
            "type": product_type,
            "users": user_base_choice,
            "metric": metric_focus,
            "notes": product_notes,
            "exact_metric": sanitized_metric_name,
            "current_value": current_value,
            "target_value": target_value,
            "expected_lift": expected_lift_val,
            "minimum_detectable_effect": mde_default,
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

# -------------------------
# Calculator (Sample size)
# -------------------------
if st.session_state.get("ai_parsed") is not None or st.session_state.get("output"):
    st.markdown("<hr>", unsafe_allow_html=True)
    with st.expander("🔢 A/B Test Calculator: Fine-tune sample size", expanded=True):
        calc_mde = st.session_state.get("calc_mde", st.session_state.get("ai_parsed", {}).get("success_criteria", {}).get("MDE", mde_default) if st.session_state.get("ai_parsed") else mde_default)
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

        if metric_type == "Numeric Value" and std_dev is not None:
            st.info(f"Standard Deviation pre-filled: {std_dev}")

        # visually grouped action buttons
        col_act1, col_act2 = st.columns([1, 1])
        with col_act1:
            refresh_btn = st.button("Refresh Calculator")
        with col_act2:
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
            st.session_state.calculated_duration_days = (users_to_test / dau) if dau > 0 and users_to_test else float("inf")

        if st.session_state.get("calculated_sample_size_per_variant") and st.session_state.get("calculated_total_sample_size"):
            st.markdown("---")
            st.metric("Users Per Variant", f"{st.session_state.calculated_sample_size_per_variant:,} users")
            st.metric("Total Sample Size", f"{st.session_state.calculated_total_sample_size:,} users")
            duration_display = f"{st.session_state.calculated_duration_days:,.0f} days" if np.isfinite(st.session_state.calculated_duration_days) else "∞"
            st.metric("Estimated Test Duration", duration_display)
            st.caption("Assumes all DAU are eligible and evenly split across variants.")
        else:
            st.info("Click 'Refresh Calculator' to compute sample size with current inputs.")

        if lock_btn:
            if st.session_state.get("calculated_sample_size_per_variant") is not None:
                st.session_state.calc_locked = True
                st.session_state.locked_stats = {
                    "confidence_level": st.session_state.last_calc_confidence,
                    "MDE": st.session_state.last_calc_mde,
                    "sample_size_required": st.session_state.calculated_total_sample_size,
                    "users_per_variant": st.session_state.calculated_sample_size_per_variant,
                    "estimated_test_duration_days": st.session_state.calculated_duration_days if np.isfinite(st.session_state.calculated_duration_days) else 'Not applicable',
                }
                st.success("Calculator values locked and will be used in the final plan.")
            else:
                st.error("Cannot lock values. Run the calculator first.")

st.markdown("<hr>", unsafe_allow_html=True)

# -------------------------
# DISPLAY AI-GENERATED PLAN (editable + final view)
# -------------------------
if st.session_state.get("ai_parsed") is None and st.session_state.get("output"):
    st.markdown("<div class='blue-section'>", unsafe_allow_html=True)
    create_header_with_help("Raw LLM Output (fix JSON here)", "When parsing fails you'll see the raw LLM output — edit it then click Parse JSON.", icon="🛠️")
    raw_edit = st.text_area("Raw LLM output / edit here", value=st.session_state.get("output", ""), height=400, key="raw_llm_edit")
    if st.button("Parse JSON"):
        parsed_try = extract_json(st.session_state.get("raw_llm_edit", raw_edit))
        if parsed_try:
            st.session_state.ai_parsed = parsed_try
            st.success("Manual parse succeeded — plan is now usable.")
        else:
            st.error("Manual parse failed — edit the text and try again.")
    st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.get("ai_parsed"):
    plan = st.session_state.ai_parsed
    # ensure context exists
    if "context" not in st.session_state:
        st.session_state.context = {"metric_unit": metric_unit}

    unit = st.session_state.context.get("metric_unit", metric_unit)

    st.markdown("<div class='green-section'>", unsafe_allow_html=True)
    create_header_with_help("Inferred Product Goal", "The AI's interpretation of your goal. Edit if needed.", icon="🎯")
    safe_display(post_process_llm_text(goal_with_units, unit))

    create_header_with_help("Problem Statement", "Clear description of the gap and why it matters.", icon="🧩")
    problem_statement = post_process_llm_text(plan.get("problem_statement", ""), unit)
    st.markdown(problem_statement or "⚠️ Problem statement not generated by the model.")
    with st.expander("Edit Problem Statement"):
        st.text_area("Problem Statement (edit)", value=problem_statement, key="editable_problem", height=160)

    create_header_with_help("Hypotheses", "Testable hypotheses with full details.", icon="🧪")
    hypotheses = plan.get("hypotheses", [])
    if not hypotheses or not isinstance(hypotheses, list):
        st.warning("No hypotheses found in the generated plan.")
        hypotheses = []

    # Display hypotheses with expandable details
    for i, h in enumerate(hypotheses):
        with st.container():
            cols = st.columns([5, 1])
            with cols[0]:
                with st.expander(f"H{i+1}: {h.get('hypothesis', '')[:60]}...", expanded=False):
                    st.markdown(f"**Hypothesis:** {h.get('hypothesis', 'N/A')}")
                    st.markdown(f"**Rationale:** {h.get('rationale', h.get('behavioral_basis', 'N/A'))}")
                    st.markdown(f"**Example Implementation:** {h.get('example_implementation', 'N/A')}")
            with cols[1]:
                if st.button("Select", key=f"select_{i}"):
                    st.session_state.selected_index = i
                    st.session_state.hypothesis_confirmed = True

    if st.session_state.get("hypothesis_confirmed") and st.session_state.get("selected_index") is not None:
        selected_hypo = hypotheses[st.session_state.selected_index]
        
        st.subheader("🔍 Selected Hypothesis Details")
        cols = st.columns([4, 1])
        with cols[0]:
            st.markdown(f"**Hypothesis:** {selected_hypo.get('hypothesis', 'N/A')}")
            st.markdown(f"**Rationale:** {selected_hypo.get('rationale', selected_hypo.get('behavioral_basis', 'N/A'))}")
            st.markdown(f"**Example:** {selected_hypo.get('example_implementation', 'N/A')}")

            # Hypothesis editing
            with st.expander("✏️ Edit This Hypothesis"):
                edited_hypo = {
                    "hypothesis": st.text_area(
                        "Hypothesis Text",
                        value=selected_hypo.get("hypothesis", ""),
                        key="editable_hypothesis"
                    ),
                    "rationale": st.text_area(
                        "Rationale",
                        value=selected_hypo.get("rationale", selected_hypo.get("behavioral_basis", "")),
                        key="editable_rationale"
                    ),
                    "example_implementation": st.text_area(
                        "Implementation Example",
                        value=selected_hypo.get("example_implementation", ""),
                        key="editable_example"
                    )
                }
                
                if st.button("Save Edits"):
                    hypotheses[st.session_state.selected_index] = edited_hypo
                    st.success("Hypothesis updated!")

        with cols[1]:
            if st.button("Clear Selection"):
                st.session_state.selected_index = None
                st.session_state.hypothesis_confirmed = False

    # Metrics display and editing
    metrics = plan.get("metrics", [])
    if metrics:
        create_header_with_help("Metrics", "Primary and secondary metrics", icon="📏")
        
        try:
            # Normalize metrics data
            normalized = []
            for m in metrics:
                if isinstance(m, dict):
                    normalized.append({
                        "name": m.get("name", "Unnamed"),
                        "formula": m.get("formula", ""),
                        "importance": m.get("importance", "Medium")
                    })
                else:
                    normalized.append({
                        "name": str(m),
                        "formula": "",
                        "importance": "Medium"
                    })

            # Enhanced metrics editor
            if hasattr(st, "data_editor"):
                df_metrics = pd.DataFrame(normalized)
                edited_df = st.data_editor(
                    df_metrics,
                    column_config={
                        "importance": st.column_config.SelectboxColumn(
                            "Importance",
                            options=["High", "Medium", "Low"],
                            required=True
                        )
                    },
                    num_rows="dynamic",
                    key="metrics_data_editor"
                )
                st.session_state.metrics_table = edited_df.to_dict(orient="records")
            else:
                st.table(pd.DataFrame(normalized))
        except Exception as e:
            st.error(f"Metrics display error: {e}")
            st.json(metrics)

    # Segments display and editing
    segments = plan.get("segments", [])
    if segments:
        create_header_with_help("Segments", "User segments for analysis", icon="👥")
        st.markdown("\n".join([f"- {s}" for s in segments if str(s).strip()]))
        with st.expander("Edit Segments"):
            st.text_area(
                "Segments (one per line)",
                value="\n".join(segments),
                key="editable_segments",
                height=120
            )

    # Risks display and editing - FIXED VERSION
    risks = plan.get("risks_and_assumptions", [])
    if risks:
        create_header_with_help("Risks & Mitigations", "Potential issues and solutions", icon="⚠️")
        
        risk_text = []
        for r in risks:
            if isinstance(r, dict):
                risk_str = f"• {r.get('risk', 'Risk')}"
                if r.get('severity'):
                    risk_str += f" (Severity: {r['severity']})"
                if r.get('mitigation'):
                    risk_str += f"\n  → Mitigation: {r['mitigation']}"
                else:
                    risk_str += "\n  → Mitigation: To be determined"
                risk_text.append(risk_str)
            else:
                risk_text.append(f"• {str(r)}\n  → Mitigation: To be determined")
        
        st.markdown("\n\n".join(risk_text))
        
        with st.expander("Edit Risks"):
            edited_risks = []
            for i, r in enumerate(risks):
                cols = st.columns([3, 1, 3])
                with cols[0]:
                    risk = st.text_input(
                        f"Risk {i+1}", 
                        value=r.get('risk', '') if isinstance(r, dict) else str(r),
                        key=f"risk_{i}"
                    )
                with cols[1]:
                    severity = st.selectbox(
                        "Severity",
                        ["High", "Medium", "Low"],
                        index=["High", "Medium", "Low"].index(
                            r.get('severity', 'Medium') if isinstance(r, dict) else 'Medium'
                        ),
                        key=f"severity_{i}"
                    )
                with cols[2]:
                    mitigation = st.text_input(
                        "Mitigation",
                        value=r.get('mitigation', 'To be determined') if isinstance(r, dict) else 'To be determined',
                        key=f"mitigation_{i}"
                    )
                edited_risks.append({
                    "risk": risk,
                    "severity": severity,
                    "mitigation": mitigation
                })

    # Next steps - GUARANTEED to appear
    next_steps = plan.get("next_steps", [])
    if not next_steps:  # Fallback if empty
        next_steps = [
            "Finalize experiment design",
            "Implement tracking metrics",
            "Recruit users for testing"
        ]

    create_header_with_help("Next Steps", "Action items to execute the test", icon="✅")
    st.markdown("\n".join([f"- {step}" for step in next_steps]))
    with st.expander("Edit Next Steps"):
        edited_steps = st.text_area(
            "Next Steps (one per line)",
            value="\n".join(next_steps),
            key="editable_next_steps",
            height=120
        )

    # Calculations Display - RESTORED SECTION
    if st.session_state.get("calculated_sample_size_per_variant") and st.session_state.get("calculated_total_sample_size"):
        st.markdown("---")
        create_header_with_help("Experiment Calculations", "Statistical requirements for valid results", icon="🧮")
        
        cols = st.columns(3)
        with cols[0]:
            st.metric("Users Per Variant", f"{st.session_state.calculated_sample_size_per_variant:,}")
        with cols[1]:
            st.metric("Total Sample Size", f"{st.session_state.calculated_total_sample_size:,}")
        with cols[2]:
            duration = st.session_state.calculated_duration_days
            duration_str = f"{duration:,.0f} days" if np.isfinite(duration) else "∞"
            st.metric("Estimated Duration", duration_str)
        
        st.caption("Assumes all DAU are eligible and evenly split across variants.")

    # Build final PRD dict
    prd_dict = {
        "goal": goal_with_units,
        "problem_statement": st.session_state.get("editable_problem", problem_statement),
        "hypotheses": [
            {
                "hypothesis": h.get("hypothesis", ""),
                "rationale": h.get("rationale", h.get("behavioral_basis", "")),
                "example_implementation": h.get("example_implementation", ""),
                **({"behavioral_basis": h["behavioral_basis"]} if "behavioral_basis" in h else {})
            } for h in hypotheses
        ],
        "metrics": st.session_state.get("metrics_table", metrics),
        "segments": [
            s.strip() for s in 
            st.session_state.get("editable_segments", "").split("\n") 
            if s.strip()
        ],
        "success_criteria": st.session_state.get("locked_stats", plan.get("success_criteria", {})),
        "risks_and_assumptions": edited_risks if 'edited_risks' in locals() else [],
        "next_steps": [
            ns.strip() for ns in 
            st.session_state.get("editable_next_steps", "").split("\n")
            if ns.strip()
        ],
        "statistical_rationale": plan.get("statistical_rationale", "")
    }

    # Final PRD preview - FIXED VERSION
    create_header_with_help("Final PRD Preview", "Production-ready experiment document", icon="📄")
    
    def render_list(items):
        if not items or not any(str(item).strip() for item in items):
            return "None specified"
        html = "<ul>"
        for item in items:
            if isinstance(item, dict):
                item_text = f"<b>{sanitize_text(item.get('name','Unnamed'))}:</b> {sanitize_text(item.get('formula',''))}"
            else:
                item_text = sanitize_text(item)
            if item_text:
                html += f"<li>{item_text}</li>"
        html += "</ul>"
        return html

    # Fixed hypotheses rendering
    hypotheses_html = "<ol>"
    for i, h in enumerate(prd_dict["hypotheses"], 1):
        hypotheses_html += f"""
        <li>
            <b>{h.get('hypothesis', '')}</b><br>
            <i>Rationale:</i> {h.get('rationale', '')}<br>
            <i>Example:</i> {h.get('example_implementation', '')}
        </li>
        """
    hypotheses_html += "</ol>"

    # Fixed risks rendering
    risks_html = "<ul>"
    for r in prd_dict.get("risks_and_assumptions", []):
        risks_html += f"""
        <li>
            {r.get('risk', '')} <i>(Severity: {r.get('severity', 'Medium')})</i><br>
            → Mitigation: {r.get('mitigation', 'To be determined')}
        </li>
        """
    risks_html += "</ul>"

    prd_html = f"""
    <div class="prd-card">
        <div class="prd-header">
            <div class="prd-logo">A/B</div>
            <div>
                <div class="prd-title">Experiment PRD</div>
                <div class="prd-subtitle">{sanitize_text(prd_dict.get('goal', ''))}</div>
            </div>
        </div>
        <div class="prd-section">
            <h3>🎯 Problem Statement</h3>
            <div class="prd-body">{sanitize_text(prd_dict.get('problem_statement', ''))}</div>
        </div>
        <div class="prd-section">
            <h3>🧪 Hypotheses</h3>
            <div class="prd-body">{hypotheses_html}</div>
        </div>
        <div class="prd-section">
            <h3>📊 Metrics</h3>
            <div class="prd-body">{render_list(prd_dict.get('metrics', []))}</div>
        </div>
        <div class="prd-section">
            <h3>⚠️ Risks & Mitigations</h3>
            <div class="prd-body">{risks_html}</div>
        </div>
        <div class="prd-section">
            <h3>🚀 Next Steps</h3>
            <div class="prd-body">{render_list(prd_dict.get('next_steps', []))}</div>
        </div>
        <div class="prd-footer">
            Generated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}
        </div>
    </div>
    """
    st.markdown(prd_html, unsafe_allow_html=True)

    # Download buttons
    col_dl1, col_dl2, col_dl3, col_dl4 = st.columns([1,1,1,1])
    with col_dl1:
        st.download_button(
            "📄 Download PRD (.txt)",
            json.dumps(prd_dict, indent=2),
            file_name="experiment_prd.txt"
        )
    with col_dl2:
        st.download_button(
            "📥 Download Plan (.json)",
            json.dumps(prd_dict, indent=2),
            file_name="experiment_plan.json"
        )
    with col_dl3:
        html_blob = f"""
        <!doctype html>
        <html>
        <head><meta charset='utf-8'></head>
        <body>
            <h1>Experiment PRD</h1>
            <pre>{json.dumps(prd_dict, indent=2)}</pre>
        </body>
        </html>
        """
        st.download_button(
            "🌐 Download PRD (.html)",
            html_blob,
            file_name="experiment_prd.html"
        )
    with col_dl4:
        if REPORTLAB_AVAILABLE:
            pdf_bytes = generate_pdf_bytes_from_prd_dict(prd_dict)
            if pdf_bytes:
                st.download_button(
                    "📁 Download PRD (.pdf)",
                    pdf_bytes,
                    file_name="experiment_prd.pdf",
                    mime="application/pdf"
                )
            else:
                st.warning("PDF generation failed")
        else:
            st.info("PDF export requires reportlab")

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
        keys_to_clear = [
            "output", "ai_parsed", "calculated_sample_size_per_variant",
            "calculated_total_sample_size", "calculated_duration_days",
            "locked_stats", "calc_locked", "selected_index",
            "hypothesis_confirmed", "last_llm_hash", "context",
            "metrics_table", "editable_problem", "editable_hypothesis",
            "editable_rationale", "editable_example", "editable_segments",
            "editable_risks", "editable_next_steps"
        ]
        for k in keys_to_clear:
            if k in st.session_state:
                del st.session_state[k]
        st.success("Session cleared.")
