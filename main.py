# main.py — Final Certified Version (A/B Test Architect)
import streamlit as st
import json
import re
import os
import pandas as pd
from typing import Any, Dict, Optional, Tuple, List
from prompt_engine import generate_experiment_plan, generate_hypothesis_details
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize
import numpy as np
import hashlib
from datetime import datetime
from io import BytesIO
import ast
from streamlit_modal import Modal

# PDF Export Setup
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

# --- Helper Functions ---
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

def html_sanitize(text: Any) -> str:
    if text is None: return ""
    text = str(text)
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    text = text.replace("'", "&apos;")
    return text

def generate_problem_statement(plan: Dict, current: float, target: float, unit: str) -> str:
    base = plan.get("problem_statement", "")
    if not base.strip():
        return base
    metric_str = f" (current: {format_value_with_unit(current, unit)} → target: {format_value_with_unit(target, unit)})"
    if metric_str not in base:
        sentences = base.split('.')
        if len(sentences) > 1:
            sentences[0] = sentences[0].strip() + metric_str + "."
            return '.'.join(sentences)
        return base + metric_str
    return base

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
    if text is None:
        st.error("No output returned from LLM.")
        return None
    if isinstance(text, dict):
        return text
    if isinstance(text, list):
        if all(isinstance(i, dict) for i in text):
            return {"items": text}
        st.error("LLM returned a JSON list when an object was expected.")
        return None
    try:
        raw = str(text)
    except Exception as e:
        st.error(f"Unexpected LLM output type: {e}")
        return None
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
    try:
        parsed_ast = ast.literal_eval(raw)
        if isinstance(parsed_ast, dict):
            return parsed_ast
        if isinstance(parsed_ast, list) and all(isinstance(i, dict) for i in parsed_ast):
            return {"items": parsed_ast}
    except Exception:
        pass
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
                else:
                    raise ValueError("Extracted JSON parsed as a list but was expected to be an object.")
            except Exception:
                try:
                    converted = _safe_single_to_double_quotes(candidate_clean)
                    parsed = json.loads(converted)
                    if isinstance(parsed, dict):
                        return parsed
                    if isinstance(parsed, list) and all(isinstance(i, dict) for i in parsed):
                        return {"items": parsed}
                    else:
                        raise ValueError("Extracted JSON with single quotes parsed but was not an object.")
                except Exception:
                    st.error("Could not parse extracted JSON block. See snippet below.")
                    st.code(candidate_clean[:3000] + ("..." if len(candidate_clean) > 3000 else ""))
                    return None
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
    units_with_space = ["USD", "count", "minutes", "hours", "days", "INR"]
    if unit in units_with_space:
        return f"{v_str} {unit}"
    else:
        return f"{v_str}{unit}"
        
def _parse_value_from_text(text: str, default_unit: str = '%') -> Tuple[Optional[float], str]:
    text = sanitize_text(text)
    match = re.match(r"([\d\.]+)\s*(\w+|%)?", text)
    if match:
        value = float(match.group(1))
        unit = match.group(2) if match.group(2) else default_unit
        return value, unit
    try:
        return float(text), default_unit
    except ValueError:
        return None, default_unit

def calculate_sample_size(baseline, mde, alpha, power, num_variants, metric_type, std_dev=None) -> Tuple[Optional[int], Optional[int]]:
    try:
        if baseline is None or mde is None:
            return None, None
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
    def pdf_sanitize(text: Any) -> str:
        if text is None: return ""
        return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    def add_section_header(title: str):
        story.append(Spacer(1, 12))
        story.append(Paragraph(title, styles["SectionHeading"]))
        story.append(Spacer(1, 6))
    story.append(Paragraph(title, styles["PRDTitle"]))
    add_section_header("1. Problem Statement")
    story.append(Paragraph(pdf_sanitize(prd.get("problem_statement", "")), styles["BodyTextCustom"]))
    add_section_header("2. Hypotheses")
    for idx, h in enumerate(prd.get("hypotheses", [])):
        story.append(Paragraph(f"<b>Hypothesis {idx + 1}:</b> {pdf_sanitize(h.get('hypothesis', ''))}", styles["BodyTextCustom"]))
        story.append(Paragraph(f"<b>Rationale:</b> {pdf_sanitize(h.get('rationale', ''))}", styles["BodyTextCustom"]))
        story.append(Paragraph(f"<b>Example Implementation:</b> {pdf_sanitize(h.get('example_implementation', ''))}", styles["BodyTextCustom"]))
        story.append(Paragraph(f"<b>Behavioral Basis:</b> {pdf_sanitize(h.get('behavioral_basis', ''))}", styles["BodyTextCustom"]))
        story.append(Spacer(1, 10))
    add_section_header("3. Variants")
    for v in prd.get("variants", []):
        story.append(Paragraph(f"<b>Control:</b> {pdf_sanitize(v.get('control', ''))}", styles["BodyTextCustom"]))
        story.append(Paragraph(f"<b>Variation:</b> {pdf_sanitize(v.get('variation', ''))}", styles["BodyTextCustom"]))
        story.append(Spacer(1, 10))
    add_section_header("4. Metrics")
    metrics_data = [['Name', 'Formula', 'Importance']]
    for m in prd.get("metrics", []):
        metrics_data.append([
            pdf_sanitize(m.get('name', '')),
            pdf_sanitize(m.get('formula', '')),
            pdf_sanitize(m.get('importance', ''))
        ])
    if len(metrics_data) > 1:
        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0f0f0')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#d0d0d0')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10.5),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ])
        metrics_table = Table(metrics_data, colWidths=[2*inch, 3*inch, 1.5*inch])
        metrics_table.setStyle(table_style)
        story.append(metrics_table)
    else:
        story.append(Paragraph("No metrics defined.", styles["BodyTextCustom"]))
    add_section_header("5. Success Criteria & Statistical Rationale")
    criteria = prd.get("success_criteria", {})
    story.append(Paragraph(f"<b>Confidence Level:</b> {criteria.get('confidence_level', '')}%", styles["BodyTextCustom"]))
    story.append(Paragraph(f"<b>Minimum Detectable Effect (MDE):</b> {criteria.get('MDE', '')}%", styles["BodyTextCustom"]))
    story.append(Paragraph(f"<b>Statistical Rationale:</b> {pdf_sanitize(prd.get('statistical_rationale', ''))}", styles["BodyTextCustom"]))
    story.append(Paragraph(f"<b>Benchmark:</b> {pdf_sanitize(criteria.get('benchmark', ''))}", styles["BodyTextCustom"]))
    story.append(Paragraph(f"<b>Monitoring:</b> {pdf_sanitize(criteria.get('monitoring', ''))}", styles["BodyTextCustom"]))
    add_section_header("6. Risks and Assumptions")
    risks_data = [['Risk', 'Severity', 'Mitigation']]
    for r in prd.get("risks_and_assumptions", []):
        risks_data.append([
            pdf_sanitize(r.get('risk', '')),
            pdf_sanitize(r.get('severity', '')),
            pdf_sanitize(r.get('mitigation', ''))
        ])
    if len(risks_data) > 1:
        risks_table = Table(risks_data, colWidths=[2.5*inch, 1*inch, 3*inch])
        risks_table.setStyle(table_style)
        story.append(risks_table)
    else:
        story.append(Paragraph("No risks defined.", styles["BodyTextCustom"]))
    add_section_header("7. Next Steps")
    next_steps_data = [['Action']]
    for step in prd.get("next_steps", []):
        next_steps_data.append([pdf_sanitize(step)])
    if len(next_steps_data) > 1:
        next_steps_table = Table(next_steps_data, colWidths=[6.5*inch])
        next_steps_table.setStyle(table_style)
        story.append(next_steps_table)
    else:
        story.append(Paragraph("No next steps defined.", styles["BodyTextCustom"]))
    doc.build(story)
    buffer.seek(0)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# --- Streamlit UI Code ---
st.set_page_config(page_title="A/B Test Architect", layout="wide")
st.markdown(
    """
<style>
.blue-section {background-color: #f6f9ff; padding: 14px; border-radius: 10px; margin-bottom: 14px;}
.green-section {background-color: #f7fff7; padding: 14px; border-radius: 10px; margin-bottom: 14px;}
.section-title {font-size: 1.15rem; font-weight: 700; color: #0b63c6; margin-bottom: 6px;}
.small-muted { color: #7a7a7a; font-size: 13px; }
.prd-card {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    max-width: 900px;
    width: 100%;
    background: #ffffff;
    border-radius: 16px;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    padding: 2.5rem;
    border: 1px solid #e5e7eb;
}
.prd-header {
    display: flex;
    align-items: center;
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid #e5e7eb;
}
.logo-wrapper {
    background: #0b63c6;
    color: white;
    padding: 1.5rem 2rem;
    border-radius: 12px;
    font-weight: 800;
    font-size: 2.5rem;
    line-height: 1;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    margin-right: 1.5rem;
    transform: rotate(-3deg);
}
.header-text h1 {
    margin: 0;
    font-size: 2.25rem;
    font-weight: 900;
    color: #052a4a;
}
.header-text p {
    margin: 0.25rem 0 0;
    font-size: 1.125rem;
    color: #4b5563;
}
.prd-section {
    margin-bottom: 2rem;
}
.prd-section-title {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1rem;
    color: #0b63c6;
}
.prd-section-title h2 {
    margin: 0;
    font-size: 1.5rem;
    font-weight: 700;
}
.prd-section-title svg {
    color: #0b63c6;
    width: 24px;
    height: 24px;
}
.prd-section-content {
    background: #f3f8ff;
    border-left: 4px solid #0b63c6;
    padding: 1.5rem;
    border-radius: 8px;
    line-height: 1.8;
    color: #1f2937;
}
.problem-statement {
    font-weight: 500;
    font-style: italic;
}
.hypotheses ol, .risks ul, .metrics ul, .next-steps ul, .stats-list {
    padding-left: 1.5rem;
    margin: 0.5rem 0 0;
}
.hypotheses li, .risks li, .metrics li, .next-steps li, .stats-list li {
    margin-bottom: 1.5rem;
    padding: 1rem;
    background: #fdfefe;
    border-radius: 8px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    border: 1px solid #e5e7eb;
    position: relative;
}
.hypotheses li:last-child, .risks li:last-child, .metrics li:last-child, .next-steps li:last-child, .stats-list li:last-child {
    margin-bottom: 0;
}
.hypotheses li p {
    margin: 0;
}
.hypotheses li p strong {
    display: block;
    margin-bottom: 0.5rem;
    color: #052a4a;
}
.hypotheses li p.rationale, .hypotheses li p.example {
    font-size: 0.9rem;
    color: #4b5563;
}
.hypothesis-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #052a4a;
}
.metrics li span.importance {
    font-weight: 600;
    color: #0b63c6;
}
.risks li span.severity {
    font-weight: 600;
}
.risks li span.severity.high { color: #ef4444; }
.risks li span.severity.medium { color: #f97316; }
.risks li span.severity.low { color: #22c55e; }
.prd-footer {
    margin-top: 2rem;
    padding-top: 1.5rem;
    border-top: 1px solid #e5e7eb;
    text-align: center;
    font-size: 0.875rem;
    color: #6b7280;
}
.edit-buttons {
    display: flex;
    justify-content: flex-end;
    gap: 0.5rem;
    margin-top: 1rem;
}
</style>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800&display=swap" rel="stylesheet">
""",
    unsafe_allow_html=True,
)

st.title("💡 A/B Test Architect — AI-assisted experiment PRD generator")
st.markdown("Create experiment PRDs, hypotheses, stats, and sample-size guidance — faster and with guardrails.")

# --- Session State Management ---
if "output" not in st.session_state:
    st.session_state.output = None
if "ai_parsed" not in st.session_state:
    st.session_state.ai_parsed = None
if "calc_locked" not in st.session_state:
    st.session_state.calc_locked = False
if "locked_stats" not in st.session_state:
    st.session_state.locked_stats = {}
if "last_llm_hash" not in st.session_state:
    st.session_state.last_llm_hash = None
if "calculate_now" not in st.session_state:
    st.session_state.calculate_now = False
if "edit_modal_open" not in st.session_state:
    st.session_state.edit_modal_open = False
if "temp_plan_edit" not in st.session_state:
    st.session_state.temp_plan_edit = {}
if "stage" not in st.session_state:
    st.session_state.stage = "input"
if "hypotheses_from_llm" not in st.session_state:
    st.session_state.hypotheses_from_llm = []

# --- Input Sections (No Change) ---
with st.expander("💡 Product Context (click to expand)", expanded=True):
    create_header_with_help("Product Context", "Provide the product context and business goal so the AI can produce a focused experiment plan.", icon="💡")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        product_type = st.selectbox("Product Type", ["SaaS", "Consumer App", "E-commerce", "Marketplace", "Gaming", "Other"], index=0, help="What kind of product are you testing?")
        dau = st.number_input("Daily Active Users (DAU) *", min_value=1, value=10000, step=1000, help="The average number of daily active users for the product. This is used to calculate the experiment duration.")
        metric_focus = st.selectbox("Primary Metric Focus", ["Activation", "Retention", "Monetization", "Engagement", "Virality"], index=0, help="The general category of metrics you're trying to move.")
        product_notes = st.text_area("Anything unique about your product or users? (optional)", placeholder="e.g. seasonality, power users, drop-off at pricing", help="Optional context to inform better suggestions.")
    with col_b:
        strategic_goal = st.text_area("High-Level Business Goal *", placeholder="e.g., Increase overall revenue from our premium tier", help="This is the broader business goal the experiment supports.")
        user_persona = st.text_input("Target User Persona (optional)", placeholder="e.g., First-time users from India, iOS users, power users", help="Focus the plan on a specific user segment.")

with st.expander("🎯 Metric Improvement Objective (click to expand)", expanded=True):
    create_header_with_help("Metric Improvement Objective", "Provide the exact metric and current vs target values. Use the proper units.", icon="🎯")
    col_m1, col_m2 = st.columns([2, 2])
    with col_m1:
        exact_metric = st.text_input("Metric to Improve * (e.g. Activation Rate, ARPU, DAU/MAU)", help="Be specific — name the metric you want to shift.")
    with col_m2:
        metric_type = st.radio("Metric Type", ["Conversion Rate", "Numeric Value"], horizontal=True)
    
    col_unit, col_values = st.columns([1, 2])
    with col_unit:
        metric_unit = st.selectbox("Metric Unit", ["%", "USD", "INR", "minutes", "count", "other"], index=0, help="Choose the unit for clarity.")
    with col_values:
        current_value_raw = st.text_input("Current Metric Value *", placeholder="e.g., 55.0 or 55.0 INR")
        target_value_raw = st.text_input("Target Metric Value *", placeholder="e.g., 60.0 or 60.0 INR")

        current_value, current_unit = _parse_value_from_text(current_value_raw, metric_unit)
        target_value, target_unit = _parse_value_from_text(target_value_raw, metric_unit)

        if current_value is None and current_value_raw:
            st.error("Invalid format for Current Metric Value. Please enter a number.")
        if target_value is None and target_value_raw:
            st.error("Invalid format for Target Metric Value. Please enter a number.")

        std_dev = None
        if metric_type == "Numeric Value":
            std_dev_raw = st.text_input("Standard Deviation of Metric * (required for numeric metrics)", placeholder="e.g., 10.5", help="The standard deviation is required for numeric metrics to compute sample sizes.")
            std_dev, _ = _parse_value_from_text(std_dev_raw)
            if std_dev is None and std_dev_raw:
                st.error("Invalid format for Standard Deviation. Please enter a number.")

    metric_inputs_valid = True
    if current_value == target_value and current_value is not None:
        st.warning("The target metric must be different from the current metric to measure change. Please adjust one or the other.")
        metric_inputs_valid = False
    
    if metric_type == "Conversion Rate" and metric_unit != "%":
        st.warning("For 'Conversion Rate' metric type, the unit should be '%'.")
        metric_inputs_valid = False

with st.expander("🧠 Generate Experiment Plan", expanded=True):
    create_header_with_help("Generate Experiment Plan", "When ready, click Generate to call the LLM and create a plan.", icon="🧠")
    sanitized_metric_name = sanitize_text(exact_metric)
    
    try:
        if current_value is not None and current_value != 0:
            expected_lift_val = round(((target_value - current_value) / current_value) * 100, 2)
            mde_default = round(abs((target_value - current_value) / current_value) * 100, 2)
        else:
            expected_lift_val = 0.0
            mde_default = 5.0
    except Exception:
        expected_lift_val = 0.0
        mde_default = 5.0

    formatted_current = format_value_with_unit(current_value, metric_unit) if sanitized_metric_name and current_value is not None else ""
    formatted_target = format_value_with_unit(target_value, metric_unit) if sanitized_metric_name and target_value is not None else ""
    goal_with_units = f"I want to improve {sanitized_metric_name} from {formatted_current} to {formatted_target}." if sanitized_metric_name else ""

    required_ok = all(
        [
            product_type,
            dau,
            metric_focus,
            sanitized_metric_name,
            metric_inputs_valid,
            strategic_goal,
            current_value is not None,
            target_value is not None
        ]
    )
    generate_btn = st.button("Generate Plan", disabled=not required_ok)
    
    if generate_btn:
        st.session_state.stage = "problem_statement"
        st.session_state.ai_parsed = None
        st.session_state.hypotheses_from_llm = []
        st.session_state.temp_plan_edit = {}
        st.session_state.calc_locked = False
        st.session_state.locked_stats = {}
        
        context = {
            "type": product_type,
            "users": f"{dau} DAU",
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
                parsed = extract_json(raw_llm)
                if parsed:
                    st.session_state.ai_parsed = parsed
                    st.session_state.temp_plan_edit = parsed.copy()
                    st.session_state.hypotheses_from_llm = parsed.get("hypotheses", [])
                    st.success("Plan generated successfully — let's refine it step-by-step.")
                else:
                    st.error("Plan generation failed. Please check inputs and try again.")
                    st.session_state.stage = "input"
            except Exception as e:
                st.error(f"LLM generation failed: {e}")
                st.session_state.stage = "input"

# --- Main Guided Workflow ---
if st.session_state.get("ai_parsed"):
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 🚀 Refine Your Experiment Plan")

    if st.session_state.stage == "problem_statement":
        st.subheader("Step 1: Problem Statement")
        st.info("Review and refine the problem statement. This sets the foundation for your experiment.")
        
        problem_statement = st.session_state.ai_parsed.get('problem_statement', '')
        st.session_state.ai_parsed['problem_statement'] = st.text_area(
            "Problem Statement",
            value=problem_statement,
            height=150,
            label_visibility="collapsed"
        )
        
        col_ps_1, col_ps_2 = st.columns([1,1])
        with col_ps_1:
            if st.button("Save Problem Statement"):
                st.session_state.stage = "hypotheses"
                st.experimental_rerun()

    elif st.session_state.stage == "hypotheses":
        st.subheader("Step 2: Choose or Create a Hypothesis")
        st.info("Select one of the AI-generated hypotheses below, or create your own to get a fresh perspective.")
        
        st.session_state.ai_parsed['hypotheses'] = []
        
        hyp_cols = st.columns(3)
        for i, h in enumerate(st.session_state.hypotheses_from_llm):
            with hyp_cols[i]:
                st.markdown(f"**Hypothesis {i+1}**")
                st.markdown(f"*{h.get('hypothesis', '')}*")
                if st.button(f"Select Hypothesis {i+1}", key=f"select_hyp_{i}"):
                    st.session_state.ai_parsed['hypotheses'].append(h)
                    st.session_state.stage = "full_plan"
                    st.experimental_rerun()
        
        st.markdown("---")
        st.markdown("Or, if none of these fit...")
        
        with st.container():
            st.markdown("### Create Your Own Hypothesis")
            new_hyp_text = st.text_input(
                "Enter your new hypothesis:",
                placeholder="e.g., If we change the button color to red, the click-through rate will increase.",
                key="new_hyp_text"
            )
            if st.button("Generate Details for This Hypothesis"):
                if new_hyp_text:
                    with st.spinner("Generating hypothesis details..."):
                        context = {
                            "strategic_goal": strategic_goal,
                            "metric_to_improve": exact_metric,
                            "problem_statement": st.session_state.ai_parsed.get('problem_statement', ''),
                            "user_persona": user_persona,
                        }
                        
                        try:
                            # New, focused LLM call for hypothesis details
                            hyp_details_raw = generate_hypothesis_details(new_hyp_text, context)
                            hyp_details_parsed = extract_json(hyp_details_raw)
                            if hyp_details_parsed:
                                # Update the main plan with the new hypothesis
                                st.session_state.ai_parsed['hypotheses'].append(hyp_details_parsed)
                                st.session_state.stage = "full_plan"
                                st.experimental_rerun()
                            else:
                                st.error("Failed to generate details for your hypothesis. Please try again.")
                        except Exception as e:
                            st.error(f"LLM call failed: {e}")


    elif st.session_state.stage == "full_plan":
        st.subheader("Step 3: Refine the Full Plan")
        st.info("Your experiment plan is ready! Now you can edit any of the sections, starting with the A/B test calculator.")
        
        # --- A/B Test Calculator Section (Unchanged, now part of the flow) ---
        with st.expander("🔢 A/B Test Calculator: Fine-tune sample size", expanded=True):
            plan = st.session_state.ai_parsed
            calc_mde = st.session_state.get("calc_mde", plan.get("success_criteria", {}).get("MDE", 5.0))
            calc_conf = st.session_state.get("calc_confidence", plan.get("success_criteria", {}).get("confidence_level", 95))
            calc_power = st.session_state.get("calc_power", 80)
            calc_variants = st.session_state.get("calc_variants", 2)
            
            col1, col2 = st.columns(2)
            with col1:
                calc_mde = st.number_input("Minimum Detectable Effect (MDE) %", min_value=0.1, max_value=50.0, value=float(calc_mde), step=0.1, key="calc_mde_key")
                calc_conf = st.number_input("Confidence Level (%)", min_value=80, max_value=99, value=int(calc_conf), step=1, key="calc_conf_key")
            with col2:
                calc_power = st.number_input("Statistical Power (%)", min_value=70, max_value=95, value=int(calc_power), step=1, key="calc_power_key")
                calc_variants = st.number_input("Number of Variants (Control + Variations)", min_value=2, max_value=5, value=int(calc_variants), step=1, key="calc_variants_key")
            
            if metric_type == "Numeric Value" and std_dev is not None:
                st.info(f"Standard Deviation pre-filled: {std_dev}")

            col_act1, col_act2 = st.columns([1, 1])
            with col_act1:
                btn_label = "Calculate"
                refresh_btn = st.button(btn_label, key="calc_btn")
            with col_act2:
                lock_btn = False
                if st.session_state.get("calculated_sample_size_per_variant"):
                    lock_btn = st.button("Lock Values for Plan", key="lock_btn")

            if refresh_btn:
                alpha_calc = 1 - (calc_conf / 100.0)
                power_calc = calc_power / 100.0
                sample_per_variant, total_sample = calculate_sample_size(
                    baseline=current_value,
                    mde=calc_mde,
                    alpha=alpha_calc,
                    power=power_calc,
                    num_variants=calc_variants,
                    metric_type=metric_type,
                    std_dev=std_dev,
                )
                st.session_state.calculated_sample_size_per_variant = sample_per_variant
                st.session_state.calculated_total_sample_size = total_sample
                users_to_test = st.session_state.calculated_total_sample_size or 0
                st.session_state.calculated_duration_days = (users_to_test / dau) if dau > 0 and users_to_test else float("inf")

            if st.session_state.get("calculated_sample_size_per_variant"):
                st.markdown("---")
                st.markdown(f"**Sample Size per Variant:** {st.session_state.calculated_sample_size_per_variant:,}")
                st.markdown(f"**Total Sample Size:** {st.session_state.calculated_total_sample_size:,}")
                st.markdown(f"**Estimated Duration:** **{round(st.session_state.calculated_duration_days, 1)}** days")
                
            if lock_btn:
                st.session_state.calc_locked = True
                st.session_state.ai_parsed['success_criteria']['MDE'] = calc_mde
                st.session_state.ai_parsed['success_criteria']['confidence_level'] = calc_conf
                st.success("Calculator values locked into the plan!")

        # --- Final Plan Preview (Read-Only) ---
        plan = st.session_state.ai_parsed
        
        # Build HTML list items for Hypotheses, Risks, and Next Steps outside the main f-string
        hypotheses_html = "".join([f"""
            <li>
                <p class='hypothesis-title'>{html_sanitize(h.get('hypothesis', ''))}</p>
                <p class='rationale'><strong>Rationale:</strong> {html_sanitize(h.get('rationale', ''))}</p>
                <p class='example'><strong>Example:</strong> {html_sanitize(h.get('example_implementation', ''))}</p>
                <p class='behavioral-basis'><strong>Behavioral Basis:</strong> {html_sanitize(h.get('behavioral_basis', ''))}</p>
            </li>
        """ for h in plan.get("hypotheses", [])])

        risks_html = "".join([f"""
            <li>
                <strong>{html_sanitize(r.get('risk', ''))}</strong>
                <br>Severity: <span class='severity {r.get('severity', 'Medium').lower()}'>{html_sanitize(r.get('severity', ''))}</span>
                <br>Mitigation: {html_sanitize(r.get('mitigation', ''))}
            </li>
        """ for r in plan.get("risks_and_assumptions", [])])

        next_steps_html = "".join([f"<li>{html_sanitize(step)}</li>" for step in plan.get("next_steps", [])])

        metrics_html = "".join([f"""
            <li>
                <strong>{html_sanitize(m.get('name', ''))}</strong>
                <br>Formula: <code>{html_sanitize(m.get('formula', ''))}</code>
                <br>Importance: <span class='importance'>{html_sanitize(m.get('importance', ''))}</span>
            </li>
        """ for m in plan.get("metrics", [])])

        st.markdown(f"<div class='prd-card'>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="prd-header">
                <div class="logo-wrapper">A/B</div>
                <div class="header-text">
                    <h1>Experiment PRD</h1>
                    <p>An AI-generated plan for {sanitize_text(exact_metric)}</p>
                </div>
            </div>
            <div class="prd-section">
                <div class="prd-section-title"><h2>1. Problem Statement</h2></div>
                <div class="prd-section-content"><p class="problem-statement">{html_sanitize(plan.get("problem_statement", ""))}</p></div>
            </div>
            <div class="prd-section">
                <div class="prd-section-title"><h2>2. Hypotheses</h2></div>
                <div class="hypotheses">
                    <ol>
                        {hypotheses_html}
                    </ol>
                </div>
            </div>
            """, unsafe_allow_html=True,
        )
        st.markdown(f"<div class='prd-section-title'><h2>3. Variants</h2></div>", unsafe_allow_html=True)
        for v in plan.get("variants", []):
            st.markdown(f"**Control:** {html_sanitize(v.get('control', ''))} | **Variation:** {html_sanitize(v.get('variation', ''))}")
        
        st.markdown(
            f"""
            <div class="prd-section">
                <div class="prd-section-title"><h2>4. Metrics</h2></div>
                <div class="metrics">
                    <ul>
                        {metrics_html}
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="prd-section">
                <div class="prd-section-title"><h2>5. Success Criteria & Statistical Rationale</h2></div>
                <div class="stats-list">
                    <p><strong>Confidence:</strong> {plan.get('success_criteria', {}).get('confidence_level', '')}%</p>
                    <p><strong>MDE:</strong> {plan.get('success_criteria', {}).get('MDE', '')}%</p>
                    <p><strong>Statistical Rationale:</strong> {html_sanitize(plan.get('statistical_rationale', ''))}</p>
                </div>
            </div>
            <div class="prd-section">
                <div class="prd-section-title"><h2>6. Risks and Assumptions</h2></div>
                <div class="risks">
                    <ul>
                        {risks_html}
                    </ul>
                </div>
            </div>
            <div class="prd-section">
                <div class="prd-section-title"><h2>7. Next Steps</h2></div>
                <div class="next-steps">
                    <ul>
                        {next_steps_html}
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True,
        )
        col_edit, col_export = st.columns([2, 1])
        with col_edit:
            edit_modal = Modal(key="edit_modal", title="Edit Experiment Plan")
            if st.button("✏️ Edit Plan"):
                st.session_state.edit_modal_open = True
                st.session_state.temp_plan_edit = plan.copy()
                edit_modal.open()
        if st.session_state.edit_modal_open:
            with edit_modal.container():
                st.header("Edit Plan")
                edited_plan = st.session_state.temp_plan_edit
                st.subheader("1. Problem Statement")
                edited_plan['problem_statement'] = st.text_area("Problem Statement", value=edited_plan.get('problem_statement', ''), height=100, key="edit_prob_stmt")
                st.markdown("---")
                st.subheader("2. Hypotheses")
                if 'hypotheses' not in edited_plan: edited_plan['hypotheses'] = []
                for i, h in enumerate(edited_plan.get("hypotheses", [])):
                    with st.expander(f"Hypothesis {i+1}", expanded=True):
                        edited_plan['hypotheses'][i]['hypothesis'] = st.text_input("Hypothesis", value=h.get('hypothesis', ''), key=f"edit_hyp_{i}")
                        edited_plan['hypotheses'][i]['rationale'] = st.text_area("Rationale", value=h.get('rationale', ''), key=f"edit_rat_{i}", height=50)
                        edited_plan['hypotheses'][i]['behavioral_basis'] = st.text_input("Behavioral Basis", value=h.get('behavioral_basis', ''), key=f"edit_behav_{i}")
                        edited_plan['hypotheses'][i]['example_implementation'] = st.text_area("Implementation Example", value=h.get('example_implementation', ''), key=f"edit_impl_{i}", height=50)
                        if st.button(f"Delete Hypothesis {i+1}", key=f"del_hyp_{i}"):
                            edited_plan['hypotheses'].pop(i)
                            st.experimental_rerun()
                if st.button("Add New Hypothesis", key="add_hyp"):
                    edited_plan['hypotheses'].append({"hypothesis": "New Hypothesis", "rationale": "", "example_implementation": "", "behavioral_basis": ""})
                    st.experimental_rerun()
                st.markdown("---")
                st.subheader("3. Risks and Assumptions")
                if 'risks_and_assumptions' not in edited_plan: edited_plan['risks_and_assumptions'] = []
                for i, r in enumerate(edited_plan.get("risks_and_assumptions", [])):
                    with st.expander(f"Risk {i+1}", expanded=True):
                        edited_plan['risks_and_assumptions'][i]['risk'] = st.text_input("Risk", value=r.get('risk', ''), key=f"edit_risk_{i}")
                        edited_plan['risks_and_assumptions'][i]['severity'] = st.selectbox("Severity", options=["High", "Medium", "Low"], index=["High", "Medium", "Low"].index(r.get('severity', 'Medium')), key=f"edit_sev_{i}")
                        edited_plan['risks_and_assumptions'][i]['mitigation'] = st.text_area("Mitigation", value=r.get('mitigation', ''), height=50, key=f"edit_mit_{i}")
                        if st.button(f"Delete Risk {i+1}", key=f"del_risk_{i}"):
                            edited_plan['risks_and_assumptions'].pop(i)
                            st.experimental_rerun()
                if st.button("Add New Risk", key="add_risk"):
                    edited_plan['risks_and_assumptions'].append({"risk": "", "severity": "Medium", "mitigation": ""})
                    st.experimental_rerun()
                st.markdown("---")
                st.subheader("4. Next Steps")
                if 'next_steps' not in edited_plan: edited_plan['next_steps'] = []
                for i, step in enumerate(edited_plan.get("next_steps", [])):
                    col_s1, col_s2 = st.columns([5,1])
                    with col_s1:
                        edited_plan['next_steps'][i] = st.text_input("Next Step", value=step, key=f"edit_step_{i}")
                    with col_s2:
                        if st.button("Delete", key=f"del_step_{i}"):
                            edited_plan['next_steps'].pop(i)
                            st.experimental_rerun()
                if st.button("Add New Next Step", key="add_step"):
                    edited_plan['next_steps'].append("")
                    st.experimental_rerun()
                st.markdown("---")
                if st.button("Save Changes and Close"):
                    st.session_state.ai_parsed = edited_plan
                    st.session_state.edit_modal_open = False
                    st.experimental_rerun()
        
        with col_export:
            col_exp1, col_exp2 = st.columns([1,1])
            with col_exp1:
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(plan, indent=2, ensure_ascii=False),
                    file_name="experiment_plan.json",
                    mime="application/json",
                )
            if REPORTLAB_AVAILABLE:
                pdf_bytes = generate_pdf_bytes_from_prd_dict(plan, title=plan.get("name", "Experiment PRD"))
                if pdf_bytes:
                    with col_exp2:
                        st.download_button(
                            label="Download PDF",
                            data=pdf_bytes,
                            file_name="experiment_plan.pdf",
                            mime="application/pdf",
                        )
        st.markdown(f"<div class='prd-footer'>Generated by A/B Test Architect on """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</div></div>""", unsafe_allow_html=True)
