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
# Removed: from streamlit_modal import Modal

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
                    st.code(candidate_clean[:2000] + ("..." if len(candidate_clean) > 2000 else ""))
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
        if not isinstance(h, dict): continue # Defensive check
        story.append(Paragraph(f"<b>Hypothesis {idx + 1}:</b> {pdf_sanitize(h.get('hypothesis', ''))}", styles["BodyTextCustom"]))
        story.append(Paragraph(f"<b>Rationale:</b> {pdf_sanitize(h.get('rationale', ''))}", styles["BodyTextCustom"]))
        story.append(Paragraph(f"<b>Example Implementation:</b> {pdf_sanitize(h.get('example_implementation', ''))}", styles["BodyTextCustom"]))
        story.append(Paragraph(f"<b>Behavioral Basis:</b> {pdf_sanitize(h.get('behavioral_basis', ''))}", styles["BodyTextCustom"]))
        story.append(Spacer(1, 10))
    
    add_section_header("3. Variants")
    for v in prd.get("variants", []):
        if not isinstance(v, dict): continue # Defensive check
        story.append(Paragraph(f"<b>Control:</b> {pdf_sanitize(v.get('control', ''))}", styles["BodyTextCustom"]))
        story.append(Paragraph(f"<b>Variation:</b> {pdf_sanitize(v.get('variation', ''))}", styles["BodyTextCustom"]))
        story.append(Spacer(1, 10))
    
    add_section_header("4. Metrics")
    metrics_data = [['Name', 'Formula', 'Importance']]
    for m in prd.get("metrics", []):
        if not isinstance(m, dict): continue # Defensive check
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
    
    # Add calculator values if available
    sample_size_per_variant = st.session_state.get('calculated_sample_size_per_variant')
    if sample_size_per_variant:
        story.append(Paragraph(f"<b>Sample Size per Variant:</b> {sample_size_per_variant:,}", styles["BodyTextCustom"]))
    total_sample_size = st.session_state.get('calculated_total_sample_size')
    if total_sample_size:
        story.append(Paragraph(f"<b>Total Sample Size:</b> {total_sample_size:,}", styles["BodyTextCustom"]))
    duration_days = st.session_state.get('calculated_duration_days')
    if duration_days:
        story.append(Paragraph(f"<b>Estimated Duration:</b> {round(duration_days, 1)} days", styles["BodyTextCustom"]))
        
    add_section_header("6. Risks and Assumptions")
    risks_data = [['Risk', 'Severity', 'Mitigation']]
    for r in prd.get("risks_and_assumptions", []):
        if not isinstance(r, dict): continue # Defensive check
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
        if not isinstance(step, str): continue # Defensive check
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
    margin-bottom: 1.5rem;
}
.prd-section-content:last-child {
    margin-bottom: 0;
}
.problem-statement {
    font-weight: 500;
    font-style: italic;
    color: #4b5563;
}
.section-list {
    list-style: none;
    padding-left: 0;
    margin: 0;
}
.section-list .list-item {
    padding: 1rem;
    background: #fdfefe;
    border-radius: 8px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    border: 1px solid #e5e7eb;
    position: relative;
    line-height: 1.6;
    margin-bottom: 1rem;
}
.section-list .list-item:last-child {
    margin-bottom: 0;
}
.section-list .list-item p {
    margin: 0;
    color: #4b5563;
}
.section-list .list-item p strong {
    display: block;
    margin-bottom: 0.5rem;
    color: #052a4a;
}
.hypothesis-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #052a4a;
}
.metrics-list .list-item p, .stats-list .list-item p, .risks-list .list-item p, .next-steps-list .list-item p, .variants-list .list-item p {
    margin: 0;
    color: #4b5563; /* Ensure consistent text color */
}
.formula-code {
    background-color: #eef2ff;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'Courier New', Courier, monospace;
    font-size: 0.9em;
    color: #3b5998;
}
.metrics-list .list-item span.importance {
    font-weight: 600;
    color: #0b63c6;
}
.risks-list .list-item span.severity {
    font-weight: 600;
}
.risks-list .list-item span.severity.high { color: #ef4444; }
.risks-list .list-item span.severity.medium { color: #f97316; }
.risks-list .list-item span.severity.low { color: #22c55e; }
.prd-footer {
    margin-top: 2rem;
    padding-top: 1.5rem;
    border-top: 1px solid #e5e7eb;
    text-align: center;
    font-size: 0.875rem;
    color: #6b7280;
}
</style>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800&display=swap" rel="stylesheet">
""",
    unsafe_allow_html=True,
)

# --- Session State Initialization ---
if "edit_modal_open" not in st.session_state:
    st.session_state.edit_modal_open = False
if "stage" not in st.session_state:
    st.session_state.stage = "input"
if "calculated_sample_size_per_variant" not in st.session_state:
    st.session_state.calculated_sample_size_per_variant = None
if "calculated_total_sample_size" not in st.session_state:
    st.session_state.calculated_total_sample_size = None
if "calculated_duration_days" not in st.session_state:
    st.session_state.calculated_duration_days = None

st.title("💡 A/B Test Architect — AI-assisted experiment PRD generator")
st.markdown("Create experiment PRDs, hypotheses, stats, and sample-size guidance — faster and with guardrails.")

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

    # --- FIX: Move this calculation outside the button block so the MDE input always has a value ---
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

    # Ensure the default MDE value is never below the min_value of 0.1 for the number_input widget
    mde_default = max(mde_default, 0.1)

    formatted_current = format_value_with_unit(current_value, metric_unit) if sanitized_metric_name and current_value is not None else ""
    formatted_target = format_value_with_unit(target_value, metric_unit) if sanitized_metric_name and target_value is not None else ""

    goal_with_units = f"I want to improve {sanitized_metric_name} from {formatted_current} to {formatted_target}."

    if not sanitized_metric_name or not strategic_goal or current_value is None or target_value is None or not metric_inputs_valid:
        st.info("💡 **Tips:** Fill in the required fields above to enable plan generation.")
        st.session_state.generate_disabled = True
    else:
        st.session_state.generate_disabled = False

    generate_plan_button = st.button("🚀 Generate Experiment Plan", disabled=st.session_state.generate_disabled, use_container_width=True)

    if generate_plan_button:
        with st.spinner("Calling the A/B Test Architect... this may take up to 30 seconds."):
            llm_context = {
                "strategic_goal": strategic_goal,
                "type": product_type,
                "users": f"{dau:,} DAU",
                "user_persona": user_persona,
                "exact_metric": sanitized_metric_name,
                "metric_type": metric_type,
                "metric_unit": metric_unit,
                "current_value": current_value,
                "target_value": target_value,
                "notes": product_notes,
                "std_dev": std_dev
            }
            llm_output = generate_experiment_plan(goal=goal_with_units, context=llm_context)
            plan_json = extract_json(llm_output)

            if plan_json:
                st.session_state.plan_json = plan_json
                st.session_state.llm_raw_output = llm_output
                st.session_state.stage = "review"
                st.session_state.show_prd_card = True
                st.session_state.edit_modal_open = False
                
# --- Main Logic and UI Sections ---
if st.session_state.stage == "input":
    st.stop()
    
if st.session_state.stage == "review":
    plan = st.session_state.plan_json
    llm_raw_output = st.session_state.llm_raw_output

    st.subheader("📊 Generated Experiment Plan")
    st.markdown("Review the generated plan below. You can make edits directly or regenerate.")

    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("🔄 Regenerate Plan", use_container_width=True):
            st.session_state.stage = "input"
            st.rerun()
    with col2:
        if st.button("📝 Edit Plan", use_container_width=True):
            st.session_state.edit_modal_open = True
            st.session_state.original_json_string = json.dumps(plan, indent=2)

    # Calculate MDE and Sample Size for display
    current_value = st.session_state.get('current_value')
    target_value = st.session_state.get('target_value')
    metric_type = st.session_state.get('metric_type')
    std_dev = st.session_state.get('std_dev')

    try:
        mde_val = plan.get('success_criteria', {}).get('MDE', 5)
        num_variants = len(plan.get('variants', []))
        if num_variants == 0: num_variants = 2 # Default to A/B
        
        calculated_per_variant, calculated_total = calculate_sample_size(
            baseline=current_value,
            mde=mde_val,
            alpha=1 - (plan.get('success_criteria', {}).get('confidence_level', 95) / 100),
            power=0.8,
            num_variants=num_variants,
            metric_type=metric_type,
            std_dev=std_dev
        )
        st.session_state.calculated_sample_size_per_variant = calculated_per_variant
        st.session_state.calculated_total_sample_size = calculated_total
        
        # Calculate duration
        dau = st.session_state.get('dau')
        if calculated_total and dau and dau > 0:
            duration_days = calculated_total / (dau * (num_variants / (num_variants + 1)))
            st.session_state.calculated_duration_days = duration_days
        else:
            st.session_state.calculated_duration_days = None

    except Exception as e:
        st.error(f"Error calculating sample size: {e}")
        st.session_state.calculated_sample_size_per_variant = None
        st.session_state.calculated_total_sample_size = None
        st.session_state.calculated_duration_days = None

    # PRD Card Display
    with st.container():
        st.markdown('<div class="prd-card">', unsafe_allow_html=True)
        # Header
        st.markdown(
            f"""
            <div class="prd-header">
                <div class="logo-wrapper">A/B</div>
                <div class="header-text">
                    <h1>Experiment PRD</h1>
                    <p>{st.session_state.get('sanitized_metric_name', 'Unnamed Metric')} Improvement</p>
                </div>
            </div>
            """, unsafe_allow_html=True
        )

        # 1. Problem Statement
        st.markdown(
            f"""
            <div class="prd-section">
                <div class="prd-section-title">
                    <h2>1. Problem Statement</h2>
                </div>
                <div class="prd-section-content">
                    <p class="problem-statement">{generate_problem_statement(plan, st.session_state.get('current_value'), st.session_state.get('target_value'), st.session_state.get('metric_unit'))}</p>
                </div>
            </div>
            """, unsafe_allow_html=True
        )
        
        # 2. Hypotheses
        st.markdown(
            f"""
            <div class="prd-section">
                <div class="prd-section-title">
                    <h2>2. Hypotheses</h2>
                </div>
                <ul class="section-list">
            """, unsafe_allow_html=True
        )
        for h in plan.get("hypotheses", []):
            st.markdown(f"""
                <li class="list-item">
                    <p><strong>Hypothesis:</strong> <span class="hypothesis-title">{html_sanitize(h.get('hypothesis', ''))}</span></p>
                    <p><strong>Rationale:</strong> {html_sanitize(h.get('rationale', ''))}</p>
                    <p><strong>Example Implementation:</strong> {html_sanitize(h.get('example_implementation', ''))}</p>
                    <p><strong>Behavioral Basis:</strong> {html_sanitize(h.get('behavioral_basis', ''))}</p>
                </li>
            """, unsafe_allow_html=True)
        st.markdown("</ul></div>", unsafe_allow_html=True)
        
        # 3. Variants
        st.markdown(
            f"""
            <div class="prd-section">
                <div class="prd-section-title">
                    <h2>3. Variants</h2>
                </div>
                <ul class="section-list variants-list">
            """, unsafe_allow_html=True
        )
        for v in plan.get("variants", []):
            st.markdown(f"""
                <li class="list-item">
                    <p><strong>Control:</strong> {html_sanitize(v.get('control', ''))}</p>
                    <p><strong>Variation:</strong> {html_sanitize(v.get('variation', ''))}</p>
                </li>
            """, unsafe_allow_html=True)
        st.markdown("</ul></div>", unsafe_allow_html=True)

        # 4. Metrics
        st.markdown(
            f"""
            <div class="prd-section">
                <div class="prd-section-title">
                    <h2>4. Metrics</h2>
                </div>
                <ul class="section-list metrics-list">
            """, unsafe_allow_html=True
        )
        for m in plan.get("metrics", []):
            st.markdown(f"""
                <li class="list-item">
                    <p><strong>Name:</strong> {html_sanitize(m.get('name', ''))}</p>
                    <p><strong>Formula:</strong> <span class="formula-code">{html_sanitize(m.get('formula', ''))}</span></p>
                    <p><strong>Importance:</strong> <span class="importance">{html_sanitize(m.get('importance', ''))}</span></p>
                </li>
            """, unsafe_allow_html=True)
        st.markdown("</ul></div>", unsafe_allow_html=True)

        # 5. Success Criteria & Stats
        stats_content = plan.get("success_criteria", {})
        sample_size_per_variant_display = f"{st.session_state.get('calculated_sample_size_per_variant', 'N/A'):,}" if st.session_state.get('calculated_sample_size_per_variant') else 'N/A'
        total_sample_size_display = f"{st.session_state.get('calculated_total_sample_size', 'N/A'):,}" if st.session_state.get('calculated_total_sample_size') else 'N/A'
        duration_display = f"{round(st.session_state.get('calculated_duration_days', 0), 1)} days" if st.session_state.get('calculated_duration_days') else 'N/A'
        st.markdown(
            f"""
            <div class="prd-section">
                <div class="prd-section-title">
                    <h2>5. Success Criteria & Statistical Rationale</h2>
                </div>
                <div class="prd-section-content">
                    <ul class="section-list stats-list">
                        <li class="list-item">
                            <p><strong>Confidence Level:</strong> {html_sanitize(stats_content.get('confidence_level', ''))}%</p>
                            <p><strong>Minimum Detectable Effect (MDE):</strong> {html_sanitize(stats_content.get('MDE', ''))}%</p>
                            <p><strong>Sample Size per Variant:</strong> {sample_size_per_variant_display}</p>
                            <p><strong>Total Sample Size:</strong> {total_sample_size_display}</p>
                            <p><strong>Estimated Duration:</strong> {duration_display}</p>
                            <p><strong>Statistical Rationale:</strong> {html_sanitize(plan.get('statistical_rationale', ''))}</p>
                            <p><strong>Benchmark:</strong> {html_sanitize(stats_content.get('benchmark', ''))}</p>
                            <p><strong>Monitoring:</strong> {html_sanitize(stats_content.get('monitoring', ''))}</p>
                        </li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True
        )

        # 6. Risks and Assumptions
        st.markdown(
            f"""
            <div class="prd-section">
                <div class="prd-section-title">
                    <h2>6. Risks and Assumptions</h2>
                </div>
                <ul class="section-list risks-list">
            """, unsafe_allow_html=True
        )
        for r in plan.get("risks_and_assumptions", []):
            severity = r.get('severity', '').lower()
            severity_class = 'high' if severity == 'high' else 'medium' if severity == 'medium' else 'low'
            st.markdown(f"""
                <li class="list-item">
                    <p><strong>Risk:</strong> {html_sanitize(r.get('risk', ''))}</p>
                    <p><strong>Severity:</strong> <span class="severity {severity_class}">{html_sanitize(r.get('severity', ''))}</span></p>
                    <p><strong>Mitigation:</strong> {html_sanitize(r.get('mitigation', ''))}</p>
                </li>
            """, unsafe_allow_html=True)
        st.markdown("</ul></div>", unsafe_allow_html=True)
        
        # 7. Next Steps
        st.markdown(
            f"""
            <div class="prd-section">
                <div class="prd-section-title">
                    <h2>7. Next Steps</h2>
                </div>
                <ul class="section-list next-steps-list">
            """, unsafe_allow_html=True
        )
        for step in plan.get("next_steps", []):
            st.markdown(f"""
                <li class="list-item">
                    <p>{html_sanitize(step)}</p>
                </li>
            """, unsafe_allow_html=True)
        st.markdown("</ul></div>", unsafe_allow_html=True)

        st.markdown('<div class="prd-footer">Generated by A/B Test Architect</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # PDF Download Button
    if REPORTLAB_AVAILABLE:
        pdf_bytes = generate_pdf_bytes_from_prd_dict(plan)
        if pdf_bytes:
            filename = f"Experiment_PRD_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            st.download_button(
                label="📄 Download PRD as PDF",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf",
                use_container_width=True
            )
    else:
        st.warning("`reportlab` library not found. PDF export is disabled. Please `pip install reportlab` to enable this feature.")

    # Show raw JSON output for debugging
    with st.expander("Show Raw LLM Output (for debugging)"):
        st.code(llm_raw_output, language="json")

    # --- Modal for editing JSON ---
    # with Modal(key="edit_modal", title="Edit Experiment Plan JSON", max_width=900) as modal:
    #     if st.session_state.edit_modal_open:
    #         if modal.is_open:
    #             edited_json_string = st.text_area(
    #                 "Edit the JSON directly:", 
    #                 value=st.session_state.get('original_json_string', '{}'), 
    #                 height=600
    #             )
    #             col_save, col_cancel = st.columns([1,1])
    #             with col_save:
    #                 if st.button("Apply Changes", use_container_width=True):
    #                     try:
    #                         edited_plan = json.loads(edited_json_string)
    #                         st.session_state.plan_json = edited_plan
    #                         st.success("Changes applied successfully!")
    #                         st.session_state.edit_modal_open = False
    #                         st.rerun()
    #                     except json.JSONDecodeError:
    #                         st.error("Invalid JSON format. Please correct the syntax.")
    #             with col_cancel:
    #                 if st.button("Cancel", use_container_width=True):
    #                     st.session_state.edit_modal_open = False
    #                     st.rerun()

    # The Modal class from streamlit_modal is not available, so this functionality is commented out.
