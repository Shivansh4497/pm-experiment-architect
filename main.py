# main.py — Final Certified Version (A/B Test Architect)
import streamlit as st
import json
import re
import os
import pandas as pd
from typing import Any, Dict, Optional, Tuple, List, Union
from pydantic import BaseModel, ValidationError
from prompt_engine import generate_experiment_plan, generate_hypothesis_details
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize
import numpy as np
import hashlib
from datetime import datetime
from io import BytesIO
import ast
import html

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

# --- Pydantic Models for Validation ---
class Hypothesis(BaseModel):
    hypothesis: str
    rationale: str
    example_implementation: str
    behavioral_basis: str

class Variant(BaseModel):
    control: str
    variation: str

class Metric(BaseModel):
    name: str
    formula: str
    importance: str

class Risk(BaseModel):
    risk: str
    severity: str
    mitigation: str

class SuccessCriteria(BaseModel):
    confidence_level: float
    MDE: float
    benchmark: str
    monitoring: str

class ExperimentPlan(BaseModel):
    problem_statement: str
    hypotheses: List[Hypothesis]
    variants: List[Variant]
    metrics: List[Metric]
    success_criteria: SuccessCriteria
    risks_and_assumptions: List[Risk]
    next_steps: List[str]
    statistical_rationale: str

# --- Improved Helper Functions ---
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
    text = text.replace("\r"," ").replace("\t", " ")
    text = re.sub(r"[ \f\v]+", " ", text)
    return text.strip()

def html_sanitize(text: Any) -> str:
    if text is None: 
        return ""
    text = str(text)
    # Only escape dangerous characters, preserve HTML structure
    return html.escape(text)

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

def validate_llm_output(raw_output: Union[str, dict]) -> Optional[Dict]:
    """Validate LLM output against our schema using Pydantic"""
    if isinstance(raw_output, str):
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            st.error("LLM returned invalid JSON. Please try again.")
            return None
    else:
        parsed = raw_output

    try:
        validated = ExperimentPlan(**parsed)
        return validated.dict()
    except ValidationError as e:
        st.error(f"LLM output validation failed: {str(e)}")
        return None

def _safe_single_to_double_quotes(s: str) -> str:
    s = re.sub(r"(?<=[:\{\[,]\s*)'([^']*?)'(?=\s*[,}\]])", r'"\1"', s)
    s = re.sub(r"'([A-Za-z0-9_ \-]+?)'\s*:", r'"\1":', s)
    return s

def extract_json(text: Any) -> Optional[Dict]:
    """Improved JSON extraction with Pydantic validation"""
    if text is None:
        st.error("No output returned from LLM.")
        return None
    if isinstance(text, dict):
        return validate_llm_output(text)
    if isinstance(text, list):
        if all(isinstance(i, dict) for i in text):
            return validate_llm_output({"items": text})
        st.error("LLM returned a JSON list when an object was expected.")
        return None
    
    try:
        raw = str(text)
    except Exception as e:
        st.error(f"Unexpected LLM output type: {e}")
        return None
    
    # Try direct JSON parse first
    try:
        parsed = json.loads(raw)
        return validate_llm_output(parsed)
    except json.JSONDecodeError:
        pass
    
    # Try extracting JSON from markdown or other wrappers
    candidate = re.search(r"```(?:json)?\n([\s\S]+?)\n```|{[\s\S]+?}", raw)
    if candidate:
        try:
            clean_candidate = candidate.group(1) if candidate.group(1) else candidate.group(0)
            clean_candidate = re.sub(r',\s*,', ',', clean_candidate)
            clean_candidate = re.sub(r',\s*\}', '}', clean_candidate)
            clean_candidate = re.sub(r',\s*\]', ']', clean_candidate)
            parsed = json.loads(clean_candidate)
            return validate_llm_output(parsed)
        except Exception:
            pass
    
    # Final fallback with single quote handling
    try:
        converted = _safe_single_to_double_quotes(raw)
        parsed = json.loads(converted)
        return validate_llm_output(parsed)
    except Exception:
        st.error("LLM output could not be parsed as valid JSON.")
        st.code(raw[:2000] + ("..." if len(raw) > 2000 else ""))
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
        # Guard against invalid inputs
        if baseline is None or mde is None:
            return None, None
            
        if metric_type == "Conversion Rate" and baseline == 0:
            st.error("Baseline cannot be zero for conversion rates.")
            return None, None
            
        if metric_type == "Numeric Value" and (std_dev is None or std_dev <= 0):
            st.error("Standard deviation must be positive for numeric metrics.")
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
            sample_size_per_variant = analysis.solve_power(
                effect_size=effect_size,
                alpha=alpha,
                power=power,
                alternative="two-sided"
            )
        elif metric_type == "Numeric Value":
            mde_absolute = float(baseline) * mde_relative
            effect_size = mde_absolute / float(std_dev)
            if effect_size == 0:
                return None, None
            analysis = TTestIndPower()
            sample_size_per_variant = analysis.solve_power(
                effect_size=effect_size,
                alpha=alpha,
                power=power,
                alternative="two-sided"
            )
        else:
            return None, None
            
        if sample_size_per_variant is None or sample_size_per_variant <= 0 or not np.isfinite(sample_size_per_variant):
            return None, None
            
        total = sample_size_per_variant * num_variants
        return int(np.ceil(sample_size_per_variant)), int(np.ceil(total))
    except Exception as e:
        st.error(f"Sample size calculation error: {str(e)}")
        return None, None

def generate_pdf_bytes_from_prd_dict(prd: Dict, title: str = "Experiment PRD") -> Optional[bytes]:
    if not REPORTLAB_AVAILABLE:
        return None
        
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="PRDTitle",
        fontSize=20,
        leading=24,
        spaceAfter=12,
        alignment=1
    ))
    styles.add(ParagraphStyle(
        name="SectionHeading",
        fontSize=13,
        leading=16,
        spaceBefore=12,
        spaceAfter=6,
        fontName="Helvetica-Bold"
    ))
    styles.add(ParagraphStyle(
        name="BodyTextCustom",
        fontSize=10.5,
        leading=14
    ))
    styles.add(ParagraphStyle(
        name="BulletText",
        fontSize=10.5,
        leading=14,
        leftIndent=12,
        bulletIndent=6
    ))
    
    story: List[Any] = []
    
    def pdf_sanitize(text: Any) -> str:
        if text is None: 
            return ""
        text = str(text)
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        
    def add_section_header(title: str):
        story.append(Spacer(1, 12))
        story.append(Paragraph(title, styles["SectionHeading"]))
        story.append(Spacer(1, 6))
        
    story.append(Paragraph(title, styles["PRDTitle"]))

    # Problem Statement Section
    add_section_header("1. Problem Statement")
    story.append(Paragraph(pdf_sanitize(prd.get("problem_statement", "")), styles["BodyTextCustom"]))
    
    # Hypotheses Section
    add_section_header("2. Hypotheses")
    for idx, h in enumerate(prd.get("hypotheses", [])):
        if not isinstance(h, dict): continue
        story.append(Paragraph(
            f"<b>Hypothesis {idx + 1}:</b> {pdf_sanitize(h.get('hypothesis', ''))}", 
            styles["BodyTextCustom"]
        ))
        story.append(Paragraph(
            f"<b>Rationale:</b> {pdf_sanitize(h.get('rationale', ''))}", 
            styles["BodyTextCustom"]
        ))
        story.append(Paragraph(
            f"<b>Example Implementation:</b> {pdf_sanitize(h.get('example_implementation', ''))}", 
            styles["BodyTextCustom"]
        ))
        story.append(Paragraph(
            f"<b>Behavioral Basis:</b> {pdf_sanitize(h.get('behavioral_basis', ''))}", 
            styles["BodyTextCustom"]
        ))
        story.append(Spacer(1, 10))
    
    # Variants Section
    add_section_header("3. Variants")
    for v in prd.get("variants", []):
        if not isinstance(v, dict): continue
        story.append(Paragraph(
            f"<b>Control:</b> {pdf_sanitize(v.get('control', ''))}", 
            styles["BodyTextCustom"]
        ))
        story.append(Paragraph(
            f"<b>Variation:</b> {pdf_sanitize(v.get('variation', ''))}", 
            styles["BodyTextCustom"]
        ))
        story.append(Spacer(1, 10))
    
    # Metrics Section
    add_section_header("4. Metrics")
    metrics_data = [['Name', 'Formula', 'Importance']]
    for m in prd.get("metrics", []):
        if not isinstance(m, dict): continue
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

        # Success Criteria Section
    add_section_header("5. Success Criteria & Statistical Rationale")
    criteria = prd.get("success_criteria", {})
    story.append(Paragraph(
        f"<b>Confidence Level:</b> {pdf_sanitize(criteria.get('confidence_level', ''))}%", 
        styles["BodyTextCustom"]
    ))
    story.append(Paragraph(
        f"<b>Minimum Detectable Effect (MDE):</b> {pdf_sanitize(criteria.get('MDE', ''))}%", 
        styles["BodyTextCustom"]
    ))
    story.append(Paragraph(
        f"<b>Statistical Rationale:</b> {pdf_sanitize(prd.get('statistical_rationale', ''))}", 
        styles["BodyTextCustom"]
    ))
    story.append(Paragraph(
        f"<b>Benchmark:</b> {pdf_sanitize(criteria.get('benchmark', ''))}", 
        styles["BodyTextCustom"]
    ))
    story.append(Paragraph(
        f"<b>Monitoring:</b> {pdf_sanitize(criteria.get('monitoring', ''))}", 
        styles["BodyTextCustom"]
    ))
    
    # Add calculator values if available
    sample_size_per_variant = st.session_state.get('calculated_sample_size_per_variant')
    if sample_size_per_variant:
        story.append(Paragraph(
            f"<b>Sample Size per Variant:</b> {sample_size_per_variant:,}", 
            styles["BodyTextCustom"]
        ))
    total_sample_size = st.session_state.get('calculated_total_sample_size')
    if total_sample_size:
        story.append(Paragraph(
            f"<b>Total Sample Size:</b> {total_sample_size:,}", 
            styles["BodyTextCustom"]
        ))
    duration_days = st.session_state.get('calculated_duration_days')
    if duration_days:
        story.append(Paragraph(
            f"<b>Estimated Duration:</b> {round(duration_days, 1)} days", 
            styles["BodyTextCustom"]
        ))
        
    # Risks Section
    add_section_header("6. Risks and Assumptions")
    risks_data = [['Risk', 'Severity', 'Mitigation']]
    for r in prd.get("risks_and_assumptions", []):
        if not isinstance(r, dict): continue
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
        
    # Next Steps Section
    add_section_header("7. Next Steps")
    next_steps_data = [['Action']]
    for step in prd.get("next_steps", []):
        if not isinstance(step, str): continue
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
st.set_page_config(
    page_title="A/B Test Architect", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Improved CSS with mobile responsiveness
st.markdown(
    """
<style>
.blue-section {background-color: #f6f9ff; padding: 14px; border-radius: 10px; margin-bottom: 14px;}
.green-section {background-color: #f7fff7; padding: 14px; border-radius: 10px; margin-bottom: 14px;}
.section-title {font-size: 1.15rem; font-weight: 700; color: #0b63c6; margin-bottom: 6px;}
.small-muted { color: #7a7a7a; font-size: 13px; }
.prd-card {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    width: 100%;
    max-width: 100%;
    background: #ffffff;
    border-radius: 16px;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    padding: 1.5rem;
    border: 1px solid #e5e7eb;
    margin: 0 auto;
}
.prd-header {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #e5e7eb;
}
.logo-wrapper {
    background: #0b63c6;
    color: white;
    padding: 1rem 1.5rem;
    border-radius: 12px;
    font-weight: 800;
    font-size: 2rem;
    line-height: 1;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    margin-bottom: 1rem;
    transform: rotate(-3deg);
}
.header-text h1 {
    margin: 0;
    font-size: 1.75rem;
    font-weight: 900;
    color: #052a4a;
    text-align: center;
}
.header-text p {
    margin: 0.25rem 0 0;
    font-size: 1rem;
    color: #4b5563;
    text-align: center;
}
.prd-section {
    margin-bottom: 1.5rem;
}
.prd-section-title {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
    color: #0b63c6;
}
.prd-section-title h2 {
    margin: 0;
    font-size: 1.25rem;
    font-weight: 700;
}
.prd-section-content {
    background: #f3f8ff;
    border-left: 4px solid #0b63c6;
    padding: 1rem;
    border-radius: 8px;
    line-height: 1.6;
    color: #1f2937;
    margin-bottom: 1rem;
    overflow-wrap: break-word;
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
    padding: 0.75rem;
    background: #fdfefe;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    border: 1px solid #e5e7eb;
    line-height: 1.5;
    margin-bottom: 0.75rem;
    overflow-wrap: break-word;
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
    margin-bottom: 0.25rem;
    color: #052a4a;
}
.hypothesis-title {
    font-size: 1rem;
    font-weight: 600;
    color: #052a4a;
}
.formula-code {
    background-color: #eef2ff;
    padding: 2px 4px;
    border-radius: 4px;
    font-family: 'Courier New', Courier, monospace;
    font-size: 0.85em;
    color: #3b5998;
}
.importance {
    font-weight: 600;
    color: #0b63c6;
}
.severity {
    font-weight: 600;
    padding: 2px 6px;
    border-radius: 4px;
}
.severity.high { 
    color: #ef4444;
    background-color: #fee2e2;
}
.severity.medium { 
    color: #f97316;
    background-color: #ffedd5;
}
.severity.low { 
    color: #22c55e;
    background-color: #dcfce7;
}
.prd-footer {
    margin-top: 1.5rem;
    padding-top: 1rem;
    border-top: 1px solid #e5e7eb;
    text-align: center;
    font-size: 0.8rem;
    color: #6b7280;
}

@media (min-width: 768px) {
    .prd-card {
        padding: 2.5rem;
        max-width: 900px;
    }
    .prd-header {
        flex-direction: row;
        align-items: center;
    }
    .logo-wrapper {
        margin-right: 1rem;
        margin-bottom: 0;
    }
    .header-text h1 {
        font-size: 2.25rem;
        text-align: left;
    }
    .header-text p {
        text-align: left;
    }
}
.section-list-item {
    overflow-wrap: break-word;
    word-break: break-word;
    hyphens: auto;
}
.section-list-item p {
    white-space: normal;
    margin-bottom: 0.5rem;
}
.section-list-item p:last-child {
    margin-bottom: 0;
}
</style>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800&display=swap" rel="stylesheet">
""",
    unsafe_allow_html=True,
)

# --- Session State Initialization ---
def init_session_state():
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
    if "temp_plan_edit" not in st.session_state:
        st.session_state.temp_plan_edit = {}
    if "ai_parsed" not in st.session_state:
        st.session_state.ai_parsed = None
    if "hypotheses_from_llm" not in st.session_state:
        st.session_state.hypotheses_from_llm = []
    if "calc_locked" not in st.session_state:
        st.session_state.calc_locked = False
    if "locked_stats" not in st.session_state:
        st.session_state.locked_stats = {}

init_session_state()

# --- Input Sections ---
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

    mde_default = max(mde_default, 0.1)

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
        st.session_state.temp_plan_edit = {}
        st.session_state.hypotheses_from_llm = []
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
                    st.session_state.hypotheses_from_llm = parsed.get("hypotheses", [])
                    st.success("Plan generated successfully — let's refine it step-by-step.")
                else:
                    st.error("Plan generation failed. Please check inputs and try again.")
                    st.session_state.stage = "input"
            except Exception as e:
                st.error(f"LLM generation failed: {str(e)}")
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
                st.rerun()

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
                    st.session_state.temp_plan_edit = st.session_state.ai_parsed.copy()
                    st.rerun()
        
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
                            hyp_details_raw = generate_hypothesis_details(new_hyp_text, context)
                            hyp_details_parsed = extract_json(hyp_details_raw)
                            if hyp_details_parsed:
                                st.session_state.ai_parsed['hypotheses'].append(hyp_details_parsed)
                                st.session_state.stage = "full_plan"
                                st.session_state.temp_plan_edit = st.session_state.ai_parsed.copy()
                                st.rerun()
                            else:
                                st.error("Failed to generate details for your hypothesis. Please try again.")
                        except Exception as e:
                            st.error(f"LLM call failed: {str(e)}")

    elif st.session_state.stage == "full_plan":
        st.subheader("Step 3: Refine the Full Plan")
        st.info("Your experiment plan is ready! Now you can edit any of the sections, starting with the A/B test calculator.")

        # --- A/B Test Calculator Section ---
        with st.expander("🔢 A/B Test Calculator: Fine-tune sample size", expanded=True):
            plan = st.session_state.ai_parsed
            
            if 'success_criteria' not in plan or not isinstance(plan['success_criteria'], dict):
                plan['success_criteria'] = {}
            calc_mde_initial = plan['success_criteria'].get('MDE', mde_default)
            
            calc_mde = st.session_state.get("calc_mde", calc_mde_initial)
            calc_conf = st.session_state.get("calc_confidence", plan['success_criteria'].get("confidence_level", 95))
            calc_power = st.session_state.get("calc_power", 80)
            calc_variants = st.session_state.get("calc_variants", 2)
            
            col1, col2 = st.columns(2)
            with col1:
                calc_mde = st.number_input("Minimum Detectable Effect (MDE) %", min_value=0.1, max_value=50.0, value=float(max(0.1, float(calc_mde))), step=0.1, key="calc_mde_key")
                calc_conf = st.number_input("Confidence Level (%)", 
                                            min_value=80, 
                                            max_value=99, 
                                            value=int(max(80, int(calc_conf))),
                                            step=1, 
                                            key="calc_conf_key")
            with col2:
                calc_power = st.number_input("Statistical Power (%)", min_value=70, max_value=95, value=int(calc_power), step=1, key="calc_power_key")
                calc_variants = st.number_input("Number of Variants (Control + Variations)", min_value=2, max_value=5, value=int(calc_variants), step=1, key="calc_variants_key")
            
            if metric_type == "Numeric Value" and std_dev is not None:
                st.info(f"Standard Deviation pre-filled: {std_dev}")

            col_act1, col_act2 = st.columns([1, 1])
            with col_act1:
                refresh_btn = st.button("Calculate", key="calc_btn")
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
                st.session_state.calculated_duration_days = (users_to_test / dau) if dau > 0 and users_to_test else "N/A"

            if st.session_state.get("calculated_sample_size_per_variant"):
                st.markdown("---")
                st.markdown(f"**Sample Size per Variant:** {st.session_state.calculated_sample_size_per_variant:,}")
                st.markdown(f"**Total Sample Size:** {st.session_state.calculated_total_sample_size:,}")
                st.markdown(f"**Estimated Duration:** **{round(st.session_state.calculated_duration_days, 1)}** days")
                
            if lock_btn:
                st.session_state.calc_locked = True
                
                if 'success_criteria' not in st.session_state.ai_parsed:
                    st.session_state.ai_parsed['success_criteria'] = {}
                
                st.session_state.ai_parsed['success_criteria']['MDE'] = calc_mde
                st.session_state.ai_parsed['success_criteria']['confidence_level'] = calc_conf
                st.success("Calculator values locked into the plan!")

        # --- Final Plan Preview (Read-Only) ---
        plan = st.session_state.ai_parsed
        
        # Build HTML for Hypotheses
        hypotheses_html = ""
        for h in plan.get("hypotheses", []):
            if not isinstance(h, dict): continue
            hypotheses_html += f"""
                <div class='section-list-item'>
                    <p class='hypothesis-title'>{html_sanitize(h.get('hypothesis', ''))}</p>
                    <p class='rationale'><strong>Rationale:</strong> {html_sanitize(h.get('rationale', ''))}</p>
                    <p class='example'><strong>Example:</strong> {html_sanitize(h.get('example_implementation', ''))}</p>
                    <p class='behavioral-basis'><strong>Behavioral Basis:</strong> {html_sanitize(h.get('behavioral_basis', ''))}</p>
                </div>
            """

        # Build HTML for Variants
        variants_html = ""
        for i, v in enumerate(plan.get("variants", [])):
            if not isinstance(v, dict): continue
            variants_html += f"""
                <div class='section-list-item'>
                    <p><strong>Control {i+1}:</strong> {html_sanitize(v.get('control', 'N/A'))}</p>
                    <p><strong>Variation {i+1}:</strong> {html_sanitize(v.get('variation', 'N/A'))}</p>
                </div>
            """

        # Build HTML for Metrics
        metrics_html = ""
        for m in plan.get("metrics", []):
            if not isinstance(m, dict): continue
            metrics_html += f"""
                <div class='section-list-item'>
                    <p><strong>Name:</strong> {html_sanitize(m.get('name', ''))}</p>
                    <p><strong>Formula:</strong> <code class='formula-code'>{html_sanitize(m.get('formula', ''))}</code></p>
                    <p><strong>Importance:</strong> <span class='importance'>{html_sanitize(m.get('importance', ''))}</span></p>
                </div>
            """
        
        # Build HTML for Success Criteria
        criteria = plan.get('success_criteria', {})
        
        sample_size_per_variant = st.session_state.get('calculated_sample_size_per_variant')
        total_sample_size = st.session_state.get('calculated_total_sample_size')
        duration_days = st.session_state.get('calculated_duration_days')

        stats_html_parts = []
        stats_html_parts.append(f"""
            <div class='section-list-item'>
                <p><strong>Confidence:</strong> {criteria.get('confidence_level', '')}%</p>
                <p><strong>MDE:</strong> {criteria.get('MDE', '')}%</p>
                <p><strong>Statistical Rationale:</strong> {html_sanitize(plan.get('statistical_rationale', 'No rationale provided.'))}</p>
            """)

        if sample_size_per_variant is not None:
            stats_html_parts.append(f"<p><strong>Sample Size per Variant:</strong> {sample_size_per_variant:,}</p>")
        if total_sample_size is not None:
            stats_html_parts.append(f"<p><strong>Total Sample Size:</strong> {total_sample_size:,}</p>")
        if duration_days is not None:
            stats_html_parts.append(f"<p><strong>Estimated Duration:</strong> {round(duration_days, 1)} days</p>")
        
        stats_html_parts.append("</div>")
        stats_html = "".join(stats_html_parts)

        # Build HTML for Risks
        risks_html = ""
        for r in plan.get("risks_and_assumptions", []):
            if not isinstance(r, dict): continue
            
            risk_text = r.get('risk', 'N/A')
            severity_text = r.get('severity', 'Medium')
            mitigation_text = r.get('mitigation', 'N/A')

            valid_severities = ['high', 'medium', 'low']
            severity_class = severity_text.lower() if severity_text and severity_text.lower() in valid_severities else 'medium'
            
            risks_html = ""
            for r in plan.get("risks_and_assumptions", []):
                if not isinstance(r, dict): continue
    
                risk_text = r.get('risk', 'N/A')
                severity_text = r.get('severity', 'Medium')
                mitigation_text = r.get('mitigation', 'N/A')

                # Validate severity class
                severity_class = "medium"
                if severity_text.lower() in ['high', 'medium', 'low']:
                    severity_class = severity_text.lower()
    
                risks_html += f"""
                    <div class='section-list-item'>
                        <p><strong>Risk:</strong> {html_sanitize(risk_text)}</p>
                        <p><strong>Severity:</strong> <span class='severity {severity_class}'>{html_sanitize(severity_text)}</span></p>
                        <p><strong>Mitigation:</strong> {html_sanitize(mitigation_text)}</p>
                    </div>
                """
        
        # Build HTML for Next Steps
        next_steps_html = ""
        for step in plan.get("next_steps", []):
            if not isinstance(step, str): continue
            next_steps_html += f"""
                <div class='section-list-item'>
                    <p>{html_sanitize(step)}</p>
                </div>
            """
        plan_html = f"""
            <div class='prd-card'>
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
                    <div class="prd-section-content">
                        <div class="section-list">{hypotheses_html}</div>
                    </div>
                </div>
                <div class="prd-section">
                    <div class="prd-section-title"><h2>3. Variants</h2></div>
                    <div class="prd-section-content">
                        <div class="section-list">{variants_html}</div>
                    </div>
                </div>
                <div class="prd-section">
                    <div class="prd-section-title"><h2>4. Metrics</h2></div>
                    <div class="prd-section-content">
                        <div class="section-list">{metrics_html}</div>
                    </div>
                </div>
                <div class="prd-section">
                    <div class="prd-section-title"><h2>5. Success Criteria & Statistical Rationale</h2></div>
                    <div class="prd-section-content">
                        <div class="section-list">{stats_html}</div>
                    </div>
                </div>
                <div class="prd-section">
                    <div class="prd-section-title"><h2>6. Risks and Assumptions</h2></div>
                    <div class="prd-section-content">
                        <div class="section-list">{risks_html}</div>
                    </div>
                </div>
                <div class="prd-section">
                    <div class="prd-section-title"><h2>7. Next Steps</h2></div>
                    <div class="prd-section-content">
                        <div class="section-list">{next_steps_html}</div>
                    </div>
                </div>
                <div class='prd-footer'>Generated by A/B Test Architect on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
            </div>
            """
        st.markdown(plan_html, unsafe_allow_html=True)
        
        with st.expander("✏️ Edit Experiment Plan", expanded=False):
            st.header("Edit Plan")

            # Initialize a copy of the plan for editing
            edited_plan = st.session_state.ai_parsed.copy()
            
            with st.form(key='edit_form'):
                st.subheader("1. Problem Statement")
                edited_plan['problem_statement'] = st.text_area("Problem Statement", value=edited_plan.get('problem_statement', ''), height=100, key="edit_problem_statement")
                st.markdown("---")
                
                st.subheader("2. Hypotheses")
                if 'hypotheses' not in edited_plan or not isinstance(edited_plan['hypotheses'], list): 
                    edited_plan['hypotheses'] = []
                
                num_hypotheses = st.number_input("Number of Hypotheses", min_value=0, value=len(edited_plan['hypotheses']), key='num_hyp')
                
                # Dynamic resizing of the hypotheses list
                if num_hypotheses > len(edited_plan['hypotheses']):
                    edited_plan['hypotheses'].extend([{"hypothesis": "", "rationale": "", "example_implementation": "", "behavioral_basis": ""}] * (num_hypotheses - len(edited_plan['hypotheses'])))
                elif num_hypotheses < len(edited_plan['hypotheses']):
                    edited_plan['hypotheses'] = edited_plan['hypotheses'][:num_hypotheses]
                    
                for i, h in enumerate(edited_plan.get("hypotheses", [])):
                    if not isinstance(h, dict): continue
                    with st.expander(f"Hypothesis {i+1}", expanded=True):
                        edited_plan['hypotheses'][i]['hypothesis'] = st.text_input("Hypothesis", value=h.get('hypothesis', ''), key=f"edit_hyp_{i}")
                        edited_plan['hypotheses'][i]['rationale'] = st.text_area("Rationale", value=h.get('rationale', ''), key=f"edit_rat_{i}", height=50)
                        edited_plan['hypotheses'][i]['behavioral_basis'] = st.text_input("Behavioral Basis", value=h.get('behavioral_basis', ''), key=f"edit_behav_{i}")
                        edited_plan['hypotheses'][i]['example_implementation'] = st.text_area("Implementation Example", value=h.get('example_implementation', ''), key=f"edit_impl_{i}", height=50)
                st.markdown("---")
                
                st.subheader("3. Variants")
                if 'variants' not in edited_plan or not isinstance(edited_plan['variants'], list): 
                    edited_plan['variants'] = []
                num_variants = st.number_input("Number of Variants", min_value=1, value=len(edited_plan['variants']), key='num_variants')

                if num_variants > len(edited_plan['variants']):
                    edited_plan['variants'].extend([{"control": "", "variation": ""}] * (num_variants - len(edited_plan['variants'])))
                elif num_variants < len(edited_plan['variants']):
                    edited_plan['variants'] = edited_plan['variants'][:num_variants]

                for i, v in enumerate(edited_plan.get('variants', [])):
                    if not isinstance(v, dict): continue
                    with st.expander(f"Variant {i+1}", expanded=True):
                        edited_plan['variants'][i]['control'] = st.text_input("Control", value=v.get('control', ''), key=f'edit_control_{i}')
                        edited_plan['variants'][i]['variation'] = st.text_input("Variation", value=v.get('variation', ''), key=f'edit_variation_{i}')
                st.markdown("---")

                st.subheader("4. Metrics")
                if 'metrics' not in edited_plan or not isinstance(edited_plan['metrics'], list): 
                    edited_plan['metrics'] = []
                num_metrics = st.number_input("Number of Metrics", min_value=1, value=len(edited_plan['metrics']), key='num_metrics')
                
                if num_metrics > len(edited_plan['metrics']):
                    edited_plan['metrics'].extend([{"name": "", "formula": "", "importance": "Primary"}] * (num_metrics - len(edited_plan['metrics'])))
                elif num_metrics < len(edited_plan['metrics']):
                    edited_plan['metrics'] = edited_plan['metrics'][:num_metrics]

                for i, m in enumerate(edited_plan.get("metrics", [])):
                    if not isinstance(m, dict): continue
                    with st.expander(f"Metric {i+1}", expanded=True):
                        edited_plan['metrics'][i]['name'] = st.text_input("Name", value=m.get('name', ''), key=f"edit_metric_name_{i}")
                        edited_plan['metrics'][i]['formula'] = st.text_input("Formula", value=m.get('formula', ''), key=f"edit_metric_formula_{i}")
                        
                        options = ["Primary", "Secondary", "Guardrail"]
                        importance_value = m.get('importance', 'Primary')
                        if importance_value not in options:
                            importance_value = "Primary"
                        
                        edited_plan['metrics'][i]['importance'] = st.selectbox("Importance", options=options, index=options.index(importance_value), key=f"edit_metric_imp_{i}")
                        
                st.markdown("---")
                
                st.subheader("5. Success Criteria & Statistical Rationale")
                if 'success_criteria' not in edited_plan or not isinstance(edited_plan['success_criteria'], dict): 
                    edited_plan['success_criteria'] = {}
                edited_plan['success_criteria']['confidence_level'] = st.number_input("Confidence Level (%)", value=edited_plan['success_criteria'].get('confidence_level', 95), key="edit_conf")
                mde_value = max(0.1, float(edited_plan['success_criteria'].get('MDE', 5.0))
                edited_plan['success_criteria']['MDE'] = st.number_input("Minimum Detectable Effect (%)", min_value=0.1, value=edited_plan['success_criteria'].get('MDE', 5.0), key="edit_mde")
                edited_plan['statistical_rationale'] = st.text_area("Statistical Rationale", value=edited_plan.get('statistical_rationale', ''), key="edit_rationale", height=100)
                st.markdown("---")
                
                st.subheader("6. Risks and Assumptions")
                if 'risks_and_assumptions' not in edited_plan or not isinstance(edited_plan['risks_and_assumptions'], list): 
                    edited_plan['risks_and_assumptions'] = []
                num_risks = st.number_input("Number of Risks", min_value=0, value=len(edited_plan['risks_and_assumptions']), key='num_risks')
                
                if num_risks > len(edited_plan['risks_and_assumptions']):
                    edited_plan['risks_and_assumptions'].extend([{"risk": "", "severity": "Medium", "mitigation": ""}] * (num_risks - len(edited_plan['risks_and_assumptions'])))
                elif num_risks < len(edited_plan['risks_and_assumptions']):
                    edited_plan['risks_and_assumptions'] = edited_plan['risks_and_assumptions'][:num_risks]

                for i, r in enumerate(edited_plan.get("risks_and_assumptions", [])):
                    if not isinstance(r, dict): continue
                    with st.expander(f"Risk {i+1}", expanded=True):
                        edited_plan['risks_and_assumptions'][i]['risk'] = st.text_input("Risk", value=r.get('risk', ''), key=f"edit_risk_{i}")
                        edited_plan['risks_and_assumptions'][i]['severity'] = st.selectbox("Severity", options=["High", "Medium", "Low"], index=["High", "Medium", "Low"].index(r.get('severity', 'Medium')), key=f"edit_risk_sev_{i}")
                        edited_plan['risks_and_assumptions'][i]['mitigation'] = st.text_area("Mitigation", value=r.get('mitigation', ''), key=f"edit_risk_mit_{i}", height=50)
                st.markdown("---")
                
                st.subheader("7. Next Steps")
                if 'next_steps' not in edited_plan or not isinstance(edited_plan['next_steps'], list): 
                    edited_plan['next_steps'] = []
                next_steps_text = "\n".join(edited_plan.get('next_steps', []))
                new_next_steps = st.text_area("Next Steps (one per line)", value=next_steps_text, height=150, key="edit_next_steps")
                edited_plan['next_steps'] = [step.strip() for step in new_next_steps.split('\n') if step.strip()]
                
                st.markdown("---")
                
                submitted = st.form_submit_button("Save Changes")
                if submitted:
                    st.session_state.ai_parsed = edited_plan
                    st.success("Plan updated successfully!")
                
            st.markdown("<hr>", unsafe_allow_html=True)
            col_export_final = st.columns([1])
            with col_export_final[0]:
                if REPORTLAB_AVAILABLE:
                    pdf_bytes = generate_pdf_bytes_from_prd_dict(plan, title=f"Experiment PRD: {sanitized_metric_name}")
                    if pdf_bytes:
                        st.download_button(
                            label="⬇️ Export to PDF",
                            data=pdf_bytes,
                            file_name=f"experiment_prd_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf"
                        )
                else:
                    st.warning("PDF export is not available. Please install reportlab (`pip install reportlab`).")
                    if st.button("⬇️ Export to JSON"):
                        st.download_button(
                            label="Download JSON",
                            data=json.dumps(plan, indent=2),
                            file_name=f"experiment_prd_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                            mime="application/json"
                        )
