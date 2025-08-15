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

# PDF Export Setup with better error handling
REPORTLAB_AVAILABLE = False
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER
    REPORTLAB_AVAILABLE = True
except Exception as e:
    REPORTLAB_AVAILABLE = False
    print(f"ReportLab import failed: {e}")

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

# --- Robust Data Sanitizer ---
def sanitize_experiment_plan(raw_plan: Union[Dict[str, Any], None]) -> Dict[str, Any]:
    """Transform raw/malformed experiment plan data into a robust structure."""
    if raw_plan is None:
        raw_plan = {}

    DEFAULT_PLAN = {
        "problem_statement": "",
        "hypotheses": [],
        "variants": [],
        "metrics": [],
        "success_criteria": {
            "confidence_level": 95.0,
            "MDE": 5.0,
            "benchmark": "",
            "monitoring": "",
        },
        "risks_and_assumptions": [],
        "next_steps": [],
        "statistical_rationale": "",
    }

    sanitized_plan = {**DEFAULT_PLAN, **raw_plan}

    # Sanitize hypotheses
    if not isinstance(sanitized_plan["hypotheses"], list):
        sanitized_plan["hypotheses"] = []
    
    for i, hyp in enumerate(sanitized_plan["hypotheses"]):
        if not isinstance(hyp, dict):
            sanitized_plan["hypotheses"][i] = {}
        sanitized_plan["hypotheses"][i] = {
            "hypothesis": str(hyp.get("hypothesis", "")),
            "rationale": str(hyp.get("rationale", "")),
            "example_implementation": str(hyp.get("example_implementation", "")),
            "behavioral_basis": str(hyp.get("behavioral_basis", "")),
        }

    # Sanitize variants
    if not isinstance(sanitized_plan["variants"], list):
        sanitized_plan["variants"] = []
    
    for i, variant in enumerate(sanitized_plan["variants"]):
        if not isinstance(variant, dict):
            sanitized_plan["variants"][i] = {}
        sanitized_plan["variants"][i] = {
            "control": str(variant.get("control", "")),
            "variation": str(variant.get("variation", "")),
        }

    # Sanitize success criteria
    if not isinstance(sanitized_plan["success_criteria"], dict):
        sanitized_plan["success_criteria"] = DEFAULT_PLAN["success_criteria"]
    
    try:
        sanitized_plan["success_criteria"]["confidence_level"] = float(
            sanitized_plan["success_criteria"].get("confidence_level", 95.0)
        )
    except (ValueError, TypeError):
        sanitized_plan["success_criteria"]["confidence_level"] = 95.0

    try:
        sanitized_plan["success_criteria"]["MDE"] = float(
            sanitized_plan["success_criteria"].get("MDE", 5.0)
        )
    except (ValueError, TypeError):
        sanitized_plan["success_criteria"]["MDE"] = 5.0

    # Sanitize metrics
    if not isinstance(sanitized_plan["metrics"], list):
        sanitized_plan["metrics"] = []
    
    for i, metric in enumerate(sanitized_plan["metrics"]):
        if not isinstance(metric, dict):
            sanitized_plan["metrics"][i] = {}
        sanitized_plan["metrics"][i] = {
            "name": str(metric.get("name", "")),
            "formula": str(metric.get("formula", "")),
            "importance": str(metric.get("importance", "Primary")),
        }

    # Sanitize risks
    if not isinstance(sanitized_plan["risks_and_assumptions"], list):
        sanitized_plan["risks_and_assumptions"] = []
    
    for i, risk in enumerate(sanitized_plan["risks_and_assumptions"]):
        if not isinstance(risk, dict):
            sanitized_plan["risks_and_assumptions"][i] = {}
        sanitized_plan["risks_and_assumptions"][i] = {
            "risk": str(risk.get("risk", "")),
            "severity": str(risk.get("severity", "Medium")).title(),
            "mitigation": str(risk.get("mitigation", "")),
        }

    # Sanitize next steps
    if not isinstance(sanitized_plan["next_steps"], list):
        sanitized_plan["next_steps"] = []
    sanitized_plan["next_steps"] = [
        str(step) for step in sanitized_plan["next_steps"] if str(step).strip()
    ]

    return sanitized_plan
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
    return html.escape(text)

def generate_problem_statement(plan: Dict, current: float, target: float, unit: str) -> str:
    plan = sanitize_experiment_plan(plan)
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
        st.warning("LLM output validation failed. Applying sanitizer to recover data.")
        return sanitize_experiment_plan(parsed)

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
        
def pdf_sanitize(text: Any) -> str:
            if text is None: 
                return ""
            text = str(text)
            return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def generate_pdf_bytes_from_prd_dict(prd: Dict, title: str = "Experiment PRD") -> Optional[bytes]:
    """Generate PDF bytes with proper formatting and error handling"""
    if not REPORTLAB_AVAILABLE:
        st.warning("PDF export requires ReportLab which is not available")
        return None

    prd = sanitize_experiment_plan(prd)
    buffer = BytesIO()
    
    try:
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=50,
            leftMargin=50,
            topMargin=50,
            bottomMargin=50
        )
        
        styles = getSampleStyleSheet()
        
        # Safe style additions - only add if they don't exist
        if 'PRDTitle' not in styles:
            styles.add(ParagraphStyle(
                name="PRDTitle",
                fontSize=20,
                leading=24,
                spaceAfter=12,
                alignment=TA_CENTER
            ))
        
        if 'SectionHeading' not in styles:
            styles.add(ParagraphStyle(
                name="SectionHeading",
                fontSize=14,
                leading=18,
                spaceBefore=12,
                spaceAfter=6,
                fontName="Helvetica-Bold"
            ))
        
        if 'BodyText' not in styles:
            styles.add(ParagraphStyle(
                name="BodyText",
                fontSize=11,
                leading=14,
                spaceAfter=6
            ))
        
        story = []
        
        # Title
        story.append(Paragraph(pdf_sanitize(title), styles["PRDTitle"]))
        story.append(Spacer(1, 24))
        
        # Problem Statement
        story.append(Paragraph("1. Problem Statement", styles["SectionHeading"]))
        story.append(Paragraph(pdf_sanitize(prd.get("problem_statement", "")), styles["BodyText"]))
        story.append(Spacer(1, 12))
        
        # Hypotheses
        story.append(Paragraph("2. Hypotheses", styles["SectionHeading"]))
        for idx, h in enumerate(prd.get("hypotheses", [])):
            story.append(Paragraph(f"<b>Hypothesis {idx+1}:</b> {pdf_sanitize(h.get('hypothesis', ''))}", styles["BodyText"]))
            story.append(Paragraph(f"<b>Rationale:</b> {pdf_sanitize(h.get('rationale', ''))}", styles["BodyText"]))
            story.append(Spacer(1, 8))
        
        # Variants
        story.append(Paragraph("3. Variants", styles["SectionHeading"]))
        for v in prd.get("variants", []):
            story.append(Paragraph(f"<b>Control:</b> {pdf_sanitize(v.get('control', ''))}", styles["BodyText"]))
            story.append(Paragraph(f"<b>Variation:</b> {pdf_sanitize(v.get('variation', ''))}", styles["BodyText"]))
            story.append(Spacer(1, 8))
        
        # Metrics Table
        story.append(Paragraph("4. Metrics", styles["SectionHeading"]))
        metrics_data = [['Name', 'Formula', 'Importance']]
        for m in prd.get("metrics", []):
            metrics_data.append([
                pdf_sanitize(m.get('name', '')),
                pdf_sanitize(m.get('formula', '')),
                pdf_sanitize(m.get('importance', ''))
            ])
        
        if len(metrics_data) > 1:
            metrics_table = Table(metrics_data, colWidths=[2*inch, 3*inch, 1.5*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f5f5f5')),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('WORDWRAP', (0,0), (-1,-1), True)
            ]))
            story.append(metrics_table)
        else:
            story.append(Paragraph("No metrics defined.", styles["BodyText"]))
        
        # Success Criteria
        story.append(Paragraph("5. Success Criteria", styles["SectionHeading"]))
        criteria = prd.get("success_criteria", {})
        story.append(Paragraph(f"<b>Confidence Level:</b> {pdf_sanitize(criteria.get('confidence_level', ''))}%", styles["BodyText"]))
        story.append(Paragraph(f"<b>Minimum Detectable Effect (MDE):</b> {pdf_sanitize(criteria.get('MDE', ''))}%", styles["BodyText"]))
        story.append(Paragraph(f"<b>Statistical Rationale:</b> {pdf_sanitize(prd.get('statistical_rationale', ''))}", styles["BodyText"]))
        
        # Add calculator values if available
        if st.session_state.get('calculated_sample_size_per_variant'):
            story.append(Paragraph(f"<b>Sample Size per Variant:</b> {st.session_state.calculated_sample_size_per_variant:,}", styles["BodyText"]))
        if st.session_state.get('calculated_total_sample_size'):
            story.append(Paragraph(f"<b>Total Sample Size:</b> {st.session_state.calculated_total_sample_size:,}", styles["BodyText"]))
        if st.session_state.get('calculated_duration_days'):
            story.append(Paragraph(f"<b>Estimated Duration:</b> {round(st.session_state.calculated_duration_days, 1)} days", styles["BodyText"]))
        
        # Risks Table
        story.append(Paragraph("6. Risks and Assumptions", styles["SectionHeading"]))
        risks_data = [['Risk', 'Severity', 'Mitigation']]
        for r in prd.get("risks_and_assumptions", []):
            risks_data.append([
                pdf_sanitize(r.get('risk', '')),
                pdf_sanitize(r.get('severity', '')),
                pdf_sanitize(r.get('mitigation', ''))
            ])
        
        if len(risks_data) > 1:
            risks_table = Table(risks_data, colWidths=[2.5*inch, 1*inch, 3*inch])
            risks_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f5f5f5')),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('WORDWRAP', (0,0), (-1,-1), True)
            ]))
            story.append(risks_table)
        else:
            story.append(Paragraph("No risks defined.", styles["BodyText"]))
        
        # Next Steps
        story.append(Paragraph("7. Next Steps", styles["SectionHeading"]))
        for step in prd.get("next_steps", []):
            story.append(Paragraph(f"• {pdf_sanitize(step)}", styles["BodyText"]))
        
        # Version Info
        story.append(Spacer(1, 24))
        story.append(Paragraph(
            f"Generated by A/B Test Architect on {datetime.now().strftime('%Y-%m-%d %H:%M')} (v{st.session_state.prd_version})", 
            ParagraphStyle(name="Footer", fontSize=9, alignment=TA_CENTER)
        ))
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
        
    except Exception as e:
        st.error(f"PDF generation failed: {str(e)}")
        return None
    finally:
        buffer.close()

def render_prd_plan(plan: Dict[str, Any]) -> None:
    """Render the full PRD plan with proper sanitization and error handling"""
    plan = sanitize_experiment_plan(plan)
    sanitized_metric_name = sanitize_text(st.session_state.get('exact_metric', ''))
    
    # Problem Statement
    st.markdown(f"""
    <div class="prd-section">
        <div class="prd-section-title"><h2>1. Problem Statement</h2></div>
        <div class="prd-section-content">
            <p class="problem-statement">{html_sanitize(plan.get("problem_statement", ""))}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hypotheses
    hypotheses_html = ""
    for h in plan.get("hypotheses", []):
        hypotheses_html += f"""
        <div class='section-list-item'>
            <p class='hypothesis-title'>{html_sanitize(h.get('hypothesis', ''))}</p>
            <p><strong>Rationale:</strong> {html_sanitize(h.get('rationale', ''))}</p>
            <p><strong>Example:</strong> {html_sanitize(h.get('example_implementation', ''))}</p>
            <p><strong>Behavioral Basis:</strong> {html_sanitize(h.get('behavioral_basis', ''))}</p>
        </div>
        """
    
    st.markdown(f"""
    <div class="prd-section">
        <div class="prd-section-title"><h2>2. Hypotheses</h2></div>
        <div class="prd-section-content">
            <div class="section-list">{hypotheses_html}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Variants
    variants_html = ""
    for i, v in enumerate(plan.get("variants", [])):
        variants_html += f"""
        <div class='section-list-item'>
            <p><strong>Control {i+1}:</strong> {html_sanitize(v.get('control', ''))}</p>
            <p><strong>Variation {i+1}:</strong> {html_sanitize(v.get('variation', ''))}</p>
        </div>
        """
    
    st.markdown(f"""
    <div class="prd-section">
        <div class="prd-section-title"><h2>3. Variants</h2></div>
        <div class="prd-section-content">
            <div class="section-list">{variants_html}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    metrics_html = ""
    for m in plan.get("metrics", []):
        metrics_html += f"""
        <div class='section-list-item'>
            <p><strong>Name:</strong> {html_sanitize(m.get('name', ''))}</p>
            <p><strong>Formula:</strong> <code class='formula-code'>{html_sanitize(m.get('formula', ''))}</code></p>
            <p><strong>Importance:</strong> <span class='importance'>{html_sanitize(m.get('importance', ''))}</span></p>
        </div>
        """
    
    st.markdown(f"""
    <div class="prd-section">
        <div class="prd-section-title"><h2>4. Metrics</h2></div>
        <div class="prd-section-content">
            <div class="section-list">{metrics_html}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Success Criteria
    criteria = plan.get('success_criteria', {})
    stats_html = f"""
    <div class='section-list-item'>
        <p><strong>Confidence Level:</strong> {html_sanitize(criteria.get('confidence_level', ''))}%</p>
        <p><strong>Minimum Detectable Effect (MDE):</strong> {html_sanitize(criteria.get('MDE', ''))}%</p>
        <p><strong>Statistical Rationale:</strong> {html_sanitize(plan.get('statistical_rationale', ''))}</p>
    """
    
    if st.session_state.get('calculated_sample_size_per_variant'):
        stats_html += f"<p><strong>Sample Size per Variant:</strong> {st.session_state.calculated_sample_size_per_variant:,}</p>"
    if st.session_state.get('calculated_total_sample_size'):
        stats_html += f"<p><strong>Total Sample Size:</strong> {st.session_state.calculated_total_sample_size:,}</p>"
    if st.session_state.get('calculated_duration_days'):
        stats_html += f"<p><strong>Estimated Duration:</strong> {round(st.session_state.calculated_duration_days, 1)} days</p>"
    
    stats_html += "</div>"
    
    st.markdown(f"""
    <div class="prd-section">
        <div class="prd-section-title"><h2>5. Success Criteria & Statistical Rationale</h2></div>
        <div class="prd-section-content">
            <div class="section-list">{stats_html}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Risks
    risks_html = ""
    for r in plan.get("risks_and_assumptions", []):
        severity = str(r.get('severity', 'Medium')).title()
        severity_class = severity.lower() if severity.lower() in ['high', 'medium', 'low'] else 'medium'
        
        risks_html += f"""
        <div class='section-list-item'>
            <p><strong>Risk:</strong> {html_sanitize(r.get('risk', ''))}</p>
            <p><strong>Severity:</strong> <span class='severity {severity_class}'>{html_sanitize(severity)}</span></p>
            <p><strong>Mitigation:</strong> {html_sanitize(r.get('mitigation', ''))}</p>
        </div>
        """
    
    st.markdown(f"""
    <div class="prd-section">
        <div class="prd-section-title"><h2>6. Risks and Assumptions</h2></div>
        <div class="prd-section-content">
            <div class="section-list">{risks_html}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Next Steps
    next_steps_html = ""
    for step in plan.get("next_steps", []):
        next_steps_html += f"""
        <div class='section-list-item'>
            <p>{html_sanitize(step)}</p>
        </div>
        """
    
    st.markdown(f"""
    <div class="prd-section">
        <div class="prd-section-title"><h2>7. Next Steps</h2></div>
        <div class="prd-section-content">
            <div class="section-list">{next_steps_html}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Version History
    with st.expander("Version History", expanded=False):
        st.write(f"PRD v{st.session_state.prd_version} - Last updated: {st.session_state.last_updated}")
        if st.button("Increment Version"):
            st.session_state.prd_version += 1
            st.session_state.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M")
            st.rerun()

        
        
# --- Streamlit UI Setup ---
st.set_page_config(
    page_title="A/B Test Architect", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Improved CSS with mobile responsiveness and better contrast
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
        word-break: break-word;
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
        word-break: break-word;
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
        color: #dc2626;
        background-color: #fee2e2;
    }
    .severity.medium { 
        color: #d97706;
        background-color: #ffedd5;
    }
    .severity.low { 
        color: #16a34a;
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

    @media (max-width: 768px) {
        .prd-card {
            padding: 1rem;
        }
        .prd-header {
            flex-direction: column;
        }
        .logo-wrapper {
            margin-right: 0;
            margin-bottom: 1rem;
        }
        .header-text h1 {
            font-size: 1.5rem;
            text-align: center;
        }
        .header-text p {
            text-align: center;
        }
        .prd-section-content {
            padding: 0.75rem;
        }
        .section-list .list-item {
            padding: 0.5rem;
        }
    }

    /* Button spacing fixes */
    .stButton>button {
        margin-bottom: 0.5rem;
    }
    /* Fix table rendering */
    .section-list-item {
        page-break-inside: avoid !important;
    }
    /* PDF-specific fixes */
    @media print {
        .prd-section {
            page-break-after: avoid;
        }
        .section-list-item {
            page-break-inside: avoid;
        }
    }
    /* Edit form fixes */
    .stTextArea textarea {
        min-height: 150px !important;
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
    if "expander_states" not in st.session_state:
        st.session_state.expander_states = {
            "product_context": True,
            "metric_objective": True,
            "generate_plan": True,
            "calculator": True,
            "edit_plan": False
        }
    if "prd_version" not in st.session_state:
        st.session_state.prd_version = 1
    if "last_updated" not in st.session_state:
        st.session_state.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M")

init_session_state()

# --- Input Sections ---
st.title("💡 A/B Test Architect — AI-assisted experiment PRD generator")
st.markdown("Create experiment PRDs, hypotheses, stats, and sample-size guidance — faster and with guardrails.")

with st.expander("💡 Product Context (click to expand)", expanded=st.session_state.expander_states["product_context"]):
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

with st.expander("🎯 Metric Improvement Objective (click to expand)", expanded=st.session_state.expander_states["metric_objective"]):
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
            elif std_dev is not None and std_dev <= 0:
                st.error("Standard deviation must be positive.")

    metric_inputs_valid = True
    if current_value == target_value and current_value is not None:
        st.warning("The target metric must be different from the current metric to measure change. Please adjust one or the other.")
        metric_inputs_valid = False
    
    if metric_type == "Conversion Rate" and metric_unit != "%":
        st.warning("For 'Conversion Rate' metric type, the unit should be '%'.")
        metric_inputs_valid = False
with st.expander("🧠 Generate Experiment Plan", expanded=st.session_state.expander_states["generate_plan"]):
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
            target_value is not None,
            (metric_type != "Numeric Value" or std_dev is not None)
        ]
    )
    
    generate_col1, generate_col2 = st.columns([1, 1])
    with generate_col1:
        generate_btn = st.button("Generate Plan", disabled=not required_ok)
    with generate_col2:
        if st.button("Reset All", key="reset_all_btn"):
            st.session_state.clear()
            st.rerun()
    
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
                    if 'success_criteria' not in parsed:
                        parsed['success_criteria'] = {}
                    
                    try:
                        parsed['success_criteria']['confidence_level'] = max(80, min(99,float(parsed['success_criteria'].get('confidence_level', 95))))
                    except (ValueError, TypeError):
                        parsed['success_criteria']['confidence_level'] = 95
                    
                    try:
                        parsed['success_criteria']['MDE'] = max(0.1, 
                            float(parsed['success_criteria'].get('MDE', mde_default)))
                    except (ValueError, TypeError):
                        parsed['success_criteria']['MDE'] = max(0.1, mde_default)
                    
                    st.session_state.ai_parsed = sanitize_experiment_plan(parsed)
                    st.session_state.hypotheses_from_llm = st.session_state.ai_parsed.get("hypotheses", [])
                    st.success("Plan generated successfully — let's refine it step-by-step.")
                else:
                    st.error("Plan generation failed. Please check inputs and try again.")
                    st.session_state.stage = "input"
            except Exception as e:
                st.error(f"LLM generation failed: {str(e)}")
                st.session_state.stage = "input"

if st.session_state.get("ai_parsed"):
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 🚀 Refine Your Experiment Plan")

    if st.session_state.stage == "problem_statement":
        st.subheader("Step 1: Problem Statement")
        st.info("Review and refine the problem statement. This sets the foundation for your experiment.")
        
        plan = sanitize_experiment_plan(st.session_state.ai_parsed)
        problem_statement = plan.get('problem_statement', '')
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
        with col_ps_2:
            if st.button("Back to Inputs"):
                st.session_state.stage = "input"
                st.rerun()

    elif st.session_state.stage == "hypotheses":
        st.subheader("Step 2: Choose or Create a Hypothesis")
        st.info("Select one of the AI-generated hypotheses below, or create your own to get a fresh perspective.")
        
        st.session_state.ai_parsed['hypotheses'] = []
        plan = sanitize_experiment_plan(st.session_state.ai_parsed)
        
        hyp_cols = st.columns(3)
        for i, h in enumerate(st.session_state.hypotheses_from_llm):
            with hyp_cols[i % 3]:
                st.markdown(f"**Hypothesis {i+1}**")
                st.markdown(f"*{h.get('hypothesis', '')}*")
                if st.button(f"Select Hypothesis {i+1}", key=f"select_hyp_{i}"):
                    st.session_state.ai_parsed['hypotheses'].append(sanitize_experiment_plan(h))
                    st.session_state.stage = "full_plan"
                    st.session_state.temp_plan_edit = sanitize_experiment_plan(st.session_state.ai_parsed.copy())
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
                                st.session_state.ai_parsed['hypotheses'].append(sanitize_experiment_plan(hyp_details_parsed))
                                st.session_state.stage = "full_plan"
                                st.session_state.temp_plan_edit = sanitize_experiment_plan(st.session_state.ai_parsed.copy())
                                st.rerun()
                            else:
                                st.error("Failed to generate details for your hypothesis. Please try again.")
                        except Exception as e:
                            st.error(f"LLM call failed: {str(e)}")
            
            if st.button("Back to Problem Statement"):
                st.session_state.stage = "problem_statement"
                st.rerun()

    elif st.session_state.stage == "full_plan":
        st.subheader("Step 3: Refine the Full Plan")
        st.info("Your experiment plan is ready! Now you can edit any of the sections, starting with the A/B test calculator.")

        plan = sanitize_experiment_plan(st.session_state.ai_parsed)
        
        with st.expander("🔢 A/B Test Calculator: Fine-tune sample size", expanded=st.session_state.expander_states["calculator"]):
            if 'success_criteria' not in plan or not isinstance(plan['success_criteria'], dict):
                plan['success_criteria'] = {}
            
            try:
                calc_mde_initial = float(plan['success_criteria'].get('MDE', mde_default))
                calc_mde_initial = max(0.1, calc_mde_initial)
            except (ValueError, TypeError):
                calc_mde_initial = max(0.1, mde_default)
            
            try:
                calc_conf_initial = int(plan['success_criteria'].get('confidence_level', 95))
                calc_conf_initial = max(80, min(99, calc_conf_initial))
            except (ValueError, TypeError):
                calc_conf_initial = 95
                
            try:
                calc_power_initial = int(st.session_state.get('calc_power', 80))
                calc_power_initial = max(70, min(95, calc_power_initial))
            except (ValueError, TypeError):
                calc_power_initial = 80
                
            calc_variants = st.session_state.get('calc_variants', 2)
            
            col1, col2 = st.columns(2)
            with col1:
                calc_mde = st.number_input("Minimum Detectable Effect (MDE) %", 
                                         min_value=0.1, 
                                         max_value=50.0, 
                                         value=float(calc_mde_initial), 
                                         step=0.1, 
                                         key="calc_mde_key")
                                         
                calc_conf = st.number_input("Confidence Level (%)", 
                                          min_value=80, 
                                          max_value=99, 
                                          value=int(calc_conf_initial),
                                          step=1, 
                                          key="calc_conf_key")
            with col2:
                calc_power = st.number_input("Statistical Power (%)", 
                                           min_value=70, 
                                           max_value=95, 
                                           value=int(calc_power_initial), 
                                           step=1, 
                                           key="calc_power_key")
                                           
                calc_variants = st.number_input("Number of Variants (Control + Variations)", 
                                              min_value=2, 
                                              max_value=5, 
                                              value=int(calc_variants), 
                                              step=1, 
                                              key="calc_variants_key")
            
            if metric_type == "Numeric Value" and std_dev is not None:
                st.info(f"Standard Deviation pre-filled: {std_dev}")

            col_act1, col_act2, col_act3 = st.columns([1, 1, 1])
            with col_act1:
                refresh_btn = st.button("🔄 Calculate", key="calc_btn")
            with col_act2:
                lock_btn = False
                if st.session_state.get("calculated_sample_size_per_variant"):
                    lock_btn = st.button("🔒 Lock Values", key="lock_btn")
            with col_act3:
                reset_btn = st.button("🔄 Reset Calculator", key="reset_calc_btn")

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
                st.session_state.calculated_duration_days = (users_to_test / dau) if dau > 0 and users_to_test else None
                st.rerun()

            if 'lock_btn' in locals() and lock_btn:
                st.session_state.calc_locked = True
                if 'success_criteria' not in st.session_state.ai_parsed:
                    st.session_state.ai_parsed['success_criteria'] = {}
                st.session_state.ai_parsed['success_criteria']['MDE'] = calc_mde
                st.session_state.ai_parsed['success_criteria']['confidence_level'] = calc_conf
                st.success("Calculator values locked into the plan!")
                st.rerun()

            if reset_btn:
                st.session_state.calculated_sample_size_per_variant = None
                st.session_state.calculated_total_sample_size = None
                st.session_state.calculated_duration_days = None
                st.rerun()

            if st.session_state.get("calculated_sample_size_per_variant"):
                st.markdown("---")
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.markdown(f"**Sample Size per Variant:**  \n{st.session_state.calculated_sample_size_per_variant:,}")
                with col_res2:
                    st.markdown(f"**Total Sample Size:**  \n{st.session_state.calculated_total_sample_size:,}")
                
                if st.session_state.calculated_duration_days:
                    st.markdown(f"**Estimated Duration:**  \n**{round(st.session_state.calculated_duration_days, 1)}** days (with {dau:,} DAU)")
                else:
                    st.markdown("**Estimated Duration:** Could not calculate (check DAU value)")

        # Render the full PRD plan
        render_prd_plan(plan)

        # Export buttons moved outside edit section
        st.markdown("---")
        col_export1, col_export2, col_export3 = st.columns([1, 1, 2])
        with col_export1:
            if REPORTLAB_AVAILABLE:
                pdf_bytes = generate_pdf_bytes_from_prd_dict(plan, title=f"Experiment PRD: {sanitized_metric_name}")
                if pdf_bytes:
                    st.download_button(
                        label="⬇️ Export to PDF",
                        data=pdf_bytes,
                        file_name=f"experiment_prd_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        help="Download a professionally formatted PDF report"
                    )
            else:
                st.warning("PDF export requires reportlab")
        with col_export2:
            st.download_button(
                label="⬇️ Export to JSON",
                data=json.dumps(plan, indent=2),
                file_name=f"experiment_prd_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                help="Download raw JSON data"
            )
        with col_export3:
            if st.button("← Back to Hypothesis Selection"):
                st.session_state.stage = "hypotheses"
                st.rerun()

        # Edit Plan Section
        with st.expander("✏️ Edit Experiment Plan", expanded=st.session_state.expander_states["edit_plan"]):
            edited_plan = sanitize_experiment_plan(st.session_state.ai_parsed.copy())
            
            with st.form(key='edit_form'):
                st.subheader("1. Problem Statement")
                edited_plan['problem_statement'] = st.text_area(
                    "Problem Statement", 
                    value=edited_plan.get('problem_statement', ''),
                    height=150,
                    key="edit_problem_statement"
                )
                
                st.subheader("2. Hypotheses")
                if 'hypotheses' not in edited_plan or not isinstance(edited_plan['hypotheses'], list): 
                    edited_plan['hypotheses'] = []
                
                num_hypotheses = st.number_input(
                    "Number of Hypotheses", 
                    min_value=0, 
                    value=len(edited_plan['hypotheses']), 
                    key='num_hyp'
                )
                
                if num_hypotheses > len(edited_plan['hypotheses']):
                    edited_plan['hypotheses'].extend([
                        {"hypothesis": "", "rationale": "", "example_implementation": "", "behavioral_basis": ""}
                        for _ in range(num_hypotheses - len(edited_plan['hypotheses']))
                    ])
                elif num_hypotheses < len(edited_plan['hypotheses']):
                    edited_plan['hypotheses'] = edited_plan['hypotheses'][:num_hypotheses]
                    
                for i, h in enumerate(edited_plan.get("hypotheses", [])):
                    with st.expander(f"Hypothesis {i+1}", expanded=(i==0)):
                        edited_plan['hypotheses'][i]['hypothesis'] = st.text_input(
                            "Hypothesis", 
                            value=h.get('hypothesis', ''), 
                            key=f"hyp_{i}_text"
                        )
                        edited_plan['hypotheses'][i]['rationale'] = st.text_area(
                            "Rationale", 
                            value=h.get('rationale', ''), 
                            height=100, 
                            key=f"hyp_{i}_rationale"
                        )
                        edited_plan['hypotheses'][i]['example_implementation'] = st.text_area(
                            "Implementation Example", 
                            value=h.get('example_implementation', ''), 
                            height=100, 
                            key=f"hyp_{i}_impl"
                        )
                        edited_plan['hypotheses'][i]['behavioral_basis'] = st.text_input(
                            "Behavioral Basis", 
                            value=h.get('behavioral_basis', ''), 
                            key=f"hyp_{i}_basis"
                        )
                
                st.subheader("3. Variants")
                if 'variants' not in edited_plan or not isinstance(edited_plan['variants'], list): 
                    edited_plan['variants'] = []
                
                num_variants = st.number_input(
                    "Number of Variants", 
                    min_value=1, 
                    value=len(edited_plan['variants']), 
                    key='num_variants'
                )
                
                if num_variants > len(edited_plan['variants']):
                    edited_plan['variants'].extend([
                        {"control": "", "variation": ""}
                        for _ in range(num_variants - len(edited_plan['variants']))
                    ])
                elif num_variants < len(edited_plan['variants']):
                    edited_plan['variants'] = edited_plan['variants'][:num_variants]
                
                for i, v in enumerate(edited_plan.get('variants', [])):
                    with st.expander(f"Variant {i+1}", expanded=(i==0)):
                        edited_plan['variants'][i]['control'] = st.text_input(
                            "Control", 
                            value=v.get('control', ''), 
                            key=f"var_{i}_control"
                        )
                        edited_plan['variants'][i]['variation'] = st.text_input(
                            "Variation", 
                            value=v.get('variation', ''), 
                            key=f"var_{i}_variation"
                        )
                
                st.subheader("4. Metrics")
                if 'metrics' not in edited_plan or not isinstance(edited_plan['metrics'], list): 
                    edited_plan['metrics'] = []
                
                num_metrics = st.number_input(
                    "Number of Metrics", 
                    min_value=1, 
                    value=len(edited_plan['metrics']), 
                    key='num_metrics'
                )
                
                if num_metrics > len(edited_plan['metrics']):
                    edited_plan['metrics'].extend([
                        {"name": "", "formula": "", "importance": "Primary"}
                        for _ in range(num_metrics - len(edited_plan['metrics']))
                    ])
                elif num_metrics < len(edited_plan['metrics']):
                    edited_plan['metrics'] = edited_plan['metrics'][:num_metrics]
                
                for i, m in enumerate(edited_plan.get("metrics", [])):
                    with st.expander(f"Metric {i+1}", expanded=(i==0)):
                        edited_plan['metrics'][i]['name'] = st.text_input(
                            "Name", 
                            value=m.get('name', ''), 
                            key=f"met_{i}_name"
                        )
                        edited_plan['metrics'][i]['formula'] = st.text_input(
                            "Formula", 
                            value=m.get('formula', ''), 
                            key=f"met_{i}_formula"
                        )
                        importance_value = m.get('importance', 'Primary')
                        if importance_value not in ["Primary", "Secondary", "Guardrail"]:
                            importance_value = "Primary"
                        edited_plan['metrics'][i]['importance'] = st.selectbox(
                            "Importance", 
                            options=["Primary", "Secondary", "Guardrail"], 
                            index=["Primary", "Secondary", "Guardrail"].index(importance_value), 
                            key=f"met_{i}_imp"
                        )
                
                st.subheader("5. Success Criteria")
                if 'success_criteria' not in edited_plan or not isinstance(edited_plan['success_criteria'], dict): 
                    edited_plan['success_criteria'] = {}
                
                try:
                    conf_level = float(edited_plan['success_criteria'].get('confidence_level', 95))
                    conf_level = max(80, min(99, conf_level))
                except (ValueError, TypeError):
                    conf_level = 95
                
                edited_plan['success_criteria']['confidence_level'] = st.number_input(
                    "Confidence Level (%)", 
                    min_value=80, 
                    max_value=99, 
                    value=int(conf_level),
                    step=1,
                    key="edit_conf_level"
                )
                
                try:
                    mde_value = float(edited_plan['success_criteria'].get('MDE', mde_default))
                    mde_value = max(0.1, mde_value)
                except (ValueError, TypeError):
                    mde_value = max(0.1, mde_default)
                
                edited_plan['success_criteria']['MDE'] = st.number_input(
                    "Minimum Detectable Effect (%)", 
                    min_value=0.1, 
                    max_value=100.0,
                    value=float(mde_value),
                    step=0.1,
                    format="%.1f",
                    key="edit_mde"
                )
                
                edited_plan['statistical_rationale'] = st.text_area(
                    "Statistical Rationale", 
                    value=edited_plan.get('statistical_rationale', ''), 
                    height=100,
                    key="edit_stat_rationale"
                )
                
                edited_plan['success_criteria']['benchmark'] = st.text_input(
                    "Benchmark", 
                    value=edited_plan['success_criteria'].get('benchmark', ''), 
                    key="edit_benchmark"
                )
                
                edited_plan['success_criteria']['monitoring'] = st.text_input(
                    "Monitoring", 
                    value=edited_plan['success_criteria'].get('monitoring', ''), 
                    key="edit_monitoring"
                )
                
                st.subheader("6. Risks and Assumptions")
                if 'risks_and_assumptions' not in edited_plan or not isinstance(edited_plan['risks_and_assumptions'], list): 
                    edited_plan['risks_and_assumptions'] = []
                
                num_risks = st.number_input(
                    "Number of Risks", 
                    min_value=0, 
                    value=len(edited_plan['risks_and_assumptions']), 
                    key='num_risks'
                )
                
                if num_risks > len(edited_plan['risks_and_assumptions']):
                    edited_plan['risks_and_assumptions'].extend([
                        {"risk": "", "severity": "Medium", "mitigation": ""}
                        for _ in range(num_risks - len(edited_plan['risks_and_assumptions']))
                    ])
                elif num_risks < len(edited_plan['risks_and_assumptions']):
                    edited_plan['risks_and_assumptions'] = edited_plan['risks_and_assumptions'][:num_risks]
                
                for i, r in enumerate(edited_plan.get("risks_and_assumptions", [])):
                    with st.expander(f"Risk {i+1}", expanded=(i==0)):
                        edited_plan['risks_and_assumptions'][i]['risk'] = st.text_input(
                            "Risk", 
                            value=r.get('risk', ''), 
                            key=f"risk_{i}_text"
                        )
                        severity = r.get('severity', 'Medium')
                        if severity not in ["High", "Medium", "Low"]:
                            severity = "Medium"
                        edited_plan['risks_and_assumptions'][i]['severity'] = st.selectbox(
                            "Severity", 
                            options=["High", "Medium", "Low"], 
                            index=["High", "Medium", "Low"].index(severity), 
                            key=f"risk_{i}_severity"
                        )
                        edited_plan['risks_and_assumptions'][i]['mitigation'] = st.text_area(
                            "Mitigation", 
                            value=r.get('mitigation', ''), 
                            height=100,
                            key=f"risk_{i}_mitigation"
                        )
                
                st.subheader("7. Next Steps")
                if 'next_steps' not in edited_plan or not isinstance(edited_plan['next_steps'], list): 
                    edited_plan['next_steps'] = []
                
                next_steps_text = "\n".join(edited_plan.get('next_steps', []))
                new_next_steps = st.text_area(
                    "Next Steps (one per line)", 
                    value=next_steps_text, 
                    height=150, 
                    key="edit_next_steps"
                )
                edited_plan['next_steps'] = [
                    step.strip() for step in new_next_steps.split('\n') if step.strip()
                ]
                
                submitted = st.form_submit_button("💾 Save Changes")
                if submitted:
                    st.session_state.ai_parsed = sanitize_experiment_plan(edited_plan)
                    st.session_state.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M")
                    st.success("Plan updated successfully!")
                    st.rerun()


# --- Final UI Polish ---
if st.session_state.get("ai_parsed") and st.session_state.stage == "full_plan":
    # Additional export buttons at bottom of page
    st.markdown("---")
    st.markdown("### Export Options")
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if REPORTLAB_AVAILABLE:
            pdf_bytes = generate_pdf_bytes_from_prd_dict(
                st.session_state.ai_parsed, 
                title=f"Experiment PRD: {sanitize_text(st.session_state.get('exact_metric', ''))}"
            )
            if pdf_bytes:
                st.download_button(
                    label="⬇️ Export to PDF (Full Report)",
                    data=pdf_bytes,
                    file_name=f"experiment_prd_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    help="Download complete professional PDF report"
                )
    
    with col2:
        st.download_button(
            label="⬇️ Export to JSON (Raw Data)",
            data=json.dumps(st.session_state.ai_parsed, indent=2),
            file_name=f"experiment_prd_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            help="Download raw JSON data for integration"
        )
    
    with col3:
        if st.button("🔄 Start New Experiment"):
            st.session_state.clear()
            st.rerun()

    # Version history expander
    with st.expander("📜 Version History", expanded=False):
        st.write(f"Current version: v{st.session_state.prd_version}")
        st.write(f"Last updated: {st.session_state.last_updated}")
        if st.button("Create New Version"):
            st.session_state.prd_version += 1
            st.session_state.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M")
            st.success(f"Created new version v{st.session_state.prd_version}")
            st.rerun()
