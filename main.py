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
from bs4 import BeautifulSoup

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
        # Ensure consistent field names
        risk_data = {
            "risk": str(risk.get("risk", risk.get("risks", ""))),
            "severity": str(risk.get("severity", "Medium")).title(),
            "mitigation": str(risk.get("mitigation", risk.get("mitigations", "")))
        }
        sanitized_plan["risks_and_assumptions"][i] = risk_data

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
    # First unescape any existing HTML
    text = html.unescape(text)
    # Then re-escape properly
    text = html.escape(text)
    # Fix common tag issues
    text = text.replace("<postrong>", "<p><strong>").replace("</postrong>", "</strong></p>")
    # Fix unclosed tags
    text = re.sub(r"<p>(.*?)(?=<p>|$)", r"<p>\1</p>", text)
    return text

def safe_html_render(html_content: str) -> None:
    from bs4 import BeautifulSoup
    try:
        # Validate HTML structure
        soup = BeautifulSoup(html_content, 'html.parser')
        # Fix any structural issues
        for item in soup.find_all(class_='section-list-item'):
            if not item.find('p'):
                item.wrap(soup.new_tag('p'))
        st.markdown(str(soup), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error rendering content: {str(e)}")
        st.code(html_content)  # Show the problematic HTML

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
def render_prd_plan(plan: Dict[str, Any]) -> None:
    """Render the full PRD plan with inline editing functionality"""
    plan = sanitize_experiment_plan(plan)
    sanitized_metric_name = sanitize_text(st.session_state.get('exact_metric', ''))
    
    # Edit modal for sections
    if st.session_state.get('editing_section'):
        with st.form(key=f"edit_{st.session_state.editing_section}"):
            section_data = plan.get(st.session_state.editing_section, "")
            
            if st.session_state.editing_section == "problem_statement":
                edited_content = st.text_area(
                    "Edit Problem Statement",
                    value=section_data,
                    height=200,
                    key="edit_problem_statement"
                )
            elif st.session_state.editing_section == "statistical_rationale":
                edited_content = st.text_area(
                    "Edit Statistical Rationale",
                    value=section_data,
                    height=150,
                    key="edit_stat_rationale"
                )
            elif st.session_state.editing_section == "next_steps":
                edited_content = st.text_area(
                    "Edit Next Steps (one per line)",
                    value="\n".join(section_data),
                    height=150,
                    key="edit_next_steps"
                )
            
            col1, col2 = st.columns([1,1])
            with col1:
                if st.form_submit_button("💾 Save Changes"):
                    if st.session_state.editing_section == "next_steps":
                        plan["next_steps"] = [step.strip() for step in edited_content.split('\n') if step.strip()]
                    else:
                        plan[st.session_state.editing_section] = edited_content
                    st.session_state.ai_parsed = plan
                    st.session_state.editing_section = None
                    st.rerun()
            with col2:
                if st.form_submit_button("❌ Cancel"):
                    st.session_state.editing_section = None
                    st.rerun()

    # Edit modal for items in lists
    if st.session_state.get('editing_item') and st.session_state.get('editing_item_index') is not None:
        item_type = st.session_state.editing_item
        item_index = st.session_state.editing_item_index
        item_data = plan.get(item_type, [])[item_index] if item_index < len(plan.get(item_type, [])) else {}
        
        with st.form(key=f"edit_{item_type}_{item_index}"):
            if item_type == "hypotheses":
                st.subheader("Edit Hypothesis")
                edited_hypothesis = st.text_input(
                    "Hypothesis",
                    value=item_data.get("hypothesis", ""),
                    key=f"edit_hypothesis_{item_index}"
                )
                edited_rationale = st.text_area(
                    "Rationale",
                    value=item_data.get("rationale", ""),
                    height=100,
                    key=f"edit_rationale_{item_index}"
                )
                edited_implementation = st.text_area(
                    "Implementation Example",
                    value=item_data.get("example_implementation", ""),
                    height=100,
                    key=f"edit_impl_{item_index}"
                )
                edited_basis = st.text_input(
                    "Behavioral Basis",
                    value=item_data.get("behavioral_basis", ""),
                    key=f"edit_basis_{item_index}"
                )
                
                edited_item = {
                    "hypothesis": edited_hypothesis,
                    "rationale": edited_rationale,
                    "example_implementation": edited_implementation,
                    "behavioral_basis": edited_basis
                }
                
            elif item_type == "variants":
                st.subheader("Edit Variant")
                edited_control = st.text_input(
                    "Control",
                    value=item_data.get("control", ""),
                    key=f"edit_control_{item_index}"
                )
                edited_variation = st.text_input(
                    "Variation",
                    value=item_data.get("variation", ""),
                    key=f"edit_variation_{item_index}"
                )
                
                edited_item = {
                    "control": edited_control,
                    "variation": edited_variation
                }
                
            elif item_type == "metrics":
                st.subheader("Edit Metric")
                edited_name = st.text_input(
                    "Name",
                    value=item_data.get("name", ""),
                    key=f"edit_metric_name_{item_index}"
                )
                edited_formula = st.text_input(
                    "Formula",
                    value=item_data.get("formula", ""),
                    key=f"edit_formula_{item_index}"
                )
                importance_value = item_data.get("importance", "Primary")
                if importance_value not in ["Primary", "Secondary", "Guardrail"]:
                    importance_value = "Primary"
                edited_importance = st.selectbox(
                    "Importance",
                    options=["Primary", "Secondary", "Guardrail"],
                    index=["Primary", "Secondary", "Guardrail"].index(importance_value),
                    key=f"edit_importance_{item_index}"
                )
                
                edited_item = {
                    "name": edited_name,
                    "formula": edited_formula,
                    "importance": edited_importance
                }
                
            elif item_type == "risks_and_assumptions":
                st.subheader("Edit Risk")
                edited_risk = st.text_input(
                    "Risk",
                    value=item_data.get("risk", ""),
                    key=f"edit_risk_{item_index}"
                )
                severity = item_data.get("severity", "Medium")
                if severity not in ["High", "Medium", "Low"]:
                    severity = "Medium"
                edited_severity = st.selectbox(
                    "Severity",
                    options=["High", "Medium", "Low"],
                    index=["High", "Medium", "Low"].index(severity),
                    key=f"edit_severity_{item_index}"
                )
                edited_mitigation = st.text_area(
                    "Mitigation",
                    value=item_data.get("mitigation", ""),
                    height=100,
                    key=f"edit_mitigation_{item_index}"
                )
                
                edited_item = {
                    "risk": edited_risk,
                    "severity": edited_severity,
                    "mitigation": edited_mitigation
                }
            
            col1, col2 = st.columns([1,1])
            with col1:
                if st.form_submit_button("💾 Save Changes"):
                    if item_index < len(plan.get(item_type, [])):
                        plan[item_type][item_index] = edited_item
                    else:
                        if item_type not in plan:
                            plan[item_type] = []
                        plan[item_type].append(edited_item)
                    st.session_state.ai_parsed = plan
                    st.session_state.editing_item = None
                    st.session_state.editing_item_index = None
                    st.rerun()
            with col2:
                if st.form_submit_button("❌ Cancel"):
                    st.session_state.editing_item = None
                    st.session_state.editing_item_index = None
                    st.rerun()

    # Problem Statement with edit button
    st.markdown(f"""
    <div class="prd-section">
        <div class="prd-section-title" style="display: flex; justify-content: space-between; align-items: center;">
            <h2>1. Problem Statement</h2>
            <button class="edit-btn" onclick="window.parent.document.querySelector('button[data-testid=\"baseButton-secondary\"]').click()">✏️ Edit</button>
        </div>
        <div class="prd-section-content">
            <p class="problem-statement">{html_sanitize(plan.get("problem_statement", ""))}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Edit Problem Statement", key="edit_problem_btn", type="secondary"):
        st.session_state.editing_section = "problem_statement"
        st.rerun()

    # Hypotheses with edit buttons
    hypotheses_html = ""
    for i, h in enumerate(plan.get("hypotheses", [])):
        hypotheses_html += f"""
        <div class='section-list-item'>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <p class='hypothesis-title'>{html_sanitize(h.get('hypothesis', ''))}</p>
                <button class="edit-btn" onclick="window.parent.document.querySelector('button[data-testid=\"baseButton-secondary-{i}\"]').click()">✏️ Edit</button>
            </div>
            <p><strong>Rationale:</strong> {html_sanitize(h.get('rationale', ''))}</p>
            <p><strong>Example:</strong> {html_sanitize(h.get('example_implementation', ''))}</p>
            <p><strong>Behavioral Basis:</strong> {html_sanitize(h.get('behavioral_basis', ''))}</p>
        </div>
        """
        if st.button(f"Edit Hypothesis {i+1}", key=f"edit_hyp_btn_{i}", type="secondary"):
            st.session_state.editing_item = "hypotheses"
            st.session_state.editing_item_index = i
            st.rerun()
    
    st.markdown(f"""
    <div class="prd-section">
        <div class="prd-section-title" style="display: flex; justify-content: space-between; align-items: center;">
            <h2>2. Hypotheses</h2>
            <button class="edit-btn" onclick="window.parent.document.querySelector('button[data-testid=\"baseButton-secondary-add-hyp\"]').click()">➕ Add New</button>
        </div>
        <div class="prd-section-content">
            <div class="section-list">{hypotheses_html}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("➕ Add New Hypothesis", key="add_hyp_btn", type="secondary"):
        st.session_state.editing_item = "hypotheses"
        st.session_state.editing_item_index = len(plan.get("hypotheses", []))
        st.rerun()
    # Variants with edit buttons
    variants_html = ""
    for i, v in enumerate(plan.get("variants", [])):
        variants_html += f"""
        <div class='section-list-item'>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4>Variant {i+1}</h4>
                <button class="edit-btn" onclick="window.parent.document.querySelector('button[data-testid=\"baseButton-secondary-var-{i}\"]').click()">✏️ Edit</button>
            </div>
            <p><strong>Control {i+1}:</strong> {html_sanitize(v.get('control', ''))}</p>
            <p><strong>Variation {i+1}:</strong> {html_sanitize(v.get('variation', ''))}</p>
        </div>
        """
        if st.button(f"Edit Variant {i+1}", key=f"edit_var_btn_{i}", type="secondary"):
            st.session_state.editing_item = "variants"
            st.session_state.editing_item_index = i
            st.rerun()
    
    st.markdown(f"""
    <div class="prd-section">
        <div class="prd-section-title" style="display: flex; justify-content: space-between; align-items: center;">
            <h2>3. Variants</h2>
            <button class="edit-btn" onclick="window.parent.document.querySelector('button[data-testid=\"baseButton-secondary-add-var\"]').click()">➕ Add New</button>
        </div>
        <div class="prd-section-content">
            <div class="section-list">{variants_html}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("➕ Add New Variant", key="add_var_btn", type="secondary"):
        st.session_state.editing_item = "variants"
        st.session_state.editing_item_index = len(plan.get("variants", []))
        st.rerun()

    # Metrics with edit buttons
    metrics_html = ""
    for m in plan.get("metrics", []):
        metrics_html += f"""
        <div class='section-list-item'>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4>{html_sanitize(m.get('name', ''))}</h4>
                <button class="edit-btn" onclick="window.parent.document.querySelector('button[data-testid=\"baseButton-secondary-met-{i}\"]').click()">✏️ Edit</button>
            </div>
            <p><strong>Formula:</strong> <code class='formula-code'>{html_sanitize(m.get('formula', ''))}</code></p>
            <p><strong>Importance:</strong> <span class='importance'>{html_sanitize(m.get('importance', ''))}</span></p>
        </div>
        """
        if st.button(f"Edit Metric {i+1}", key=f"edit_met_btn_{i}", type="secondary"):
            st.session_state.editing_item = "metrics"
            st.session_state.editing_item_index = i
            st.rerun()
    
    st.markdown(f"""
    <div class="prd-section">
        <div class="prd-section-title" style="display: flex; justify-content: space-between; align-items: center;">
            <h2>4. Metrics</h2>
            <button class="edit-btn" onclick="window.parent.document.querySelector('button[data-testid=\"baseButton-secondary-add-met\"]').click()">➕ Add New</button>
        </div>
        <div class="prd-section-content">
            <div class="section-list">{metrics_html}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("➕ Add New Metric", key="add_met_btn", type="secondary"):
        st.session_state.editing_item = "metrics"
        st.session_state.editing_item_index = len(plan.get("metrics", []))
        st.rerun()

    # Success Criteria with edit button
    criteria = plan.get('success_criteria', {})
    stats_html = f"""
    <div class='section-list-item'>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h4>Success Criteria</h4>
            <button class="edit-btn" onclick="window.parent.document.querySelector('button[data-testid=\"baseButton-secondary-edit-stats\"]').click()">✏️ Edit</button>
        </div>
        <p><strong>Confidence Level:</strong> {html_sanitize(criteria.get('confidence_level', ''))}%</p>
        <p><strong>Minimum Detectable Effect (MDE):</strong> {html_sanitize(criteria.get('MDE', ''))}%</p>
        <p><strong>Statistical Rationale:</strong> {html_sanitize(plan.get('statistical_rationale', ''))}</p>
    """
    
    if st.session_state.get('calculated_sample_size_per_variant'):
        stats_html += f"<p><strong>Sample Size per Variant:</strong> {st.session_state.calculated_sample_size_per_variant:,}</p>"
    if st.session_state.get('calculated_total_sample_size'):
        stats_html += f"<p><strong>Total Sample Size:</strong> {st.session_state.calculated_total_sample_size:,}</p>"
    if st.session_state.get('calculated_duration_days'):
        stats_html += f"<p><strong>Estimated Duration:</strong> **{round(st.session_state.calculated_duration_days, 1)}** days (with {dau:,} DAU)</p>"
    
    stats_html += "</div>"
    
    st.markdown(f"""
    <div class="prd-section">
        <div class="prd-section-title">
            <h2>5. Success Criteria & Statistical Rationale</h2>
        </div>
        <div class="prd-section-content">
            <div class="section-list">{stats_html}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Edit Success Criteria", key="edit_stats_btn", type="secondary"):
        st.session_state.editing_section = "statistical_rationale"
        st.rerun()

    # Risks with edit buttons
    risks_content = []
    for i, r in enumerate(plan.get("risks_and_assumptions", [])):
        severity = str(r.get('severity', 'Medium')).title()
        severity_class = severity.lower() if severity.lower() in ['high', 'medium', 'low'] else 'medium'
        risks_content.append(f"""
        <div class='section-list-item'>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4>Risk {i+1}</h4>
                <button class="edit-btn" onclick="window.parent.document.querySelector('button[data-testid=\"baseButton-secondary-risk-{i}\"]').click()">✏️ Edit</button>
            </div>
            <p><strong>Risk:</strong> {html_sanitize(r.get('risk', r.get('risks', '')))}</p>
            <p><strong>Severity:</strong> <span class='severity {severity_class}'>{html_sanitize(severity)}</span></p>
            <p><strong>Mitigation:</strong> {html_sanitize(r.get('mitigation', r.get('mitigations', '')))}</p>
        </div>
        """)
        if st.button(f"Edit Risk {i+1}", key=f"edit_risk_btn_{i}", type="secondary"):
            st.session_state.editing_item = "risks_and_assumptions"
            st.session_state.editing_item_index = i
            st.rerun()

    risks_html = "\n".join(risks_content)
    
    safe_html_render(f"""
    <div class="prd-section">
        <div class="prd-section-title" style="display: flex; justify-content: space-between; align-items: center;">
            <h2>6. Risks and Assumptions</h2>
            <button class="edit-btn" onclick="window.parent.document.querySelector('button[data-testid=\"baseButton-secondary-add-risk\"]').click()">➕ Add New</button>
        </div>
        <div class="prd-section-content">
            <div class="section-list">{risks_html}</div>
        </div>
    </div>
    """)

    if st.button("➕ Add New Risk", key="add_risk_btn", type="secondary"):
        st.session_state.editing_item = "risks_and_assumptions"
        st.session_state.editing_item_index = len(plan.get("risks_and_assumptions", []))
        st.rerun()

    # Next Steps with edit button
    next_steps_html = ""
    for i, step in enumerate(plan.get("next_steps", [])):
        next_steps_html += f"""
        <div class='section-list-item'>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <p>{html_sanitize(step)}</p>
                <button class="edit-btn" onclick="window.parent.document.querySelector('button[data-testid=\"baseButton-secondary-step-{i}\"]').click()">✏️ Edit</button>
            </div>
        </div>
        """
        if st.button(f"Edit Step {i+1}", key=f"edit_step_btn_{i}", type="secondary"):
            st.session_state.editing_item = "next_steps"
            st.session_state.editing_item_index = i
            st.rerun()
    
    safe_html_render(f"""
    <div class="prd-section">
        <div class="prd-section-title" style="display: flex; justify-content: space-between; align-items: center;">
            <h2>7. Next Steps</h2>
            <button class="edit-btn" onclick="window.parent.document.querySelector('button[data-testid=\"baseButton-secondary-add-step\"]').click()">➕ Add New</button>
        </div>
        <div class="prd-section-content">
            <div class="section-list">{next_steps_html}</div>
        </div>
    </div>
    """)

    if st.button("➕ Add New Step", key="add_step_btn", type="secondary"):
        st.session_state.editing_item = "next_steps"
        st.session_state.editing_item_index = len(plan.get("next_steps", []))
        st.rerun()
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

        # Render the full PRD plan with inline edit buttons
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

        # Version history expander
        with st.expander("📜 Version History", expanded=False):
            st.write(f"Current version: v{st.session_state.prd_version}")
            st.write(f"Last updated: {st.session_state.last_updated}")
            if st.button("Create New Version"):
                st.session_state.prd_version += 1
                st.session_state.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M")
                st.success(f"Created new version v{st.session_state.prd_version}")
                st.rerun()
