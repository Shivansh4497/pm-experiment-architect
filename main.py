# main.py — Final Certified Version (A/B Test Architect)
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
    """Escapes special HTML characters from a string."""
    if text is None: return ""
    text = str(text)
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    text = text.replace("'", "&apos;")
    return text

def generate_problem_statement(plan: Dict, current: float, target: float, unit: str) -> str:
    """Auto-inserts target metric into problem statement"""
    base = plan.get("problem_statement", "")
    if not base.strip():
        return base
    
    metric_str = f" (current: {format_value_with_unit(current, unit)} → target: {format_value_with_unit(target, unit)})"
    
    # Insert after first sentence if not already present
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
    """Extracts a numeric value and unit from a string with validation."""
    text = sanitize_text(text)
    match = re.match(r"([\d\.]+)\s*(\w+|%)?", text)
    if not match:
        try:
            return float(text), default_unit
        except ValueError:
            return None, default_unit
    
    value = float(match.group(1))
    unit = match.group(2) if match.group(2) else default_unit
    
    # Add validation for unit mismatch
    if default_unit != '%' and unit != default_unit:  # '%' is special case (common in LLM outputs)
        st.warning(f"Unit mismatch: Using '{unit}' from input instead of selected '{default_unit}'")
    
    return value, unit

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
    
    def pdf_sanitize(text: Any) -> str:
        if text is None: return ""
        text = str(text)
        # Escape XML special chars
        text = (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&apos;"))
        return text
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Sanitize all content
    sanitized_prd = {
        k: pdf_sanitize(v) if isinstance(v, str) else v
        for k, v in prd.items()
    }
    
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
    doc.build(story)
    buffer.seek(0)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
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
</style>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800&display=swap" rel="stylesheet">
""",
    unsafe_allow_html=True,
)

st.title("💡 A/B Test Architect — AI-assisted experiment PRD generator")
st.markdown("Create experiment PRDs, hypotheses, stats, and sample-size guidance — faster and with guardrails.")

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
if "raw_llm_edit" not in st.session_state:
    st.session_state.raw_llm_edit = ""

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
            "Metric Unit", ["%", "USD", "INR", "minutes", "count", "other"], index=0, help="Choose the unit for clarity."
        )
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
            std_dev_raw = st.text_input(
                "Standard Deviation of Metric * (required for numeric metrics)",
                placeholder="e.g., 10.5",
                help="The standard deviation is required for numeric metrics to compute sample sizes.",
            )
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
            user_base_choice,
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
        st.session_state.selected_index = None
        st.session_state.hypothesis_confirmed = False
        st.session_state.calc_locked = False
        st.session_state.locked_stats = {}
        st.session_state.ai_parsed = None # Clear old plan

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
                st.session_state.raw_llm_edit = st.session_state.output
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

        col_act1, col_act2 = st.columns([1, 1])
        with col_act1:
            btn_label = "Calculate" if not st.session_state.get("calculated_sample_size_per_variant") else "Refresh Calculator"
            refresh_btn = st.button(btn_label)
        with col_act2:
            lock_btn = False
            if st.session_state.get("calculated_sample_size_per_variant"):
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
            
            # New metrics row
            cols = st.columns(4)
            with cols[0]:
                st.metric("Current DAU", user_base_choice)
            with cols[1]:
                st.metric("Primary Metric", sanitized_metric_name)
            with cols[2]:
                st.metric("Current Value", formatted_current)
            with cols[3]:
                st.metric("Target Metric", formatted_target)
            
            st.markdown("---")
            
            # Old metrics row
            cols = st.columns(3)
            with cols[0]:
                st.metric("Users Per Variant", f"{st.session_state.calculated_sample_size_per_variant:,}")
            with cols[1]:
                st.metric("Total Sample Size", f"{st.session_state.calculated_total_sample_size:,}")
            with cols[2]:
                duration_display = f"{st.session_state.calculated_duration_days:,.0f} days" if np.isfinite(st.session_state.calculated_duration_days) else "∞"
                st.metric("Estimated Duration", duration_display)
            
            st.caption("Assumes all DAU are eligible and evenly split across variants.")
        else:
            st.info("Click 'Calculate' to compute sample size with current inputs.")

        if lock_btn and st.session_state.get("calculated_sample_size_per_variant"):
            st.session_state.calc_locked = True
            st.session_state.locked_stats = {
                "confidence_level": st.session_state.last_calc_confidence,
                "statistical_power": st.session_state.last_calc_power,
                "mde": st.session_state.last_calc_mde,
                "users_per_variant": st.session_state.calculated_sample_size_per_variant,
                "total_sample_size": st.session_state.calculated_total_sample_size,
                "estimated_test_duration_days": st.session_state.calculated_duration_days if np.isfinite(st.session_state.calculated_duration_days) else 'Not applicable',
            }
            st.success("Calculator values locked and will be used in the final plan.")
        st.markdown("<hr>", unsafe_allow_html=True)

if st.session_state.get("ai_parsed") is None and st.session_state.get("output"):
    st.markdown("<div class='blue-section'>", unsafe_allow_html=True)
    create_header_with_help("Raw LLM Output (fix JSON here)", "When parsing fails you'll see the raw LLM output — edit it then click Parse JSON.", icon="🛠️")
    raw_edit = st.text_area("Raw LLM output / edit here", value=st.session_state.get("raw_llm_edit", ""), height=400, key="raw_llm_edit")
    if st.button("Parse JSON"):
        parsed_try = extract_json(raw_edit)
        if parsed_try:
            st.session_state.ai_parsed = parsed_try
            st.success("Manual parse succeeded — plan is now usable.")
        else:
            st.error("Manual parse failed — edit the text and try again.")
    st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.get("ai_parsed"):
    plan = st.session_state.ai_parsed
    if "context" not in st.session_state:
        st.session_state.context = {"metric_unit": metric_unit}

    unit = st.session_state.context.get("metric_unit", metric_unit)
    
    st.markdown("<div class='green-section'>", unsafe_allow_html=True)
    create_header_with_help("Inferred Product Goal", "The AI's interpretation of your goal. Edit if needed.", icon="🎯")
    safe_display(post_process_llm_text(goal_with_units, unit))

    create_header_with_help("Problem Statement", "Clear description of the gap and why it matters.", icon="🧩")
    problem_statement = generate_problem_statement(plan, current_value, target_value, unit)
    st.markdown(problem_statement or "⚠️ Problem statement not generated by the model.")
    with st.expander("Edit Problem Statement"):
        st.text_area("Problem Statement (edit)", value=problem_statement, key="editable_problem", height=160)

    create_header_with_help("Hypotheses", "Testable hypotheses with full details.", icon="🧪")
    hypotheses = plan.get("hypotheses", [])
    if not hypotheses or not isinstance(hypotheses, list):
        st.warning("No hypotheses found in the generated plan.")
        hypotheses = []
        
    for i, h in enumerate(hypotheses):
        with st.container():
            cols = st.columns([5, 1])
            with cols[0]:
                st.markdown(f"""
                    **H{i+1}:** {h.get('hypothesis', '')}  
                    *Rationale:* {h.get('rationale', h.get('behavioral_basis', ''))}  
                    *Example:* {h.get('example_implementation', '')}
                """)
            with cols[1]:
                if st.button("Select", key=f"select_{i}"):
                    st.session_state.selected_index = i
                    st.session_state.hypothesis_confirmed = True

    if st.session_state.get("hypothesis_confirmed") and st.session_state.get("selected_index") is not None:
        # Add safety check for hypothesis indexing
        if st.session_state.selected_index >= len(hypotheses):
            st.session_state.selected_index = None
            st.session_state.hypothesis_confirmed = False
        else:
            selected_hypo = hypotheses[st.session_state.selected_index]

            st.subheader("🔍 Selected Hypothesis Details")
            cols = st.columns([4, 1])
            with cols[0]:
                st.markdown(f"**Hypothesis:** {selected_hypo.get('hypothesis', 'N/A')}")
                st.markdown(f"**Rationale:** {selected_hypo.get('rationale', selected_hypo.get('behavioral_basis', 'N/A'))}")
                st.markdown(f"**Example:** {selected_hypo.get('example_implementation', 'N/A')}")

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

    metrics = plan.get("metrics", [])
    if metrics:
        create_header_with_help("Metrics", "Primary and secondary metrics", icon="📏")
        
        try:
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

    risks = plan.get("risks_and_assumptions", [])
    if risks:
        create_header_with_help("Risks & Mitigations", "Potential issues and solutions", icon="⚠️")
        
        risk_text = []
        edited_risks = []
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

    next_steps = plan.get("next_steps", [
        "Finalize experiment design",
        "Implement tracking metrics",
        "Recruit users for testing"
    ])

    create_header_with_help("Next Steps", "Action items to execute the test", icon="✅")
    st.markdown("\n".join([f"- {step}" for step in next_steps]))
    with st.expander("Edit Next Steps"):
        edited_steps = st.text_area(
            "Next Steps (one per line)",
            value="\n".join(next_steps),
            key="editable_next_steps",
            height=120
        )

    if st.session_state.get("calculated_sample_size_per_variant") and st.session_state.get("calculated_total_sample_size"):
        st.markdown("---")
        create_header_with_help("Experiment Calculations", "Statistical requirements for valid results", icon="🧮")
        
        cols = st.columns(4)
        with cols[0]:
            st.metric("Current DAU", user_base_choice)
        with cols[1]:
            st.metric("Primary Metric", sanitized_metric_name)
        with cols[2]:
            st.metric("Current Value", formatted_current)
        with cols[3]:
            st.metric("Target Metric", formatted_target)
        
        st.markdown("---")
        
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

    create_header_with_help("Final PRD Preview", "Production-ready experiment document", icon="📄")
    
    def render_list(items):
        if not items or not any(str(item).strip() for item in items):
            return "None specified"
        html = "<ul>"
        for item in items:
            if isinstance(item, dict):
                item_text = f"<b>{item.get('name','Unnamed')}:</b> {item.get('formula','')}"
            else:
                item_text = str(item)
            if item_text:
                html += f"<li>{item_text}</li>"
        html += "</ul>"
        return html

    # --- Start of the updated PRD HTML section ---
    
    # Render Hypotheses (only the selected one)
    hypotheses_html = ""
    selected_index = st.session_state.get("selected_index")
    if selected_index is not None and len(prd_dict["hypotheses"]) > selected_index:
        selected_hypo = prd_dict["hypotheses"][selected_index]
        hypotheses_html = f"""
        <ol>
            <li>
                <p class="hypothesis-title">{selected_hypo.get('hypothesis', 'Hypothesis not available.')}</p>
                <p class="rationale"><i>Rationale:</i> {selected_hypo.get('rationale', '')}</p>
                <p class="example"><i>Example:</i> {selected_hypo.get('example_implementation', '')}</p>
            </li>
        </ol>
        """
    else:
        hypotheses_html = "<p>No hypothesis selected for the final PRD.</p>"

    # Render Risks
    risks_html = "<ul>"
    for r in prd_dict.get("risks_and_assumptions", []):
        severity_class = r.get('severity', 'Medium').lower()
        risks_html += f"""
        <li>
            <p>Potential risk: {r.get('risk', '')} <span class="severity {severity_class}">(Severity: {r.get('severity', 'Medium')})</span></p>
            <p>Mitigation: {r.get('mitigation', 'To be determined')}</p>
        </li>
        """
    risks_html += "</ul>"

    # Render Metric Calculations
    stats_html = ""
    stats = st.session_state.get("locked_stats")
    if stats:
        stats_html = f"""
        <div class="prd-section">
            <div class="prd-section-title">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2.5" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25m0-1.5h1.5a2.25 2.25 0 012.25 2.25V21m0 0l-3-3m3 3l3-3m-3 3v-3.75m-4.5-5.25H7.5m3-2.25H12m-3 5.25h.375M12 10.5h.375M12 10.5v3.75m9-8.25H15M3.75 9h16.5" />
                </svg>
                <h2>Experiment Calculations</h2>
            </div>
            <ul class="stats-list">
                <li>Confidence Level: {stats.get('confidence_level', 'N/A')}%</li>
                <li>Statistical Power: {stats.get('statistical_power', 'N/A')}%</li>
                <li>Minimum Detectable Effect (MDE): {stats.get('mde', 'N/A')}%</li>
                <li>Users per Variant: {stats.get('users_per_variant', 'N/A'):,}</li>
                <li>Total Sample Size: {stats.get('total_sample_size', 'N/A'):,}</li>
                <li>Estimated Test Duration: {f"{stats.get('estimated_test_duration_days', 'N/A'):,.0f} days" if isinstance(stats.get('estimated_test_duration_days'), (int, float)) else stats.get('estimated_test_duration_days', 'N/A')}</li>
            </ul>
        </div>
        """
    
    # Combine all sections into the final PRD HTML
    prd_html = f"""
    <div class="prd-card">
        <div class="prd-header">
            <div class="logo-wrapper">A/B</div>
            <div class="header-text">
                <h1>Experiment PRD</h1>
                <p>{prd_dict.get('goal', '')}</p>
            </div>
        </div>

        <div class="prd-section">
            <div class="prd-section-title">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2.5" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 01-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 013.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 013.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 01-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.455L14.25 6l1.035-.259a3.375 3.375 0 002.455-2.455L18 2.25l.259 1.035a3.375 3.375 0 002.455 2.455L21.75 6l-1.035.259a3.375 3.375 0 00-2.455 2.455zM19.5 17.25l-1.5.75L18 17.25m-1.5-.75l1.5.75V17.25l-1.5-.75L18 17.25z" />
                </svg>
                <h2>Problem Statement</h2>
            </div>
            <div class="prd-section-content problem-statement">{prd_dict.get('problem_statement', 'No problem statement provided.')}</div>
        </div>

        <div class="prd-section">
            <div class="prd-section-title">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2.5" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 12c0-1.232-.046-2.453-.138-3.662a4.004 4.004 0 00-3.7-3.7 48.678 48.678 0 00-7.324 0 4.005 4.005 0 00-3.7 3.7c-.017.22-.032.44-.046.662M19.5 12l3-3m-3 3l-3-3m-12 3c0 1.232.046 2.453.138 3.662a4.004 4.004 0 003.7 3.7 48.657 48.657 0 007.324 0 4.005 4.005 0 003.7-3.7c.017-.22.032-.44.046-.662M4.5 12l3 3m-3-3l-3 3" />
                </svg>
                <h2>Hypothesis</h2>
            </div>
            {hypotheses_html}
        </div>

        {stats_html}

        <div class="prd-section">
            <div class="prd-section-title">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2.5" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25m0-1.5h1.5a2.25 2.25 0 012.25 2.25V21m0 0l-3-3m3 3l3-3m-3 3v-3.75m-4.5-5.25H7.5m3-2.25H12m-3 5.25h.375M12 10.5h.375M12 10.5v3.75m9-8.25H15M3.75 9h16.5" />
                </svg>
                <h2>Metrics</h2>
            </div>
            <ul class="metrics">{render_list(prd_dict.get('metrics', []))}</ul>
        </div>
        
        <div class="prd-section">
            <div class="prd-section-title">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2.5" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m-9.303 12.337L1.5 21.75a3.75 3.75 0 005.303 5.303l1.5-1.5a3.75 3.75 0 005.303-5.303L12 9zM12 15.75h.007V15.75z" />
                </svg>
                <h2>Risks & Mitigations</h2>
            </div>
            {risks_html}
        </div>

        <div class="prd-section">
            <div class="prd-section-title">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2.5" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M12 6v12m-3-3h6M6 12h12" />
                </svg>
                <h2>Next Steps</h2>
            </div>
            <ul class="next-steps">{render_list(prd_dict.get('next_steps', []))}</ul>
        </div>

        <div class="prd-footer">
            Generated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}
        </div>
    </div>
    """
    st.markdown(prd_html, unsafe_allow_html=True)

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
            "editable_risks", "editable_next_steps", "raw_llm_edit"
        ]
        for k in keys_to_clear:
            if k in st.session_state:
                del st.session_state[k]
        st.success("Session cleared.")
