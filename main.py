# ============================================================
# main.py — A/B Test Architect (Merged, Refactored, Restored)
# Part 1/3 — Imports, Config, Constants, Helper Functions
# ============================================================

# ==========================
# Imports
# ==========================
import os
import re
import io
import json
import math
import time
import uuid
import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

# Statsmodels imports (used for robust sample size calc)
try:
    from statsmodels.stats.power import NormalIndPower, TTestIndPower
    from statsmodels.stats.proportion import proportion_effectsize
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

# Conditional ReportLab import for PDF generation
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# Try to import the original prompt engine if available (this preserves original behavior)
try:
    from prompt_engine import generate_experiment_plan  # original signature expected in many places
    PROMPT_ENGINE_AVAILABLE = True
except Exception:
    # keep a placeholder so app doesn't crash; later calls should handle fallback
    PROMPT_ENGINE_AVAILABLE = False

# ==========================
# Streamlit page config & CSS fallback
# ==========================
st.set_page_config(page_title="A/B Test Architect", layout="wide")

# Minimal inline CSS fallback (original had large CSS; we will preserve visual but keep succinct)
INLINE_CSS = """
<style>
.section-title {font-size:1.1rem; font-weight:700; color:#0b63c6;}
.small-muted { color:#7a7a7a; font-size:13px; }
.prd-card { background:#fff; border-radius:12px; padding:20px; border:1px solid #e5e7eb; }
</style>
"""
st.markdown(INLINE_CSS, unsafe_allow_html=True)

# ==========================
# Constants & defaults
# ==========================
APP_VERSION = "refactor-merged-1.0"
DAU_MAP = {"< 10K": 5000, "10K–100K": 50000, "100K–1M": 500000, "> 1M": 2000000}
UNITS_WITH_SPACE = ["USD", "count", "minutes", "hours", "days", "INR"]
DEFAULT_MDE = 5.0

# ==========================
# Helper UI bits
# ==========================
def create_header_with_help(header_text: str, help_text: str, icon: str = "🔗"):
    """Reusable header block with help tooltip (keeps original style)."""
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="font-size: 1.4rem;">{icon}</div>
                <div class="section-title" style="margin-bottom: 0;">{header_text}</div>
            </div>
            <span style="font-size: 0.95rem; color: #666; cursor: help; float: right;" title="{html_sanitize(help_text)}">❓</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==========================
# Text utilities (restored + improved)
# ==========================
def sanitize_text(text: Any) -> str:
    """Return a safe single-line string for display."""
    if text is None:
        return ""
    try:
        s = str(text)
    except Exception:
        return ""
    s = s.replace("\r", " ").replace("\t", " ")
    s = re.sub(r"[ \f\v]+", " ", s)
    return s.strip()

def html_sanitize(text: Any) -> str:
    """Escape suspicious HTML characters for attributes/tooltips."""
    if text is None:
        return ""
    t = str(text)
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = t.replace('"', "&quot;").replace("'", "&apos;")
    return t

def safe_display(text: Any, method=st.info):
    """Convenience wrapper to show sanitized text in UI."""
    method(sanitize_text(text))

# ==========================
# JSON extraction and parsing helpers (restored)
# ==========================
def _extract_json_first_braces(text: str) -> Optional[str]:
    """
    Extracts the first balanced JSON object starting at the first '{' encountered.
    This helps with LLM outputs that append text before or after a JSON block.
    """
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
    """
    Heuristic replacement of single quotes with double quotes for JSON-like strings.
    Not perfect, but helps with common LLM formatting issues.
    """
    try:
        # convert trailing/leading single-quoted keys and simple single-quoted values
        s = re.sub(r"(?<=[:\{\[,]\s*)'([^']*?)'(?=\s*[,}\]])", r'"\1"', s)
        s = re.sub(r"'([A-Za-z0-9_ \-]+?)'\s*:", r'"\1":', s)
        return s
    except Exception:
        return s

def extract_json(text: Any) -> Optional[Dict]:
    """
    Robust JSON extraction that tries multiple strategies:
    - If already a dict/list, return normalized dict
    - Try json.loads on the entire string
    - Try ast.literal_eval
    - Try extracting the first JSON brace block and parsing
    - Try heuristic single->double quote conversion
    On failure, show the raw snippet to the user (UI layer should show st.code)
    """
    import ast
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

    # Attempt direct JSON parse
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

    # Try ast literal eval (accepts single quotes)
    try:
        parsed_ast = ast.literal_eval(raw)
        if isinstance(parsed_ast, dict):
            return parsed_ast
        if isinstance(parsed_ast, list) and all(isinstance(i, dict) for i in parsed_ast):
            return {"items": parsed_ast}
    except Exception:
        pass

    # Try to extract json block
    candidate = _extract_json_first_braces(raw)
    if candidate:
        candidate_clean = candidate
        # strip code fences
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
            # try ast literal eval on candidate
            try:
                parsed_ast = ast.literal_eval(candidate_clean)
                if isinstance(parsed_ast, dict):
                    return parsed_ast
                if isinstance(parsed_ast, list) and all(isinstance(i, dict) for i in parsed_ast):
                    return {"items": parsed_ast}
            except Exception:
                # try converting single quotes to double quotes
                try:
                    converted = _safe_single_to_double_quotes(candidate_clean)
                    parsed = json.loads(converted)
                    if isinstance(parsed, dict):
                        return parsed
                    if isinstance(parsed, list) and all(isinstance(i, dict) for i in parsed):
                        return {"items": parsed}
                    else:
                        st.error("Extracted JSON with single quotes parsed but was not an object.")
                except Exception:
                    st.error("Could not parse extracted JSON block. See snippet below.")
                    st.code(candidate_clean[:3000] + ("..." if len(candidate_clean) > 3000 else ""))
                    return None

    # Try cleaning entire output with single->double conversion
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

# ==========================
# Post processing helpers (restored)
# ==========================
def post_process_llm_text(text: Any, unit: str) -> str:
    """Small textual cleanups tailored to units (e.g., '%' double escaping)."""
    if text is None:
        return ""
    s = sanitize_text(text)
    if unit == "%":
        s = s.replace("%%", "%")
        s = re.sub(r"\s+%", "%", s)
    return s

def _parse_value_from_text(text: str, default_unit: str = '%') -> Tuple[Optional[float], str]:
    """
    Parse a numeric value and optional unit from a free-text input.
    Returns (value, unit). Value may be None.
    """
    text = sanitize_text(text)
    match = re.match(r"([\d\.]+)\s*(\w+|%)?", text)
    if match:
        try:
            value = float(match.group(1))
        except Exception:
            return None, default_unit
        unit = match.group(2) if match.group(2) else default_unit
        return value, unit
    try:
        return float(text), default_unit
    except Exception:
        return None, default_unit

# ==========================
# Sample size calculation (robust restored version)
# ==========================
def calculate_sample_size(baseline, mde, alpha, power, num_variants, metric_type, std_dev=None) -> Tuple[Optional[int], Optional[int]]:
    """
    Returns (per_variant, total) sample sizes using statsmodels when available.
    - baseline: if conversion rate, expected baseline percentage (e.g., 5.0 for 5%)
    - mde: percent change to detect (e.g., 10 for 10%)
    - alpha: significance level (note: NOT percent, e.g., 0.05)
    - power: power (as decimal, e.g., 0.8)
    - num_variants: number of groups including control
    - metric_type: 'Conversion Rate' or 'Numeric Value'
    - std_dev: required for numeric metrics
    """
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
            if STATSMODELS_AVAILABLE:
                effect_size = proportion_effectsize(baseline_prop, expected_prop)
                if effect_size == 0:
                    return None, None
                analysis = NormalIndPower()
                sample_size_per_variant = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative="two-sided")
            else:
                # fallback z-test approximate
                pooled = (baseline_prop + expected_prop) / 2
                z_alpha = 1.96  # conservative for 0.05
                z_beta = 0.84   # conservative for 0.8
                delta = abs(expected_prop - baseline_prop)
                sample_size_per_variant = (2 * (z_alpha + z_beta) ** 2 * pooled * (1 - pooled)) / (delta ** 2)
        elif metric_type == "Numeric Value":
            try:
                std_dev_val = float(std_dev) if std_dev is not None else None
            except Exception:
                std_dev_val = None
            if std_dev_val is None or std_dev_val == 0:
                return None, None
            mde_absolute = float(baseline) * mde_relative
            if STATSMODELS_AVAILABLE:
                effect_size = mde_absolute / std_dev_val
                if effect_size == 0:
                    return None, None
                analysis = TTestIndPower()
                sample_size_per_variant = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative="two-sided")
            else:
                # fallback manual calculation using z approximations
                z_alpha = 1.96
                z_beta = 0.84
                sample_size_per_variant = (2 * (z_alpha + z_beta) ** 2 * (std_dev_val ** 2)) / (mde_absolute ** 2)
        else:
            return None, None

        if sample_size_per_variant is None or sample_size_per_variant <= 0 or not np.isfinite(sample_size_per_variant):
            return None, None

        total = sample_size_per_variant * num_variants
        return int(math.ceil(sample_size_per_variant)), int(math.ceil(total))
    except Exception:
        return None, None

# ==========================
# PDF Export (restored & hardened)
# ==========================
def generate_pdf_bytes_from_prd_dict(prd: Dict, title: str = "Experiment PRD") -> Optional[bytes]:
    """
    Create a simple PDF representation of the PRD dictionary using ReportLab.
    If reportlab is not installed, return None.
    """
    if not REPORTLAB_AVAILABLE:
        return None
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="PRDTitle", fontSize=18, leading=22, spaceAfter=12, alignment=1))
        styles.add(ParagraphStyle(name="SectionHeading", fontSize=12, leading=14, spaceBefore=8, spaceAfter=6))
        styles.add(ParagraphStyle(name="BodyTextCustom", fontSize=10.5, leading=12))
        styles.add(ParagraphStyle(name="BulletText", fontSize=10.5, leading=12, leftIndent=12, bulletIndent=6))

        story: List[Any] = []
        story.append(Paragraph(title, styles["PRDTitle"]))
        story.append(Spacer(1, 6))

        def pdf_sanitize(text: Any) -> str:
            if text is None:
                return ""
            s = str(text)
            s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            return s

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

        # Build story from prd dict keys in a stable order
        keys_order = ["goal", "problem_statement", "hypotheses", "metrics", "segments", "success_criteria", "risks_and_assumptions", "next_steps", "statistical_rationale"]
        for k in keys_order:
            if k in prd:
                add_paragraph(k.replace("_", " ").title(), prd[k])
        # add any other keys
        for k, v in prd.items():
            if k not in keys_order:
                add_paragraph(k.replace("_", " ").title(), v)

        doc.build(story)
        buffer.seek(0)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes
    except Exception:
        return None

# ==========================
# Small utility: format with unit (restored original behavior)
# ==========================
def format_value_with_unit_original(value: Any, unit: str) -> str:
    """
    Keep original formatting semantics where integer floats are shown without decimals
    and certain units have a space.
    """
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
    if unit in UNITS_WITH_SPACE:
        return f"{v_str} {unit}"
    else:
        return f"{v_str}{unit}"

# Alias to keep original code references working
format_value_with_unit = format_value_with_unit_original

# ==========================
# End of Part 1/3 helpers
# ==========================
# ============================================================
# Part 2/3 — State Initialization, App Header, Product Context,
# Metric Objective, Plan Generation, Hypotheses & Editors
# ============================================================

# ==========================
# Session state defaults (restore original keys)
# ==========================
defaults = {
    "output": "",
    "ai_parsed": None,
    "calc_locked": False,
    "locked_stats": {},
    "selected_index": None,
    "hypothesis_confirmed": False,
    "last_llm_hash": None,
    "calculate_now": False,
    "metrics_table": None,
    "raw_llm_edit": "",
    "editable_problem": "",
    "editable_hypothesis": "",
    "editable_rationale": "",
    "editable_example": "",
    "editable_segments": "",
    "editable_risks": "",
    "editable_next_steps": "",
    "calculated_sample_size_per_variant": None,
    "calculated_total_sample_size": None,
    "calculated_duration_days": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==========================
# App header + description (restored)
# ==========================
st.title("💡 A/B Test Architect — AI-assisted experiment PRD generator")
st.markdown("Create experiment PRDs, hypotheses, stats, and sample-size guidance — faster and with guardrails.")

# Top-level product-context expander (restored)
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

# ==========================
# Metric Improvement Objective (restored)
# ==========================
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

# ==========================
# Generate Experiment Plan (restored)
# ==========================
with st.expander("🧠 Generate Experiment Plan", expanded=True):
    create_header_with_help("Generate Experiment Plan", "When ready, click Generate to call the LLM and create a plan.", icon="🧠")
    sanitized_metric_name = sanitize_text(exact_metric) if exact_metric else ""
    
    try:
        if current_value is not None and current_value != 0:
            expected_lift_val = round(((target_value - current_value) / current_value) * 100, 2)
            mde_default = round(abs((target_value - current_value) / current_value) * 100, 2)
        else:
            expected_lift_val = 0.0
            mde_default = DEFAULT_MDE
    except Exception:
        expected_lift_val = 0.0
        mde_default = DEFAULT_MDE

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
                # Use prompt_engine.generate_experiment_plan if available (restored behavior)
                if PROMPT_ENGINE_AVAILABLE:
                    raw_llm = generate_experiment_plan(goal_with_units, context)
                else:
                    # graceful fallback: return a simple placeholder JSON to avoid crashes
                    raw_llm = json.dumps({
                        "problem_statement": f"OPPORTUNITY: Improve {sanitized_metric_name}.",
                        "hypotheses": [
                            {"hypothesis": f"If we make a change to improve {sanitized_metric_name}, then it will increase.", "rationale": "", "example_implementation": "", "behavioral_basis": ""}
                        ],
                        "metrics": [{"name": sanitized_metric_name, "formula": "", "importance": "High"}],
                        "next_steps": ["Draft mockups (Design)", "Instrument metrics (Engineering)"],
                        "success_criteria": {"MDE": mde_default}
                    })
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

# ==========================
# Raw LLM editor if parsing fails (restored)
# ==========================
if st.session_state.get("ai_parsed") is None and st.session_state.get("output"):
    st.markdown("<div class='section-title'>Raw LLM Output (fix JSON here)</div>", unsafe_allow_html=True)
    raw_edit = st.text_area("Raw LLM output / edit here", value=st.session_state.get("raw_llm_edit", ""), height=400, key="raw_llm_edit")
    if st.button("Parse JSON"):
        parsed_try = extract_json(raw_edit)
        if parsed_try:
            st.session_state.ai_parsed = parsed_try
            st.success("Manual parse succeeded — plan is now usable.")
        else:
            st.error("Manual parse failed — edit the text and try again.")

# ==========================
# Parsed plan display (Hypotheses, Segments, Risks, Next steps) (restored)
# ==========================
if st.session_state.get("ai_parsed"):
    plan = st.session_state.ai_parsed

    # Inferred Product Goal
    st.markdown("### Inferred Product Goal")
    safe_display(post_process_llm_text(goal_with_units, metric_unit))

    # Problem Statement (auto-insert metrics if missing)
    st.markdown("### Problem Statement")
    problem_statement = plan.get("problem_statement", "")
    # If plan has a problem statement, prefer that; else build one
    if not problem_statement:
        problem_statement = f"OPPORTUNITY: Move {sanitized_metric_name} from {formatted_current} to {formatted_target} — {sanitize_text(strategic_goal)}"
    # Auto-insert metric numbers when possible
    def generate_problem_statement_local(plan_obj, current, target, unit):
        if not plan_obj:
            return ""
        base = plan_obj.get("problem_statement", "") or ""
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
    problem_statement = generate_problem_statement_local(plan, current_value, target_value, metric_unit)
    st.markdown(problem_statement or "⚠️ Problem statement not generated by the model.")

    # Hypotheses list with Select/Edit buttons
    st.markdown("### Hypotheses")
    hypotheses = plan.get("hypotheses", []) or []
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

    # If selected, show edit UI for selected hypothesis (restored)
    if st.session_state.get("hypothesis_confirmed") and st.session_state.get("selected_index") is not None:
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
                    # write back into session plan
                    st.session_state.ai_parsed["hypotheses"] = hypotheses
                    st.success("Hypothesis updated!")

        with cols[1]:
            if st.button("Clear Selection"):
                st.session_state.selected_index = None
                st.session_state.hypothesis_confirmed = False

    # Metrics display (brief) — detailed editor below
    metrics = plan.get("metrics", [])
    if metrics:
        st.markdown("### Metrics")
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
            st.table(pd.DataFrame(normalized))
        except Exception as e:
            st.error(f"Metrics display error: {e}")
            st.json(metrics)

    # Segments editor (restored)
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

    # Risks editor (restored)
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

    # Next steps editor (restored)
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
# ============================================================
# Part 3/3 — Metrics Editor, Calculator, PRD Preview, Exports
# ============================================================

# Metrics Editor — allows adding/removing/editing metrics
if st.session_state.get("ai_parsed"):
    st.subheader("📏 Metrics Editor")
    metrics_list = st.session_state.ai_parsed.get("metrics", [])
    if not isinstance(metrics_list, list):
        metrics_list = []

    for i, m in enumerate(metrics_list):
        cols = st.columns([3, 3, 2, 1])
        with cols[0]:
            name = st.text_input(
                f"Metric Name {i+1}",
                value=m.get("name", ""),
                key=f"metric_name_{i}"
            )
        with cols[1]:
            formula = st.text_input(
                f"Formula {i+1}",
                value=m.get("formula", ""),
                key=f"metric_formula_{i}"
            )
        with cols[2]:
            importance = st.selectbox(
                f"Importance {i+1}",
                ["High", "Medium", "Low"],
                index=["High", "Medium", "Low"].index(m.get("importance", "Medium")),
                key=f"metric_importance_{i}"
            )
        with cols[3]:
            if st.button("❌", key=f"remove_metric_{i}"):
                metrics_list.pop(i)
                st.session_state.ai_parsed["metrics"] = metrics_list
                st.experimental_rerun()

        metrics_list[i] = {"name": name, "formula": formula, "importance": importance}

    if st.button("➕ Add Metric"):
        metrics_list.append({"name": "", "formula": "", "importance": "Medium"})

    st.session_state.ai_parsed["metrics"] = metrics_list

# ==========================
# Sample size & duration calculator (restored)
# ==========================
st.subheader("📊 Sample Size Calculator")
if st.session_state.get("ai_parsed") and st.session_state.get("hypothesis_confirmed"):
    if metric_type == "Conversion Rate":
        calc_result = calculate_sample_size(
            current_rate=current_value,
            target_rate=target_value,
            mde=(target_value - current_value) / current_value * 100 if current_value else DEFAULT_MDE,
            alpha=0.05,
            power=0.8,
            std_dev=None
        )
    else:
        calc_result = calculate_sample_size(
            current_rate=None,
            target_rate=None,
            mde=(target_value - current_value) / current_value * 100 if current_value else DEFAULT_MDE,
            alpha=0.05,
            power=0.8,
            std_dev=std_dev
        )

    if calc_result:
        st.session_state.calculated_sample_size_per_variant = calc_result["sample_size_per_variant"]
        st.session_state.calculated_total_sample_size = calc_result["total_sample_size"]
        st.session_state.calculated_duration_days = calc_result["duration_days"]

        st.markdown(f"**Sample size per variant:** {calc_result['sample_size_per_variant']}")
        st.markdown(f"**Total sample size:** {calc_result['total_sample_size']}")
        st.markdown(f"**Estimated duration (days):** {calc_result['duration_days']}")

# ==========================
# PRD Preview (restored)
# ==========================
st.subheader("📄 Experiment PRD Preview")
if st.session_state.get("ai_parsed"):
    plan = st.session_state.ai_parsed
    prd_markdown = f"""
# Experiment PRD

## Problem Statement
{plan.get('problem_statement', '')}

## Hypotheses
"""
    for h in plan.get("hypotheses", []):
        prd_markdown += f"- **Hypothesis:** {h.get('hypothesis', '')}\n"
        prd_markdown += f"  - Rationale: {h.get('rationale', '')}\n"
        prd_markdown += f"  - Example Implementation: {h.get('example_implementation', '')}\n"

    prd_markdown += "\n## Metrics\n"
    for m in plan.get("metrics", []):
        prd_markdown += f"- {m.get('name', '')} ({m.get('importance', 'Medium')})\n"

    if plan.get("segments"):
        prd_markdown += "\n## Segments\n"
        for s in plan["segments"]:
            prd_markdown += f"- {s}\n"

    if plan.get("risks_and_assumptions"):
        prd_markdown += "\n## Risks & Assumptions\n"
        for r in plan["risks_and_assumptions"]:
            if isinstance(r, dict):
                prd_markdown += f"- {r.get('risk', '')} (Severity: {r.get('severity', 'Medium')})\n"
                prd_markdown += f"  - Mitigation: {r.get('mitigation', '')}\n"
            else:
                prd_markdown += f"- {str(r)}\n"

    if plan.get("next_steps"):
        prd_markdown += "\n## Next Steps\n"
        for step in plan["next_steps"]:
            prd_markdown += f"- {step}\n"

    # Display PRD
    st.markdown(prd_markdown)

# ==========================
# Export buttons (restored)
# ==========================
col_export_pdf, col_export_md = st.columns([1, 1])
with col_export_pdf:
    if st.button("📥 Export as PDF"):
        try:
            pdf_bytes = export_prd_as_pdf(st.session_state.ai_parsed)
            st.download_button("Download PDF", data=pdf_bytes, file_name="experiment_prd.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"PDF export failed: {e}")
with col_export_md:
    if st.button("📥 Export as Markdown"):
        try:
            md_content = export_prd_as_md(st.session_state.ai_parsed)
            st.download_button("Download Markdown", data=md_content.encode("utf-8"), file_name="experiment_prd.md", mime="text/markdown")
        except Exception as e:
            st.error(f"Markdown export failed: {e}")

# ==========================
# Footer
# ==========================
st.markdown("---")
st.caption("A/B Test Architect — Built with ❤️ to make experiments faster and smarter.")
