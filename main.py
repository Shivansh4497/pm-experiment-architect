# main.py — Full updated app with improved UX, hypothesis enrichment, tabbed PRD preview/edit,
# per-section regeneration, and a floating tips panel (manual refresh).

import json
import math
import random
import string
import textwrap
from io import BytesIO
from typing import Any, Dict, List, Optional

import streamlit as st

# --- NEW: Import scipy for statistical calculations and add availability check ---
try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Try optional libs for exports
PDF_AVAILABLE = False
DOCX_AVAILABLE = False
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    # Warning is now handled contextually when the user tries to export

try:
    from docx import Document
    from docx.shared import Pt
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    # Warning is now handled contextually when the user tries to export

# Import prompt engine functions (best-effort)
PROMPT_ENGINE_AVAILABLE = False
_generate_hypotheses = None
_generate_hypothesis_details = None
_generate_experiment_plan = None
_validate_experiment_plan = None
_generate_dynamic_tips = None

try:
    from prompt_engine import (
        generate_hypotheses as _generate_hypotheses,
        expand_hypothesis_with_details as _generate_hypothesis_details,
        generate_experiment_plan as _generate_experiment_plan,
        generate_tips as _generate_dynamic_tips,
        _client,
    )
    PROMPT_ENGINE_AVAILABLE = True
except ImportError as e:
    # This will only show a warning if the file is expected but fails, not if it's missing.
    pass

# -------------------------
# Styling and UI helpers
# -------------------------
def inject_global_css():
    css = r"""
    <style>
    /* Page container */
    .block-container { max-width: 1200px; padding-top: 1rem; }

    /* Header gradient */
    header [role="banner"] { background: linear-gradient(90deg, #0ea5e9 0%, #6366f1 100%) !important; color: white !important; }

    /* Buttons and cards */
    .stButton > button { border-radius: 10px; padding: 0.5rem 0.75rem; font-weight: 600; }
    .card { border:1px solid #e6eef8; border-radius:12px; padding:1rem; background:linear-gradient(180deg,#ffffff,#fbfdff); box-shadow: 0 1px 3px rgba(15,23,42,0.04); }
    .muted { color: #6b7280; font-size:0.9rem; }

    /* Floating tips button */
    .floating-tips-button {
        position: fixed;
        right: 22px;
        bottom: 22px;
        z-index: 9999;
    }
    .floating-tips-panel {
        position: fixed;
        right: 22px;
        bottom: 80px;
        z-index: 9998;
        width: 360px;
        max-height: 70vh;
        overflow: auto;
        background: #ffffff;
        border-radius: 12px;
        box-shadow: 0 8px 30px rgba(2,6,23,0.2);
        border: 1px solid #e6eef8;
        padding: 16px;
    }
    .tip-card {
        background: #f8fafc;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 8px;
        font-size: 0.95rem;
    }

    /* Tab styling - make tabs look pill-like */
    .css-1avcm0n > div[role="tablist"] button {
        padding: 0.5rem 0.9rem;
        border-radius: 999px;
        margin-right: 6px;
    }
    
    /* --- NEW: Style for calculator results --- */
    .stAlert[data-baseweb="alert"] > div:first-child {
         background-color: #e0f2fe !important;
         border-left: 5px solid #0ea5e9 !important;
         color: #0c4a6e !important;
         border-radius: 8px;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# -------------------------
# Utility functions
# -------------------------
def sanitize_text(x: Any) -> str:
    """Convert any input to clean string, handling None and collections."""
    if x is None:
        return ""
    if isinstance(x, (dict, list)):
        try:
            return json.dumps(x, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(x)
    return str(x).strip()

def ensure_list(x: Any) -> List[Any]:
    """Ensure output is always a list, converting single items if needed."""
    if x is None:
        return []
    return x if isinstance(x, list) else [x]

def ensure_dict(x: Any) -> Dict[str, Any]:
    """Ensure output is always a dict, converting if needed."""
    return x if isinstance(x, dict) else {}

def safe_int(x: Any, default: int = 0) -> int:
    """Safely convert to int with fallback."""
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return int(x)
        s = str(x).strip()
        return int(float(s)) if s else default
    except (ValueError, TypeError):
        return default

def safe_float(x: Any, default: float = 0.0) -> float:
    """Safely convert to float with fallback."""
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace("%", "")
        return float(s) if s else default
    except (ValueError, TypeError):
        return default

def extract_json_from_text(text: Optional[str]) -> Dict[str, Any]:
    """Robust JSON extraction from potentially messy LLM output."""
    if not text:
        return {}
    
    text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass
    
    # Try array format
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start:end+1])
            return {"list": parsed} if isinstance(parsed, list) else {}
        except json.JSONDecodeError:
            pass
    
    return {}

def generate_experiment_id(prefix: str = "EXP") -> str:
    """Generate random experiment ID with prefix."""
    rand = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
    return f"{prefix}-{rand}"

# --- NEW: Statistical Calculator Function ---
def calculate_sample_size(baseline_rate, mde, alpha=0.05, power=0.8):
    """
    Calculate the required sample size per variant for a two-tailed test.
    Returns None if inputs are invalid.
    """
    if not SCIPY_AVAILABLE:
        return None
    if not (0 < baseline_rate < 1 and mde > 0):
        return None

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    p1 = baseline_rate
    p2 = baseline_rate * (1 + mde)

    if p2 >= 1:
        return None

    p_avg = (p1 + p2) / 2
    
    numerator = (z_alpha * math.sqrt(2 * p_avg * (1 - p_avg))) + (z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))
    denominator = p2 - p1
    
    if denominator == 0:
        return None

    return math.ceil((numerator / denominator) ** 2)

# -------------------------
# Export helpers (PDF/DOCX/JSON)
# -------------------------
def pdf_safe(text: str) -> str:
    """Make text safe for PDF generation by replacing tabs."""
    return text.replace("\t", "    ")

def generate_pdf_bytes_from_prd_dict(plan: Dict[str, Any]) -> Optional[bytes]:
    """Generate PDF bytes from PRD dict using ReportLab."""
    if not PDF_AVAILABLE:
        return None

    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        def add_heading(txt: str, level: int = 1) -> None:
            style = styles["Heading1"] if level == 1 else styles["Heading2"]
            story.append(Paragraph(pdf_safe(txt), style))
            story.append(Spacer(1, 8))

        def add_paragraph(txt: str) -> None:
            story.append(Paragraph(pdf_safe(txt), styles["BodyText"]))
            story.append(Spacer(1, 6))

        # Metadata
        meta = plan.get("metadata", {})
        title = meta.get("title", "Experiment PRD")
        add_heading(title, level=1)
        add_paragraph(f"<b>Experiment ID:</b> {meta.get('experiment_id','')}")
        add_paragraph(f"<b>Team:</b> {meta.get('team','')}")
        add_paragraph(f"<b>Owner:</b> {meta.get('owner','')}")

        # Problem statement
        add_heading("Problem Statement", level=2)
        add_paragraph(plan.get("problem_statement", ""))

        # Hypotheses
        hyps = ensure_list(plan.get("hypotheses"))
        for i, h in enumerate(hyps, 1):
            add_heading(f"Hypothesis {i}", level=2)
            add_paragraph(f"<b>Hypothesis:</b> {h.get('hypothesis','')}")
            if h.get("rationale"):
                add_paragraph(f"<b>Rationale:</b> {h.get('rationale')}")
            if h.get("example_implementation"):
                add_paragraph(f"<b>Example Implementation:</b> {h.get('example_implementation')}")
            if h.get("behavioral_basis"):
                add_paragraph(f"<b>Behavioral Basis:</b> {h.get('behavioral_basis')}")

        # Proposed solution & variants
        add_heading("Proposed Solution & Variants", level=2)
        add_paragraph(plan.get("proposed_solution", ""))
        vars_table = [["Control", "Variation", "Notes"]]
        for v in ensure_list(plan.get("variants")):
            vars_table.append([v.get("control", ""), v.get("variation", ""), v.get("notes", "")])
        
        t = Table(vars_table, colWidths=[150, 150, 150])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]))
        story.append(t)
        story.append(Spacer(1, 8))

        # Metrics
        metrics_table = [["Metric", "Formula", "Importance"]]
        for m in ensure_list(plan.get("metrics")):
            metrics_table.append([m.get("name", ""), m.get("formula", ""), m.get("importance", "")])
        
        t2 = Table(metrics_table, colWidths=[150, 220, 80])
        t2.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.grey)]))
        story.append(t2)
        story.append(Spacer(1, 8))

        # Risks & other sections
        add_heading("Risks & Mitigation", level=2)
        for r in ensure_list(plan.get("risks_and_assumptions")):
            add_paragraph(f"{r.get('risk','')} — {r.get('mitigation','')} (Severity: {r.get('severity','')})")

        add_heading("Experiment Design & Rollout Plan", level=2)
        ed = ensure_dict(plan.get("experiment_design", {}))
        add_paragraph(f"Traffic Allocation: {ed.get('traffic_allocation','')}")
        add_paragraph(f"Sample size / variant: {ed.get('sample_size_per_variant','')}")
        add_paragraph(f"Total sample size: {ed.get('total_sample_size','')}")
        add_paragraph(f"Estimated Duration (days): {ed.get('test_duration_days','')}")

        # Success & Learning
        add_heading("Success & Learning Criteria", level=2)
        sc = ensure_dict(plan.get("success_criteria", {}))
        add_paragraph(f"Confidence level: {sc.get('confidence_level','')}%")
        add_paragraph(f"Power: {sc.get('power','')}%")
        add_paragraph(f"MDE: {sc.get('MDE','')}%")

        # Statistical rationale
        if plan.get("statistical_rationale"):
            add_heading("Statistical Rationale", level=2)
            add_paragraph(plan.get("statistical_rationale"))

        doc.build(story)
        buffer.seek(0)
        return buffer.read()
    except Exception as e:
        st.error(f"PDF generation failed: {str(e)}")
        return None

def generate_docx_bytes_from_plan(plan: Dict[str, Any]) -> Optional[bytes]:
    """Generate DOCX bytes from PRD dict using python-docx."""
    if not DOCX_AVAILABLE:
        return None

    try:
        doc = Document()
        meta = plan.get("metadata", {})
        doc.add_heading(meta.get("title", "Experiment PRD"), level=0)
        doc.add_paragraph(f"Experiment ID: {meta.get('experiment_id','')}")
        doc.add_paragraph(f"Team: {meta.get('team','')}")
        doc.add_paragraph(f"Owner: {meta.get('owner','')}")

        doc.add_heading("Problem Statement", level=1)
        doc.add_paragraph(plan.get("problem_statement", ""))

        hyps = ensure_list(plan.get("hypotheses"))
        for i, h in enumerate(hyps, 1):
            doc.add_heading(f"Hypothesis {i}", level=1)
            doc.add_paragraph(h.get("hypothesis", ""))
            if h.get("rationale"):
                doc.add_paragraph("Rationale: " + h.get("rationale", ""))
            if h.get("example_implementation"):
                doc.add_paragraph("Example Implementation: " + h.get("example_implementation", ""))

        doc.add_heading("Proposed Solution & Variants", level=1)
        doc.add_paragraph(plan.get("proposed_solution", ""))

        # Finish file
        buffer = BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        return buffer.read()
    except Exception as e:
        st.error(f"DOCX generation failed: {str(e)}")
        return None

# -------------------------
# Plan normalization / defaults
# -------------------------
DEFAULT_PLAN: Dict[str, Any] = {
    "metadata": {
        "title": "Untitled Experiment", 
        "team": "", 
        "owner": "", 
        "experiment_id": ""
    },
    "problem_statement": "",
    "hypotheses": [],
    "proposed_solution": "",
    "variants": [{"control": "", "variation": "", "notes": ""}],
    "metrics": [],
    "guardrail_metrics": [],
    "experiment_design": {
        "traffic_allocation": "50/50",
        "sample_size_per_variant": 0,
        "total_sample_size": 0,
        "test_duration_days": 0,
        "dau_coverage_percent": 0.0,
        "power": 80.0,
        # --- NEW: Add calculator fields to default plan ---
        "baseline_conversion_rate": 10.0,
        "mde_percent": 5.0,
        "daily_traffic_for_experiment": 1000,
    },
    "success_criteria": {
        "confidence_level": 95.0, 
        "power": 80.0, 
        "MDE": 1.0, 
        "benchmark": "", 
        "monitoring": ""
    },
    "success_learning_criteria": {
        "definition_of_success": "", 
        "stopping_rules": "", 
        "rollback_criteria": ""
    },
    "risks_and_assumptions": [],
    "statistical_rationale": "",
}

def sanitize_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize PRD structure, fill defaults, and coerce types."""
    if not isinstance(plan, dict):
        plan = {}
    
    # Start with deep copy of defaults
    merged = json.loads(json.dumps(DEFAULT_PLAN))

    # metadata
    merged["metadata"].update(ensure_dict(plan.get("metadata", {})))
    if not merged["metadata"].get("experiment_id"):
        merged["metadata"]["experiment_id"] = generate_experiment_id()

    # scalars
    for k in ["problem_statement", "proposed_solution", "statistical_rationale"]:
        if k in plan:
            merged[k] = sanitize_text(plan.get(k))

    # hypotheses
    merged["hypotheses"] = []
    for h in ensure_list(plan.get("hypotheses")):
        if isinstance(h, dict):
            merged["hypotheses"].append({
                "hypothesis": sanitize_text(h.get("hypothesis", "")),
                "rationale": sanitize_text(h.get("rationale", "")),
                "example_implementation": sanitize_text(h.get("example_implementation", "")),
                "behavioral_basis": sanitize_text(h.get("behavioral_basis", ""))
            })

    # variants
    merged["variants"] = []
    for v in ensure_list(plan.get("variants")):
        if isinstance(v, dict):
            merged["variants"].append({
                "control": sanitize_text(v.get("control", "")),
                "variation": sanitize_text(v.get("variation", "")),
                "notes": sanitize_text(v.get("notes", ""))
            })

    # metrics
    merged["metrics"] = []
    for m in ensure_list(plan.get("metrics")):
        if isinstance(m, dict):
            importance = m.get("importance", "Primary")
            if importance not in ("Primary", "Secondary"):
                importance = "Primary"
            merged["metrics"].append({
                "name": sanitize_text(m.get("name", "")),
                "formula": sanitize_text(m.get("formula", "")),
                "importance": importance
            })

    # guardrails
    merged["guardrail_metrics"] = []
    for g in ensure_list(plan.get("guardrail_metrics")):
        if isinstance(g, dict):
            direction = g.get("direction", "Decrease")
            if direction not in ("Increase", "Decrease", "No Change"):
                direction = "Decrease"
            merged["guardrail_metrics"].append({
                "name": sanitize_text(g.get("name", "")),
                "direction": direction,
                "threshold": sanitize_text(g.get("threshold", ""))
            })

    # experiment design
    ed = ensure_dict(plan.get("experiment_design", {}))
    merged["experiment_design"] = {
        "traffic_allocation": sanitize_text(ed.get("traffic_allocation", "50/50")),
        "sample_size_per_variant": safe_int(ed.get("sample_size_per_variant", 0)),
        "total_sample_size": safe_int(ed.get("total_sample_size", 0)),
        "test_duration_days": safe_int(ed.get("test_duration_days", 0)),
        "dau_coverage_percent": safe_float(ed.get("dau_coverage_percent", 0.0)),
        "power": safe_float(ed.get("power", 80.0)),
        # --- NEW: Ensure calculator fields are sanitized ---
        "baseline_conversion_rate": safe_float(ed.get("baseline_conversion_rate", 10.0)),
        "mde_percent": safe_float(ed.get("mde_percent", 5.0)),
        "daily_traffic_for_experiment": safe_int(ed.get("daily_traffic_for_experiment", 1000)),
    }

    # success criteria
    sc = ensure_dict(plan.get("success_criteria", {}))
    merged["success_criteria"] = {
        "confidence_level": safe_float(sc.get("confidence_level", 95.0)),
        "power": safe_float(sc.get("power", merged["experiment_design"]["power"])),
        "MDE": safe_float(sc.get("MDE", 1.0)),
        "benchmark": sanitize_text(sc.get("benchmark", "")),
        "monitoring": sanitize_text(sc.get("monitoring", "")),
    }

    # success learning criteria
    sl = ensure_dict(plan.get("success_learning_criteria", {}))
    merged["success_learning_criteria"] = {
        "definition_of_success": sanitize_text(sl.get("definition_of_success", "")),
        "stopping_rules": sanitize_text(sl.get("stopping_rules", "")),
        "rollback_criteria": sanitize_text(sl.get("rollback_criteria", "")),
    }

    # risks
    merged["risks_and_assumptions"] = []
    for r in ensure_list(plan.get("risks_and_assumptions")):
        if isinstance(r, dict):
            sev = r.get("severity", "Medium")
            if sev not in ("High", "Medium", "Low"):
                sev = "Medium"
            merged["risks_and_assumptions"].append({
                "risk": sanitize_text(r.get("risk", "")),
                "severity": sev,
                "mitigation": sanitize_text(r.get("mitigation", ""))
            })

    merged["statistical_rationale"] = sanitize_text(plan.get("statistical_rationale", ""))

    return merged

# -------------------------
# PRD to Markdown conversion
# -------------------------
def prd_to_markdown(plan: Dict[str, Any]) -> str:
    """Convert PRD dict to nicely formatted Markdown."""
    plan = sanitize_plan(plan)
    md = []
    meta = plan.get("metadata", {})
    
    md.append(f"# {meta.get('title', 'Experiment PRD')}")
    md.append("")
    md.append(f"**Owner:** {meta.get('owner','-')}  |  **Team:** {meta.get('team','-')}  |  **ID:** {meta.get('experiment_id','-')}")
    md.append("")
    
    md.append("## Problem Statement")
    md.append(plan.get("problem_statement", "") or "-")
    md.append("")
    
    md.append("## Hypothesis")
    hyps = ensure_list(plan.get("hypotheses"))
    if hyps:
        for i, h in enumerate(hyps, 1):
            md.append(f"### Hypothesis {i}")
            md.append(h.get("hypothesis", ""))
            if h.get("rationale"):
                md.append(f"**Rationale:** {h.get('rationale')}")
            if h.get("example_implementation"):
                md.append(f"**Example Implementation:** {h.get('example_implementation')}")
            if h.get("behavioral_basis"):
                md.append(f"**Behavioral Basis:** {h.get('behavioral_basis')}")
            md.append("")
    else:
        md.append("- Not provided")
        md.append("")

    md.append("## Proposed Solution & Variants")
    md.append(plan.get("proposed_solution", "") or "-")
    md.append("")
    for v in ensure_list(plan.get("variants")):
        md.append(f"- **Control:** {v.get('control','')}")
        md.append(f"- **Variation:** {v.get('variation','')}")
        if v.get("notes"):
            md.append(f"  - _Notes:_ {v.get('notes')}")
    md.append("")

    if plan.get("metrics"):
        md.append("## Success Metrics")
        for m in plan.get("metrics", []):
            md.append(f"- **{m.get('name','')}** — {m.get('formula','')} (_{m.get('importance','')}_)")
        md.append("")

    if plan.get("guardrail_metrics"):
        md.append("## Guardrails")
        for g in plan.get("guardrail_metrics", []):
            md.append(f"- **{g.get('name','')}** — {g.get('direction','')} {g.get('threshold','')}")
        md.append("")

    md.append("## Experiment Design & Rollout")
    ed = ensure_dict(plan.get("experiment_design", {}))
    md.append(f"- **Traffic Allocation:** {ed.get('traffic_allocation','')}")
    md.append(f"- **Sample Size / Variant:** {ed.get('sample_size_per_variant','')}")
    md.append(f"- **Total Sample Size:** {ed.get('total_sample_size','')}")
    md.append(f"- **Estimated Duration (days):** {ed.get('test_duration_days','')}")
    if ed.get("dau_coverage_percent") is not None:
        md.append(f"- **DAU Coverage:** {ed.get('dau_coverage_percent')}%")
    md.append("")

    if plan.get("risks_and_assumptions"):
        md.append("## Risks & Mitigation")
        for r in plan.get("risks_and_assumptions", []):
            md.append(f"- **{r.get('risk','')}** (_{r.get('severity','')}_): {r.get('mitigation','')}")
        md.append("")

    sl = ensure_dict(plan.get("success_learning_criteria", {}))
    if any(sl.values()):
        md.append("## Success & Learning Criteria")
        md.append(f"- **Definition of Success:** {sl.get('definition_of_success','')}")
        md.append(f"- **Stopping Rules:** {sl.get('stopping_rules','')}")
        md.append(f"- **Rollback Criteria:** {sl.get('rollback_criteria','')}")
        md.append("")

    sc = ensure_dict(plan.get("success_criteria", {}))
    md.append("## Statistical Rationale & Success Criteria")
    md.append(f"- **Confidence Level:** {sc.get('confidence_level','')}%")
    md.append(f"- **Power:** {sc.get('power','')}%")
    md.append(f"- **MDE:** {sc.get('MDE','')}%")
    if plan.get("statistical_rationale"):
        md.append("")
        md.append("### Statistical Rationale")
        md.append(plan.get("statistical_rationale", ""))

    return "\n\n".join(md)

# -------------------------
# Tips helper
# -------------------------
def generate_tips(context: Dict[str, Any], current_step: str) -> List[str]:
    """Generate contextual tips using prompt engine or fallback."""
    # Try LLM first
    try:
        if PROMPT_ENGINE_AVAILABLE and _generate_dynamic_tips:
            out = _generate_dynamic_tips(current_step, context)
            if isinstance(out, list) and out:
                return out
            if isinstance(out, str):
                return [l.strip() for l in out.splitlines() if l.strip()][:5]
    except Exception as e:
        st.error(f"Tips generation failed: {str(e)}")

    # Static fallback tips
    base_tips = {
        "inputs": [
            "🎯 Keep your business goal crisp and measurable (e.g., 'Increase checkout conversion by 3%').",
            "📊 Use a single, well-defined primary metric — avoid compound metrics.",
            "🔢 Provide baseline numbers (e.g., '25% CTR') so the tool can size experiments."
        ],
        "hypothesis": [
            "✍️ Make your hypothesis falsifiable: 'If X, then Y for Z users.'",
            "📌 Tie the hypothesis to the metric and the user persona explicitly.",
            "⚡ Start small — propose lightweight UI or copy changes first."
        ],
        "prd": [
            "🛡️ Add at least one guardrail metric to prevent regressions (e.g., retention, error rate).",
            "📐 Ensure sample size estimates are realistic given DAU coverage.",
            "🔁 Define stopping and rollback rules clearly to reduce risk during rollout."
        ]
    }
    return base_tips.get(current_step, base_tips["prd"])

# -------------------------
# Main application
# -------------------------
def main():
    inject_global_css()
    st.set_page_config(layout="wide", page_title="Experiment Architect")
    st.title("💡 Experiment Architect")
    st.markdown("Automated A/B testing PRD generator. Based on a business problem, it generates hypotheses and a full experiment plan.")

    # State initialization
    if "sidebar_context" not in st.session_state:
        st.session_state["sidebar_context"] = {
            "business_goal": "",
            "product_type": "",
            "user_persona": "",
            "key_metric": "",
            "current_value": "",
            "target_value": ""
        }
    if "hypotheses" not in st.session_state:
        st.session_state["hypotheses"] = []
    if "chosen_hypothesis" not in st.session_state:
        st.session_state["chosen_hypothesis"] = {}
    if "experiment_plan" not in st.session_state:
        st.session_state["experiment_plan"] = {}
    if "final_prd" not in st.session_state:
        st.session_state["final_prd"] = json.loads(json.dumps(DEFAULT_PLAN))
    if "show_tips_panel" not in st.session_state:
        st.session_state["show_tips_panel"] = False
    if "tips_list" not in st.session_state:
        st.session_state["tips_list"] = []
    
    # Sidebar for inputs
    with st.sidebar:
        st.subheader("1. Business Context")
        ctx = st.session_state["sidebar_context"]
        ctx["business_goal"] = st.text_input("Business Goal", value=ctx["business_goal"] or "increase revenue")
        ctx["product_type"] = st.text_input("Product Type", value=ctx["product_type"] or "mobile gaming app")
        ctx["user_persona"] = st.text_input("Target Persona", value=ctx["user_persona"] or "18 to 22 age group US users")
        ctx["key_metric"] = st.text_input("Key Metric", value=ctx["key_metric"] or "ARPU")
        ctx["current_value"] = st.text_input("Current Value", value=ctx["current_value"] or ".55")
        ctx["target_value"] = st.text_input("Target Value", value=ctx["target_value"] or ".65")
        
        # Action button to generate hypotheses
        if st.button("Generate Hypotheses", use_container_width=True):
            if PROMPT_ENGINE_AVAILABLE and _generate_hypotheses and _client:  # Add _client check
                with st.spinner("Generating hypotheses..."):
                    try:
                        generated_hypotheses = _generate_hypotheses(ctx)
                        if generated_hypotheses and "error" not in generated_hypotheses[0]:
                            st.session_state["hypotheses"] = generated_hypotheses
                            st.success("Hypotheses generated!")
                        else:
                            st.error("Failed to generate hypotheses - check your API key and inputs")
                    except Exception as e:
                        st.error(f"Hypothesis generation failed: {str(e)}")
            else:
                st.error("LLM service unavailable - check Groq API configuration")

    # Main content area
    if not st.session_state.get("hypotheses"):
        st.info("Enter your product details in the sidebar and click 'Generate Hypotheses' to get started.")
    elif not st.session_state.get("chosen_hypothesis"):
        st.subheader("2. Choose a Hypothesis")
        st.info("Hypotheses have been generated. Scroll down to view and select one.")
        
        # Display AI-generated hypotheses for user selection
        with st.expander("🤖 AI-Generated Hypotheses", expanded=True):
            if not st.session_state["hypotheses"]:
                st.warning("No hypotheses were generated. Please try again or check the prompt engine.")
            else:
                for i, h in enumerate(st.session_state["hypotheses"]):
                    with st.container(border=True):
                        st.markdown(f"**Hypothesis {i+1}:** {h.get('hypothesis','')}")
                        st.markdown(f"**Rationale:** {h.get('rationale','')}")
                        st.markdown(f"**Implementation:** {h.get('example_implementation','')}")
                        st.markdown(f"**Behavioral Basis:** {h.get('behavioral_basis','')}")

                        if st.button(f"Choose This One", key=f"choose_hyp_{i}"):
                            st.session_state["chosen_hypothesis"] = h
                            
                            if PROMPT_ENGINE_AVAILABLE and _generate_experiment_plan:
                                with st.spinner("Generating full PRD..."):
                                    prd_raw = _generate_experiment_plan(st.session_state["sidebar_context"], h)
                                    if isinstance(prd_raw, dict):
                                        st.session_state["experiment_plan"] = prd_raw
                                    else:
                                        st.session_state["experiment_plan"] = extract_json_from_text(prd_raw)
                                    st.session_state["final_prd"] = sanitize_plan(st.session_state["experiment_plan"])
                                    st.success("Full PRD generated!")
                            else:
                                st.session_state["experiment_plan"] = DEFAULT_PLAN
                                st.session_state["final_prd"] = sanitize_plan(DEFAULT_PLAN)
                                st.warning("Prompt engine not available. Generated a default PRD plan.")
                            st.rerun()

    else:
        # PRD editing and preview tabs
        st.subheader("3. Edit & Finalize PRD")
        st.markdown("Edit the generated PRD sections below. Use the **Preview** tab to see the final document.")
        
        tab1, tab2 = st.tabs(["✍️ Edit PRD", "👁️ Preview & Export"])

        with tab1:
            # Metadata
            with st.expander("📝 Metadata", expanded=True):
                meta = st.session_state["final_prd"]["metadata"]
                meta["title"] = st.text_input("Title", value=meta["title"], key="edit_title")
                meta["team"] = st.text_input("Team", value=meta["team"], key="edit_team")
                meta["owner"] = st.text_input("Owner", value=meta["owner"], key="edit_owner")
                st.session_state["final_prd"]["metadata"] = meta

            # Problem Statement
            with st.expander("❓ Problem Statement", expanded=False):
                st.session_state["final_prd"]["problem_statement"] = st.text_area(
                    "Problem Statement",
                    value=st.session_state["final_prd"].get("problem_statement", ""),
                    height=120,
                    key="edit_problem"
                )
                if st.button("♻️ Regenerate Problem Statement", key="regen_problem"):
                    with st.spinner("Regenerating problem statement..."):
                        try:
                            ctx = st.session_state.get("sidebar_context", {})
                            hyp = st.session_state.get("chosen_hypothesis", {})
                            if PROMPT_ENGINE_AVAILABLE and _generate_experiment_plan:
                                raw_new = _generate_experiment_plan(ctx, hyp)
                                parsed_new = raw_new if isinstance(raw_new, dict) else extract_json_from_text(raw_new)
                                new_plan = sanitize_plan(parsed_new)
                                st.session_state["final_prd"]["problem_statement"] = new_plan.get("problem_statement", "")
                                st.success("Problem statement regenerated.")
                            else:
                                st.warning("LLM not available.")
                        except Exception as e:
                            st.error(f"Failed to regenerate problem statement: {str(e)}")

            # Proposed Solution & Variants
            with st.expander("🛠️ Proposed Solution & Variants", expanded=False):
                st.session_state["final_prd"]["proposed_solution"] = st.text_area(
                    "Proposed Solution", 
                    value=st.session_state["final_prd"].get("proposed_solution", ""), 
                    height=120, 
                    key="edit_solution"
                )
                
                variants = ensure_list(st.session_state["final_prd"].get("variants", []))
                if not variants:
                    variants = [{"control": "", "variation": "", "notes": ""}]
                
                v0 = variants[0]
                v0["control"] = st.text_area(
                    "Control", 
                    value=v0.get("control",""), 
                    height=80, 
                    key="edit_vctrl"
                )
                v0["variation"] = st.text_area(
                    "Variation", 
                    value=v0.get("variation",""), 
                    height=80, 
                    key="edit_vvar"
                )
                v0["notes"] = st.text_input(
                    "Notes", 
                    value=v0.get("notes",""), 
                    key="edit_vnotes"
                )
                st.session_state["final_prd"]["variants"] = [v0]
                
                if st.button("♻️ Regenerate Solution & Variants", key="regen_variants"):
                    with st.spinner("Regenerating variants..."):
                        try:
                            ctx = st.session_state.get("sidebar_context", {})
                            hyp = st.session_state.get("final_prd", {}).get("hypotheses", [{}])[0]
                            if PROMPT_ENGINE_AVAILABLE and _generate_experiment_plan:
                                raw_new = _generate_experiment_plan(ctx, hyp)
                                parsed_new = raw_new if isinstance(raw_new, dict) else extract_json_from_text(raw_new)
                                new_plan = sanitize_plan(parsed_new)
                                st.session_state["final_prd"]["proposed_solution"] = new_plan.get("proposed_solution", "")
                                st.session_state["final_prd"]["variants"] = new_plan.get("variants", [])
                                st.success("Solution & variants regenerated.")
                            else:
                                st.warning("LLM not available.")
                        except Exception as e:
                            st.error(f"Failed to regenerate variants: {str(e)}")

            # --- MODIFIED: This section is now dynamic and has a new title ---
            with st.expander("📊 Success Metrics", expanded=False):
                if 'metrics' not in st.session_state.final_prd or not isinstance(st.session_state.final_prd['metrics'], list):
                    st.session_state.final_prd['metrics'] = []
                
                for i, metric in enumerate(st.session_state.final_prd['metrics']):
                    cols = st.columns([4, 4, 2, 1])
                    metric['name'] = cols[0].text_input("Metric Name", value=metric.get('name', ''), key=f"metric_name_{i}")
                    metric['formula'] = cols[1].text_input("How is it measured?", value=metric.get('formula', ''), key=f"metric_formula_{i}")
                    metric['importance'] = cols[2].selectbox("Importance", ["Primary", "Secondary"], index=0 if metric.get('importance', 'Primary') == 'Primary' else 1, key=f"metric_imp_{i}")
                    if cols[3].button("➖", key=f"remove_metric_{i}", help="Remove this metric"):
                        st.session_state.final_prd['metrics'].pop(i)
                        st.rerun()
                
                if st.button("Add Metric"):
                    st.session_state.final_prd['metrics'].append({'name': '', 'formula': '', 'importance': 'Secondary'})
                    st.rerun()

            # --- MODIFIED: This section is now dynamic ---
            with st.expander("🛡️ Guardrail Metrics", expanded=False):
                if 'guardrail_metrics' not in st.session_state.final_prd or not isinstance(st.session_state.final_prd['guardrail_metrics'], list):
                    st.session_state.final_prd['guardrail_metrics'] = []

                for i, guardrail in enumerate(st.session_state.final_prd['guardrail_metrics']):
                    cols = st.columns([4, 3, 3, 1])
                    guardrail['name'] = cols[0].text_input("Guardrail Name", value=guardrail.get('name', ''), key=f"guard_name_{i}")
                    guardrail['direction'] = cols[1].selectbox("Desired Direction", ["Increase", "Decrease", "No Change"], index=["Increase", "Decrease", "No Change"].index(guardrail.get("direction", "No Change")), key=f"guard_dir_{i}")
                    guardrail['threshold'] = cols[2].text_input("Threshold", value=guardrail.get('threshold', ''), key=f"guard_thresh_{i}")
                    if cols[3].button("➖", key=f"remove_guard_{i}", help="Remove this guardrail"):
                        st.session_state.final_prd['guardrail_metrics'].pop(i)
                        st.rerun()

                if st.button("Add Guardrail"):
                    st.session_state.final_prd['guardrail_metrics'].append({'name': '', 'direction': 'No Change', 'threshold': ''})
                    st.rerun()

            # --- NEW: Integrated Statistical Calculator Expander ---
            with st.expander("📐 Experiment Design & Calculator", expanded=True):
                if not SCIPY_AVAILABLE:
                    st.error("Statistical calculation requires the 'scipy' library. Please install it (`pip install scipy`) to enable this feature.")
                else:
                    st.info("Adjust the values below to automatically calculate sample size and test duration.")
                    # Ensure the experiment_design dict exists
                    if "experiment_design" not in st.session_state.final_prd:
                        st.session_state.final_prd["experiment_design"] = {}
                    ed = st.session_state.final_prd["experiment_design"]

                    c1, c2, c3 = st.columns(3)
                    # --- NEW: Aggressive input validation via number_input parameters ---
                    ed['baseline_conversion_rate'] = c1.number_input(
                        "Baseline Conversion Rate (%)", min_value=0.01, max_value=99.99,
                        value=ed.get('baseline_conversion_rate', 10.0), step=0.1, format="%.2f",
                        help="The current conversion rate of your control group (e.g., 15.5 for 15.5%)."
                    )
                    ed['mde_percent'] = c2.number_input(
                        "Minimum Detectable Effect (MDE) (%)", min_value=0.1,
                        value=ed.get('mde_percent', 5.0), step=0.1, format="%.1f",
                        help="The minimum relative lift you want to detect (e.g., 5 for a 5% lift)."
                    )
                    ed['daily_traffic_for_experiment'] = c3.number_input(
                        "Daily Users in Experiment", min_value=1,
                        value=ed.get('daily_traffic_for_experiment', 1000), step=100,
                        help="Total daily users who will see either the control or variation."
                    )
                    
                    # Perform Calculation in real-time
                    baseline_rate_dec = ed['baseline_conversion_rate'] / 100.0
                    mde_dec = ed['mde_percent'] / 100.0
                    power_dec = st.session_state.final_prd.get('success_criteria', {}).get('power', 80.0) / 100.0
                    
                    sample_size = calculate_sample_size(baseline_rate_dec, mde_dec, power=power_dec)
                    
                    # Update state and display results
                    if sample_size:
                        ed['sample_size_per_variant'] = sample_size
                        ed['total_sample_size'] = sample_size * 2
                        daily_traffic = ed.get('daily_traffic_for_experiment', 0)
                        ed['test_duration_days'] = math.ceil(ed['total_sample_size'] / daily_traffic) if daily_traffic > 0 else 0
                        
                        st.success(
                            f"""
                            **Required Sample Size (per variant):** `{ed['sample_size_per_variant']:,}`\n
                            **Total Required Sample Size:** `{ed['total_sample_size']:,}`\n
                            **Estimated Test Duration:** `{ed['test_duration_days']} days`
                            """
                        )
                    else:
                        st.warning("Invalid inputs for calculation. Please check that MDE is not too high and baseline is valid.")

            # --- MODIFIED: This section is now dynamic ---
            with st.expander("⚠️ Risks & Assumptions", expanded=False):
                if 'risks_and_assumptions' not in st.session_state.final_prd or not isinstance(st.session_state.final_prd['risks_and_assumptions'], list):
                    st.session_state.final_prd['risks_and_assumptions'] = []

                for i, risk in enumerate(st.session_state.final_prd['risks_and_assumptions']):
                    cols = st.columns([5, 2, 5, 1])
                    risk['risk'] = cols[0].text_input("Risk", value=risk.get('risk', ''), key=f"risk_risk_{i}")
                    risk['severity'] = cols[1].selectbox("Severity", ["High", "Medium", "Low"], index=["High", "Medium", "Low"].index(risk.get("severity", "Medium")), key=f"risk_sev_{i}")
                    risk['mitigation'] = cols[2].text_input("Mitigation", value=risk.get('mitigation', ''), key=f"risk_mit_{i}")
                    if cols[3].button("➖", key=f"remove_risk_{i}", help="Remove this risk"):
                        st.session_state.final_prd['risks_and_assumptions'].pop(i)
                        st.rerun()

                if st.button("Add Risk"):
                    st.session_state.final_prd['risks_and_assumptions'].append({'risk': '', 'severity': 'Medium', 'mitigation': ''})
                    st.rerun()

            # Success & Learning Criteria
            with st.expander("📘 Success & Learning Criteria", expanded=False):
                sl = ensure_dict(st.session_state["final_prd"].get("success_learning_criteria", {}))
                sl["definition_of_success"] = st.text_input(
                    "Definition of Success", 
                    value=sl.get("definition_of_success", ""), 
                    key="edit_sl_def"
                )
                sl["stopping_rules"] = st.text_input(
                    "Stopping Rules", 
                    value=sl.get("stopping_rules", ""), 
                    key="edit_sl_stop"
                )
                sl["rollback_criteria"] = st.text_input(
                    "Rollback Criteria", 
                    value=sl.get("rollback_criteria", ""), 
                    key="edit_sl_roll"
                )
                st.session_state["final_prd"]["success_learning_criteria"] = sl
                
                if st.button("♻️ Regenerate Success & Learning", key="regen_success_learning"):
                    with st.spinner("Regenerating success criteria..."):
                        try:
                            ctx = st.session_state.get("sidebar_context", {})
                            hyp = st.session_state.get("final_prd", {}).get("hypotheses", [{}])[0]
                            if PROMPT_ENGINE_AVAILABLE and _generate_experiment_plan:
                                raw_new = _generate_experiment_plan(ctx, hyp)
                                parsed_new = raw_new if isinstance(raw_new, dict) else extract_json_from_text(raw_new)
                                new_plan = sanitize_plan(parsed_new)
                                st.session_state["final_prd"]["success_learning_criteria"] = new_plan.get("success_learning_criteria", {})
                                st.success("Success & Learning regenerated.")
                            else:
                                st.warning("LLM not available.")
                        except Exception as e:
                            st.error(f"Failed to regenerate success & learning: {str(e)}")

            # Statistical Rationale
            with st.expander("📐 Statistical Rationale", expanded=False):
                st.session_state["final_prd"]["statistical_rationale"] = st.text_area(
                    "Statistical Rationale", 
                    value=st.session_state["final_prd"].get("statistical_rationale", ""), 
                    height=120, 
                    key="edit_stat"
                )
                if st.button("♻️ Regenerate Statistical Rationale", key="regen_stats"):
                    with st.spinner("Regenerating statistical rationale..."):
                        try:
                            ctx = st.session_state.get("sidebar_context", {})
                            hyp = st.session_state.get("final_prd", {}).get("hypotheses", [{}])[0]
                            if PROMPT_ENGINE_AVAILABLE and _generate_experiment_plan:
                                raw_new = _generate_experiment_plan(ctx, hyp)
                                parsed_new = raw_new if isinstance(raw_new, dict) else extract_json_from_text(raw_new)
                                new_plan = sanitize_plan(parsed_new)
                                st.session_state["final_prd"]["statistical_rationale"] = new_plan.get("statistical_rationale", "")
                                st.success("Statistical rationale regenerated.")
                            else:
                                st.warning("LLM not available.")
                        except Exception as e:
                            st.error(f"Failed to regenerate statistical rationale: {str(e)}")

            # Save edits back
            st.session_state["final_prd"] = sanitize_plan(st.session_state["final_prd"])

            # Quick quality check
            if st.button("🔍 Quick AI Quality Check", key="qc_button"):
                with st.spinner("Running quality check..."):
                    try:
                        if PROMPT_ENGINE_AVAILABLE and _validate_experiment_plan:
                            issues = _validate_experiment_plan(st.session_state["final_prd"])
                            st.success("Quality check complete.")
                            st.info(json.dumps(issues, indent=2))
                        else:
                            st.info("No validation engine available. Consider reviewing metrics and guardrails manually.")
                    except Exception as e:
                        st.error(f"Quality check failed: {str(e)}")

        # Preview tab
        with tab2:
            st.markdown("### Preview — Live document view")
            preview_md = prd_to_markdown(st.session_state["final_prd"])
            st.markdown(preview_md, unsafe_allow_html=True)

            # Export buttons
            col_p1, col_p2, col_p3 = st.columns([1, 1, 1])
            with col_p1:
                if st.button("Export: Markdown", key="export_md"):
                    md_text = prd_to_markdown(st.session_state["final_prd"])
                    st.download_button(
                        "Download .md", 
                        data=md_text, 
                        file_name="experiment_prd.md", 
                        mime="text/markdown", 
                        key="dl_md_btn"
                    )
            with col_p2:
                if st.button("Export: JSON", key="export_json"):
                    json_str = json.dumps(st.session_state["final_prd"], indent=2)
                    st.download_button(
                        "Download .json", 
                        data=json_str, 
                        file_name="experiment_prd.json", 
                        mime="application/json", 
                        key="dl_json_btn"
                    )
            with col_p3:
                if st.button("Export: PDF", key="export_pdf"):
                    if PDF_AVAILABLE:
                        try:
                            pdf_bytes = generate_pdf_bytes_from_prd_dict(st.session_state["final_prd"])
                            if pdf_bytes:
                                st.download_button(
                                    "Download .pdf", 
                                    data=pdf_bytes, 
                                    file_name="experiment_prd.pdf", 
                                    mime="application/pdf", 
                                    key="dl_pdf_btn"
                                )
                        except Exception as e:
                            st.error(f"PDF generation failed: {str(e)}")
                    else:
                        st.warning("PDF export not available (reportlab not installed)")
            
            # DOCX export
            if st.button("Export: DOCX", key="export_docx"):
                if DOCX_AVAILABLE:
                    try:
                        docx_bytes = generate_docx_bytes_from_plan(st.session_state["final_prd"])
                        if docx_bytes:
                            st.download_button(
                                "Download .docx", 
                                data=docx_bytes, 
                                file_name="experiment_prd.docx", 
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
                                key="dl_docx_btn"
                            )
                    except Exception as e:
                        st.error(f"DOCX generation failed: {str(e)}")
                else:
                    st.warning("DOCX export not available (python-docx not installed)")

        # Reset and minor actions
        colr1, colr2 = st.columns([1, 1])
        with colr1:
            if st.button("🔁 Reset PRD", key="reset_prd"):
                st.session_state.pop("experiment_plan", None)
                st.session_state.pop("final_prd", None)
                st.success("PRD reset. You can generate a new one.")
        with colr2:
            if st.button("🧾 Save to session", key="save_session"):
                st.session_state["experiment_plan"] = st.session_state.get("final_prd", {})
                st.success("Saved current PRD to session_state['experiment_plan'].")

    # Floating Tips Panel
    if st.session_state.get("show_tips_panel", False):
        # Build context for tips
        ctx = st.session_state.get("sidebar_context", {})
        # Current step detection
        if "final_prd" in st.session_state:
            step = "prd"
        elif "chosen_hypothesis" in st.session_state:
            step = "hypothesis"
        else:
            step = "inputs"

        # Container styling via markdown (floating panel)
        panel_html = """
        <div class="floating-tips-panel">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
            <div style="font-weight:700">💡 Tips</div>
            <div><button id="close_tips" style="border:none;background:#fff;cursor:pointer;font-size:14px">❌</button></div>
          </div>
        """
        # render existing tips if any
        tips = st.session_state.get("tips_list", [])
        if tips:
            for t in tips:
                panel_html += f'<div class="tip-card">{t}</div>'
        else:
            panel_html += '<div class="muted">No tips yet. Click "Refresh Tips" to get tailored guidance.</div>'

        panel_html += """
          <div style="display:flex;gap:8px;margin-top:8px">
            <form action="#" method="post">
              <input type="submit" id="refresh_tips" value="🔄 Refresh Tips" style="padding:8px 12px;border-radius:8px;border:1px solid #e2e8f0;background:#eef2ff;cursor:pointer;font-weight:600">
            </form>
          </div>
        </div>
        """

        st.markdown(panel_html, unsafe_allow_html=True)

        # Buttons can't be captured inside the raw HTML reliably; provide Streamlit controls under panel for actions.
        refresh = st.button("🔄 Refresh Tips", key="refresh_tips_streamlit")
        close = st.button("Close Tips", key="close_tips_streamlit")
        if refresh:
            with st.spinner("Refreshing tips..."):
                try:
                    tips_out = generate_tips(ctx, step)
                    # normalize to strings (limit 5)
                    normalized = [str(x).strip() for x in (tips_out or [])][:5]
                    st.session_state["tips_list"] = normalized
                    st.success("Tips updated.")
                except Exception as e:
                    st.error(f"Failed to get tips: {str(e)}")
        if close:
            st.session_state["show_tips_panel"] = False

    # Floating button to open tips (bottom-right)
    floating_html = """
    <div class="floating-tips-button">
      <button onclick="window.parent.document.querySelector('button[kind=\"primary\"]').click();" style="background:#6366f1;color:#fff;border:none;padding:12px;border-radius:999px;box-shadow:0 6px 18px rgba(99,102,241,0.3);cursor:pointer;font-weight:700">
        💡 Tips
      </button>
    </div>
    """
    # We render a static floating button; functionality is via the header toggles and streamlit buttons.
    st.markdown(floating_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
