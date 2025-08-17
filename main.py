import json
import math
import random
import string
from io import BytesIO
from typing import List, Dict, Any, Optional

import streamlit as st

# Try optional libraries for exports
PDF_AVAILABLE = False
DOCX_AVAILABLE = False
try:
    # ReportLab for PDF
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

try:
    # python-docx for DOCX
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_BREAK
    DOCX_AVAILABLE = True
except Exception:
    DOCX_AVAILABLE = False

# Prompt engine functions
try:
    from prompt_engine import (
        generate_experiment_plan,
        generate_hypothesis_details,
        validate_experiment_plan,
        generate_hypotheses,
    )
    PROMPT_ENGINE_AVAILABLE = True
except Exception:
    PROMPT_ENGINE_AVAILABLE = False


# -------------------------
# UI Styling helpers (CSS) and Markdown renderer
# -------------------------
def inject_global_css():
    css = """
    <style>
    [data-testid="stHeader"] {background: linear-gradient(90deg, #0ea5e9 0%, #6366f1 100%); color: white;}
    [data-testid="stHeader"] * {color: white !important;}
    .block-container {max-width: 1200px; padding-top: 1rem;}
    .stButton > button {border-radius: 10px; padding: 0.6rem 1rem; font-weight: 600;}
    .stTabs [data-baseweb="tab-list"] button {padding: 0.6rem 1rem; border-radius: 999px; margin-right: 6px;}
    [data-testid="stExpander"] {border: 1px solid #e5e7eb; border-radius: 12px; overflow: hidden; background: #ffffff;}
    [data-testid="stExpander"] details summary {font-weight:600;}
    .card {border:1px solid #e5e7eb; border-radius:16px; padding:1rem; background:#fff;}
    .pill {display:inline-block; padding:4px 10px; border-radius:999px; background:#eef2ff; margin-right:6px; font-size:0.85rem;}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def prd_to_markdown(plan: Dict[str, Any]) -> str:
    plan = sanitize_plan(plan)
    md = []
    md.append(f"# {plan.get('metadata',{}).get('title','Experiment PRD')}")
    md.append("")
    md.append(f"**Owner:** {plan.get('metadata',{}).get('owner','-')}  |  **Team:** {plan.get('metadata',{}).get('team','-')}  |  **ID:** {plan.get('metadata',{}).get('experiment_id','-')}")
    md.append("")
    md.append("## Problem Statement")
    md.append(plan.get("problem_statement","-") or "-")
    md.append("")
    md.append("## Hypothesis")
    if plan.get("hypotheses"):
        h = plan["hypotheses"][0]
        md.append(f"- **Hypothesis:** {h.get('hypothesis','')}")
        if h.get("rationale"):
            md.append(f"- **Rationale:** {h.get('rationale','')}")
        if h.get("example_implementation"):
            md.append(f"- **Example:** {h.get('example_implementation','')}")
        if h.get("behavioral_basis"):
            md.append(f"- **Behavioral Basis:** {h.get('behavioral_basis','')}")
    else:
        md.append("- Not set")
    md.append("")
    md.append("## Proposed Solution & Variants")
    md.append(plan.get("proposed_solution","-") or "-")
    for v in plan.get("variants", []):
        md.append(f"- **Control:** {v.get('control','')}")
        md.append(f"- **Variation:** {v.get('variation','')}")
        if v.get("notes"):
            md.append(f"  - _Notes:_ {v.get('notes')}")
    md.append("")
    if plan.get("metrics"):
        md.append("## Metrics")
        for m in plan["metrics"]:
            md.append(f"- **{m.get('name','')}** — {m.get('formula','')} (_{m.get('importance','')}_)")
    if plan.get("guardrail_metrics"):
        md.append("")
        md.append("## Guardrails")
        for g in plan["guardrail_metrics"]:
            md.append(f"- **{g.get('name','')}** — {g.get('direction','')} {g.get('threshold','')}")
    md.append("")
    md.append("## Experiment Design")
    ed = plan.get("experiment_design",{})
    md.append(f"- Traffic Allocation: {ed.get('traffic_allocation','')}")
    md.append(f"- Sample Size / Variant: {ed.get('sample_size_per_variant','')}")
    md.append(f"- Total Sample Size: {ed.get('total_sample_size','')}")
    md.append(f"- Estimated Duration (days): {ed.get('test_duration_days','')}")
    if ed.get("dau_coverage_percent") is not None:
        md.append(f"- DAU Coverage: {ed.get('dau_coverage_percent')}%")
    md.append("")
    if plan.get("risks_and_assumptions"):
        md.append("## Risks & Mitigation")
        for r in plan["risks_and_assumptions"]:
            md.append(f"- **{r.get('risk','')}** (_{r.get('severity','')}_): {r.get('mitigation','')}")
    md.append("")
    if plan.get("success_learning_criteria"):
        sl = plan["success_learning_criteria"]
        md.append("## Success & Learning Criteria")
        md.append(f"- Definition of Success: {sl.get('definition_of_success','')}")
        md.append(f"- Stopping Rules: {sl.get('stopping_rules','')}")
        md.append(f"- Rollback Criteria: {sl.get('rollback_criteria','')}")
    sc = plan.get("success_criteria",{})
    md.append("")
    md.append("## Statistical Rationale & Success Criteria")
    md.append(f"- Confidence Level: {sc.get('confidence_level','')}%")
    if sc.get("power"):
        md.append(f"- Power: {sc.get('power')}%")
    md.append(f"- MDE: {sc.get('MDE','')}%")
    if sc.get("benchmark"):
        md.append(f"- Benchmark: {sc.get('benchmark')}")
    if sc.get("monitoring"):
        md.append(f"- Monitoring: {sc.get('monitoring')}")
    if plan.get("statistical_rationale"):
        md.append("")
        md.append(str(plan.get("statistical_rationale")))
    return "\n".join(md)


# -------------------------
# Data models / defaults
# -------------------------
DEFAULT_PLAN: Dict[str, Any] = {
    "metadata": {
        "title": "Untitled Experiment",
        "team": "",
        "owner": "",
        "experiment_id": "",
    },
    "problem_statement": "",
    "hypotheses": [],
    "proposed_solution": "",
    "variants": [{"control": "", "variation": "", "notes": ""}],
    "metrics": [
        {"name": "", "formula": "", "importance": "Primary"},
    ],
    "guardrail_metrics": [
        {"name": "", "direction": "Decrease", "threshold": ""},
    ],
    "experiment_design": {
        "traffic_allocation": "50/50",
        "sample_size_per_variant": 0,
        "total_sample_size": 0,
        "test_duration_days": 0,
        "dau_coverage_percent": 0.0,
        "power": 80.0,
    },
    "success_criteria": {
        "confidence_level": 95.0,
        "power": 80.0,
        "MDE": 1.0,
        "benchmark": "",
        "monitoring": "",
    },
    "success_learning_criteria": {
        "definition_of_success": "",
        "stopping_rules": "",
        "rollback_criteria": "",
    },
    "risks_and_assumptions": [
        {"risk": "", "severity": "Medium", "mitigation": ""},
    ],
    "statistical_rationale": "",
}

# -------------------------
# Utils
# -------------------------
def sanitize_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (dict, list)):
        return json.dumps(x, ensure_ascii=False)
    return str(x).strip()


def ensure_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def ensure_dict(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    return {}


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            return int(x)
        s = str(x).strip()
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace("%", "")
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def generate_experiment_id(prefix: str = "EXP") -> str:
    rand = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
    return f"{prefix}-{rand}"


def extract_json_from_text(text: Optional[str]) -> Dict[str, Any]:
    """
    Extract a JSON object from a possibly-messy LLM response.
    """
    if not text:
        return {}
    text = text.strip()
    # Try direct JSON first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to find a JSON block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Try list form
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return {}


def calculate_sample_size(mde_percent: float, baseline_rate: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """
    Very rough sample size calculation for proportion differences.
    """
    try:
        from math import sqrt
        from scipy.stats import norm  # type: ignore
        z_alpha = norm.ppf(1 - alpha / 2.0)
        z_beta = norm.ppf(power)
        p1 = baseline_rate
        p2 = baseline_rate * (1 + mde_percent / 100.0)
        pooled = (p1 * (1 - p1) + p2 * (1 - p2))
        n = 2 * (z_alpha + z_beta) ** 2 * pooled / ((p2 - p1) ** 2)
        return int(n)
    except Exception:
        return 0


def pdf_safe(text: str) -> str:
    return text.replace("\t", "    ")


def generate_pdf_bytes_from_prd_dict(plan: Dict[str, Any]) -> Optional[bytes]:
    """
    Generate a lightweight PDF of the PRD using ReportLab if available.
    """
    if not PDF_AVAILABLE:
        return None

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    def add_heading(txt, level=1):
        style = styles["Heading1"] if level == 1 else styles["Heading2"]
        story.append(Paragraph(pdf_safe(txt), style))
        story.append(Spacer(1, 12))

    def add_paragraph(txt):
        story.append(Paragraph(pdf_safe(txt), styles["BodyText"]))
        story.append(Spacer(1, 8))

    def add_table(data, col_widths=None):
        t = Table(data, colWidths=col_widths)
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 12))

    # Metadata
    add_heading(plan.get("metadata", {}).get("title", "Experiment PRD"), level=1)
    meta = plan.get("metadata", {})
    add_paragraph(f"<b>Experiment ID:</b> {meta.get('experiment_id','')}")
    add_paragraph(f"<b>Team:</b> {meta.get('team','')}")
    add_paragraph(f"<b>Owner:</b> {meta.get('owner','')}")

    # Problem Statement
    add_heading("Problem Statement", level=2)
    add_paragraph(plan.get("problem_statement", ""))

    # Hypotheses
    hyps = ensure_list(plan.get("hypotheses"))
    if hyps:
        for i, h in enumerate(hyps, start=1):
            add_heading(f"Hypothesis {i}", level=2)
            add_paragraph(f"<b>Hypothesis:</b> {h.get('hypothesis','')}")
            if h.get("rationale"):
                add_paragraph(f"<b>Rationale:</b> {h.get('rationale')}")
            if h.get("example_implementation"):
                add_paragraph(f"<b>Example Implementation:</b> {h.get('example_implementation')}")
            if h.get("behavioral_basis"):
                add_paragraph(f"<b>Behavioral Basis:</b> {h.get('behavioral_basis')}")

    # Proposed Solution & Variants
    add_heading("Proposed Solution", level=2)
    add_paragraph(plan.get("proposed_solution", ""))

    vars_data = [["Control", "Variation", "Notes"]]
    for v in ensure_list(plan.get("variants")):
        vars_data.append([v.get("control", ""), v.get("variation", ""), v.get("notes", "")])
    add_heading("Variants", level=2)
    add_table(vars_data, col_widths=[150, 150, 200])

    # Metrics
    metrics_data = [["Name", "Formula", "Importance"]]
    for m in ensure_list(plan.get("metrics")):
        metrics_data.append([m.get("name", ""), m.get("formula", ""), m.get("importance", "")])
    add_heading("Metrics", level=2)
    add_table(metrics_data, col_widths=[150, 200, 100])

    guard_data = [["Guardrail", "Direction", "Threshold"]]
    for g in ensure_list(plan.get("guardrail_metrics")):
        guard_data.append([g.get("name", ""), g.get("direction", ""), g.get("threshold", "")])
    add_heading("Guardrails", level=2)
    add_table(guard_data, col_widths=[180, 100, 170])

    # Experiment Design
    ed = ensure_dict(plan.get("experiment_design"))
    add_heading("Experiment Design", level=2)
    add_paragraph(f"<b>Traffic Allocation:</b> {ed.get('traffic_allocation','')}")
    add_paragraph(f"<b>Sample Size per Variant:</b> {ed.get('sample_size_per_variant','')}")
    add_paragraph(f"<b>Total Sample Size:</b> {ed.get('total_sample_size','')}")
    add_paragraph(f"<b>Estimated Duration (days):</b> {ed.get('test_duration_days','')}")
    if ed.get("dau_coverage_percent") is not None:
        add_paragraph(f"<b>DAU Coverage (%):</b> {ed.get('dau_coverage_percent')}")
    if ed.get("power") is not None:
        add_paragraph(f"<b>Statistical Power (%):</b> {ed.get('power')}")

    # Success & Learning
    sc = ensure_dict(plan.get("success_criteria"))
    sl = ensure_dict(plan.get("success_learning_criteria"))
    add_heading("Success & Learning", level=2)
    add_paragraph(f"<b>Confidence Level (%):</b> {sc.get('confidence_level','')}")
    add_paragraph(f"<b>Power (%):</b> {sc.get('power','')}")
    add_paragraph(f"<b>MDE (%):</b> {sc.get('MDE','')}")
    if sc.get("benchmark"):
        add_paragraph(f"<b>Benchmark:</b> {sc.get('benchmark')}")
    if sc.get("monitoring"):
        add_paragraph(f"<b>Monitoring:</b> {sc.get('monitoring')}")
    if sl.get("definition_of_success"):
        add_paragraph(f"<b>Definition of Success:</b> {sl.get('definition_of_success')}")
    if sl.get("stopping_rules"):
        add_paragraph(f"<b>Stopping Rules:</b> {sl.get('stopping_rules')}")
    if sl.get("rollback_criteria"):
        add_paragraph(f"<b>Rollback Criteria:</b> {sl.get('rollback_criteria')}")

    # Risks
    r_data = [["Risk", "Severity", "Mitigation"]]
    for r in ensure_list(plan.get("risks_and_assumptions")):
        r_data.append([r.get("risk", ""), r.get("severity", ""), r.get("mitigation", "")])
    add_heading("Risks & Assumptions", level=2)
    add_table(r_data, col_widths=[200, 80, 220])

    # Statistical rationale
    if plan.get("statistical_rationale"):
        add_heading("Statistical Rationale", level=2)
        add_paragraph(plan.get("statistical_rationale"))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


def generate_docx_bytes_from_plan(plan: Dict[str, Any]) -> Optional[bytes]:
    """
    Generate a DOCX file content from PRD plan if python-docx is available.
    """
    if not DOCX_AVAILABLE:
        return None

    def add_heading(p, text, level=1):
        run = p.add_paragraph().add_run(text)
        if level == 1:
            run.font.size = Pt(18)
        elif level == 2:
            run.font.size = Pt(14)
        else:
            run.font.size = Pt(12)

    doc = Document()
    meta = plan.get("metadata", {})

    # Title
    doc.add_heading(meta.get("title", "Experiment PRD"), 0)
    doc.add_paragraph(f"Experiment ID: {meta.get('experiment_id','')}")
    doc.add_paragraph(f"Team: {meta.get('team','')}")
    doc.add_paragraph(f"Owner: {meta.get('owner','')}")

    # Problem
    doc.add_heading("Problem Statement", level=1)
    doc.add_paragraph(plan.get("problem_statement", ""))

    # Hypotheses
    hyps = ensure_list(plan.get("hypotheses"))
    for i, h in enumerate(hyps, start=1):
        doc.add_heading(f"Hypothesis {i}", level=1)
        doc.add_paragraph(f"Hypothesis: {h.get('hypothesis','')}")
        if h.get("rationale"):
            doc.add_paragraph(f"Rationale: {h.get('rationale')}")
        if h.get("example_implementation"):
            doc.add_paragraph(f"Example Implementation: {h.get('example_implementation')}")
        if h.get("behavioral_basis"):
            doc.add_paragraph(f"Behavioral Basis: {h.get('behavioral_basis')}")

    # Proposed Solution & Variants
    doc.add_heading("Proposed Solution", level=1)
    doc.add_paragraph(plan.get("proposed_solution", ""))

    doc.add_heading("Variants", level=1)
    for v in ensure_list(plan.get("variants")):
        p = doc.add_paragraph()
        p.add_run("Control: ").bold = True
        p.add_run(v.get("control", ""))
        p = doc.add_paragraph()
        p.add_run("Variation: ").bold = True
        p.add_run(v.get("variation", ""))
        if v.get("notes"):
            p = doc.add_paragraph()
            p.add_run("Notes: ").italic = True
            p.add_run(v.get("notes", ""))

    # Metrics
    doc.add_heading("Metrics", level=1)
    for m in ensure_list(plan.get("metrics")):
        p = doc.add_paragraph()
        p.add_run(f"{m.get('name','')} ").bold = True
        p.add_run(f"— {m.get('formula','')} ({m.get('importance','')})")

    # Guardrails
    doc.add_heading("Guardrails", level=1)
    for g in ensure_list(plan.get("guardrail_metrics")):
        doc.add_paragraph(f"{g.get('name','')} — {g.get('direction','')} {g.get('threshold','')}")

    # Experiment Design
    ed = ensure_dict(plan.get("experiment_design"))
    doc.add_heading("Experiment Design", level=1)
    doc.add_paragraph(f"Traffic Allocation: {ed.get('traffic_allocation','')}")
    doc.add_paragraph(f"Sample Size per Variant: {ed.get('sample_size_per_variant','')}")
    doc.add_paragraph(f"Total Sample Size: {ed.get('total_sample_size','')}")
    doc.add_paragraph(f"Estimated Duration (days): {ed.get('test_duration_days','')}")
    if ed.get("dau_coverage_percent") is not None:
        doc.add_paragraph(f"DAU Coverage (%): {ed.get('dau_coverage_percent')}")
    if ed.get("power") is not None:
        doc.add_paragraph(f"Statistical Power (%): {ed.get('power')}")

    # Success & Learning
    sc = ensure_dict(plan.get("success_criteria"))
    sl = ensure_dict(plan.get("success_learning_criteria"))
    doc.add_heading("Success & Learning", level=1)
    doc.add_paragraph(f"Confidence Level (%): {sc.get('confidence_level','')}")
    doc.add_paragraph(f"Power (%): {sc.get('power','')}")
    doc.add_paragraph(f"MDE (%): {sc.get('MDE','')}")
    if sc.get("benchmark"):
        doc.add_paragraph(f"Benchmark: {sc.get('benchmark')}")
    if sc.get("monitoring"):
        doc.add_paragraph(f"Monitoring: {sc.get('monitoring')}")
    if sl.get("definition_of_success"):
        doc.add_paragraph(f"Definition of Success: {sl.get('definition_of_success')}")
    if sl.get("stopping_rules"):
        doc.add_paragraph(f"Stopping Rules: {sl.get('stopping_rules')}")
    if sl.get("rollback_criteria"):
        doc.add_paragraph(f"Rollback Criteria: {sl.get('rollback_criteria')}")

    # Risks
    doc.add_heading("Risks & Assumptions", level=1)
    for r in ensure_list(plan.get("risks_and_assumptions")):
        doc.add_paragraph(f"{r.get('risk','')} — {r.get('severity','')}: {r.get('mitigation','')}")

    # Statistical rationale
    if plan.get("statistical_rationale"):
        doc.add_heading("Statistical Rationale", level=1)
        doc.add_paragraph(plan.get("statistical_rationale"))

    f = BytesIO()
    doc.save(f)
    f.seek(0)
    return f.read()


# Backwards-compat alias
def generate_docx_bytes_from_prd_dict(plan: Dict[str, Any]) -> Optional[bytes]:
    return generate_docx_bytes_from_plan(plan)


def sanitize_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize shapes, fill defaults, coerce types.
    """
    if not isinstance(plan, dict):
        plan = {}
    merged = json.loads(json.dumps(DEFAULT_PLAN))  # deep copy-ish

    # metadata
    merged["metadata"].update(ensure_dict(plan.get("metadata", {})))
    if not merged["metadata"].get("experiment_id"):
        merged["metadata"]["experiment_id"] = generate_experiment_id()

    # scalar fields
    for k in ["problem_statement", "proposed_solution", "statistical_rationale"]:
        if k in plan:
            merged[k] = sanitize_text(plan.get(k))

    # lists dicts
    merged["hypotheses"] = []
    for h in ensure_list(plan.get("hypotheses")):
        if isinstance(h, dict):
            merged["hypotheses"].append(
                {
                    "hypothesis": sanitize_text(h.get("hypothesis", "")),
                    "rationale": sanitize_text(h.get("rationale", "")),
                    "example_implementation": sanitize_text(h.get("example_implementation", "")),
                    "behavioral_basis": sanitize_text(h.get("behavioral_basis", "")),
                }
            )

    merged["variants"] = []
    for v in ensure_list(plan.get("variants")):
        if isinstance(v, dict):
            merged["variants"].append(
                {
                    "control": sanitize_text(v.get("control", "")),
                    "variation": sanitize_text(v.get("variation", "")),
                    "notes": sanitize_text(v.get("notes", "")),
                }
            )

    merged["metrics"] = []
    for m in ensure_list(plan.get("metrics")):
        if isinstance(m, dict):
            importance = m.get("importance", "Primary")
            if importance not in ("Primary", "Secondary"):
                importance = "Primary"
            merged["metrics"].append(
                {
                    "name": sanitize_text(m.get("name", "")),
                    "formula": sanitize_text(m.get("formula", "")),
                    "importance": importance,
                }
            )

    merged["guardrail_metrics"] = []
    for g in ensure_list(plan.get("guardrail_metrics")):
        if isinstance(g, dict):
            direction = g.get("direction", "Decrease")
            if direction not in ("Increase", "Decrease", "No Change"):
                direction = "Decrease"
            merged["guardrail_metrics"].append(
                {
                    "name": sanitize_text(g.get("name", "")),
                    "direction": direction,
                    "threshold": sanitize_text(g.get("threshold", "")),
                }
            )

    # experiment design
    ed = ensure_dict(plan.get("experiment_design", {}))
    merged["experiment_design"] = {
        "traffic_allocation": sanitize_text(ed.get("traffic_allocation", "50/50")),
        "sample_size_per_variant": safe_int(ed.get("sample_size_per_variant", 0)),
        "total_sample_size": safe_int(ed.get("total_sample_size", 0)),
        "test_duration_days": safe_int(ed.get("test_duration_days", 0)),
        "dau_coverage_percent": safe_float(ed.get("dau_coverage_percent", 0.0)),
        "power": safe_float(ed.get("power", 80.0)),
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

    sl = ensure_dict(plan.get("success_learning_criteria", {}))
    merged["success_learning_criteria"] = {
        "definition_of_success": sanitize_text(sl.get("definition_of_success", "")),
        "stopping_rules": sanitize_text(sl.get("stopping_rules", "")),
        "rollback_criteria": sanitize_text(sl.get("rollback_criteria", "")),
    }

    merged["risks_and_assumptions"] = []
    for r in ensure_list(plan.get("risks_and_assumptions")):
        if isinstance(r, dict):
            sev = r.get("severity", "Medium")
            if sev not in ("High", "Medium", "Low"):
                sev = "Medium"
            merged["risks_and_assumptions"].append(
                {
                    "risk": sanitize_text(r.get("risk", "")),
                    "severity": sev,
                    "mitigation": sanitize_text(r.get("mitigation", "")),
                }
            )

    return merged
# -------------------------
# Streamlit App
# -------------------------
def main():
    st.set_page_config(
        page_title="PM Experiment Architect",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_global_css()

    st.title("🧪 PM Experiment Architect")
    st.caption("From hypothesis to polished A/B Test PRD — fast, structured, and statistically sane.")

    # -------------------------
    # Sidebar: Step 1 Inputs
    # -------------------------
    with st.sidebar:
        st.header("Step 1: Inputs")
        st.markdown(
            "<div class='card'>Provide the minimum context. The outputs will be fully personalized.</div>",
            unsafe_allow_html=True,
        )

        high_level_goal = st.text_input(
            "High-Level Business Goal",
            value=st.session_state.get("high_level_goal", ""),
            placeholder="e.g., Increase checkout conversion rate",
            key="high_level_goal",
        )
        product_type = st.selectbox(
            "Product Type",
            ["Mobile App", "E-commerce Website", "SaaS Dashboard", "Browser Extension", "Marketplace", "B2B Admin Console"],
            index=st.session_state.get("product_type_index", 0),
            key="product_type",
        )
        target_user = st.text_input(
            "Target User Persona",
            value=st.session_state.get("target_user", ""),
            placeholder="e.g., First-time visitor, Returning buyer, Enterprise admin",
            key="target_user",
        )
        key_metric = st.text_input(
            "Primary Metric",
            value=st.session_state.get("key_metric", ""),
            placeholder="e.g., CTR, Purchase Conversion Rate, Activation",
            key="key_metric",
        )
        current_val = st.text_input(
            "Current Value (baseline)",
            value=st.session_state.get("current_value", ""),
            placeholder="e.g., 25% or 0.25",
            key="current_value",
        )
        target_val = st.text_input(
            "Target Value",
            value=st.session_state.get("target_value", ""),
            placeholder="e.g., 28% or 0.28",
            key="target_value",
        )

        st.markdown("---")
        st.subheader("Success Criteria (optional)")
        conf = st.slider("Confidence Level (%)", min_value=80, max_value=99, value=int(st.session_state.get("confidence_level", 95)), step=1, key="confidence_level")
        power = st.slider("Statistical Power (%)", min_value=70, max_value=99, value=int(st.session_state.get("power", 80)), step=1, key="power")
        mde = st.slider("Minimum Detectable Effect (%)", min_value=0, max_value=50, value=int(st.session_state.get("MDE", 1)), step=1, key="MDE")

        st.markdown("---")
        st.subheader("Experiment Design (optional)")
        dau_cov = st.slider("DAU Coverage (%)", min_value=0, max_value=100, value=int(st.session_state.get("dau_coverage_percent", 0)), step=1, key="dau_coverage_percent")
        traffic = st.selectbox("Traffic Allocation", ["50/50", "90/10", "80/20", "70/30", "60/40"], index=0, key="traffic_allocation")

        # Persist context
        context = {
            "high_level_goal": high_level_goal,
            "product_type": product_type,
            "target_user": target_user,
            "key_metric": key_metric,
            "current_value": current_val,
            "target_value": target_val,
            "success_criteria": {
                "confidence_level": conf,
                "power": power,
                "MDE": mde,
            },
            "experiment_design": {
                "dau_coverage_percent": dau_cov,
                "traffic_allocation": traffic,
            },
        }
        st.session_state["sidebar_context"] = context

    # -------------------------
    # Step 2: Hypothesis → Auto-enrich
    # -------------------------
    st.header("Step 2: Write Your Hypothesis")

    with st.container(border=True):
        st.markdown("Write a single, clear hypothesis. We'll expand it into rationale, example implementation, and behavioral basis.")

        user_hypothesis_text = st.text_area(
            "Your hypothesis (one or two lines)",
            value=st.session_state.get("user_hypothesis_text", ""),
            height=100,
            key="user_hypothesis_text",
            help='Format example: "We believe that surfacing personalized product tiles on the home screen will increase CTR for first-time visitors."'
        )

        col_h1, col_h2 = st.columns([1, 1])
        with col_h1:
            enrich_clicked = st.button("Enrich Hypothesis with AI", type="primary", use_container_width=True)
        with col_h2:
            clear_clicked = st.button("Clear", use_container_width=True)

        if clear_clicked:
            st.session_state.pop("chosen_hypothesis", None)

        if enrich_clicked:
            if not user_hypothesis_text.strip():
                st.warning("Please enter a hypothesis first.")
            elif not PROMPT_ENGINE_AVAILABLE:
                st.error("Prompt engine not available. Cannot enrich hypothesis.")
            else:
                with st.spinner("Expanding hypothesis..."):
                    try:
                        # Build context from current sidebar inputs
                        context_for_h = st.session_state.get("sidebar_context", {})
                        raw = generate_hypothesis_details(user_hypothesis_text, context_for_h)
                        parsed = extract_json_from_text(raw)
                        if not parsed:
                            st.error("Could not parse enriched hypothesis JSON.")
                        else:
                            # Normalize keys + store
                            chosen = {
                                "hypothesis": sanitize_text(parsed.get("hypothesis", user_hypothesis_text)),
                                "rationale": sanitize_text(parsed.get("rationale", "")),
                                "example_implementation": sanitize_text(parsed.get("example_implementation", "")),
                                "behavioral_basis": sanitize_text(parsed.get("behavioral_basis", "")),
                            }
                            st.session_state["chosen_hypothesis"] = chosen
                    except Exception as e:
                        st.error(f"Error enriching hypothesis: {e}")

    # Show editable enriched hypothesis (if present)
    if "chosen_hypothesis" in st.session_state:
        st.markdown("#### Review & Edit")
        ch = st.session_state["chosen_hypothesis"]

        col1, col2 = st.columns(2)
        with col1:
            st.session_state["edited_hypothesis"] = st.text_area(
                "Hypothesis",
                value=ch.get("hypothesis", ""),
                height=100,
                key="edited_hypothesis"
            )
            st.session_state["edited_behavioral"] = st.text_input(
                "Behavioral Basis",
                value=ch.get("behavioral_basis", ""),
                key="edited_behavioral"
            )
        with col2:
            st.session_state["edited_rationale"] = st.text_area(
                "Rationale",
                value=ch.get("rationale", ""),
                height=120,
                key="edited_rationale"
            )
            st.session_state["edited_example"] = st.text_area(
                "Example Implementation",
                value=ch.get("example_implementation", ""),
                height=120,
                key="edited_example"
            )

        # Keep a clean object for Step 3
        st.session_state["chosen_hypothesis"] = {
            "hypothesis": sanitize_text(st.session_state.get("edited_hypothesis", ch.get("hypothesis", ""))),
            "rationale": sanitize_text(st.session_state.get("edited_rationale", ch.get("rationale", ""))),
            "example_implementation": sanitize_text(st.session_state.get("edited_example", ch.get("example_implementation", ""))),
            "behavioral_basis": sanitize_text(st.session_state.get("edited_behavioral", ch.get("behavioral_basis", ""))),
        }
    else:
        st.info("Enter a hypothesis above and click **Enrich Hypothesis with AI**.")

    st.markdown("---")

    # -------------------------
    # Step 3: Generate PRD (personalized)
    # -------------------------
    st.header("Step 3: Generate PRD")

    col_g1, col_g2 = st.columns([1, 2])
    with col_g1:
        owner = st.text_input("Experiment Owner", value=st.session_state.get("owner", ""), key="owner")
        team = st.text_input("Team", value=st.session_state.get("team", ""), key="team")
        exp_id = st.text_input("Experiment ID", value=st.session_state.get("experiment_id", generate_experiment_id()), key="experiment_id")

    with col_g2:
        st.caption("The PRD will be generated using all inputs above and your enriched hypothesis. You can refine later.")

    generate_clicked = st.button("Generate PRD", type="primary", use_container_width=True)

    if generate_clicked:
        if not PROMPT_ENGINE_AVAILABLE:
            st.error("Prompt engine not available.")
        elif "chosen_hypothesis" not in st.session_state:
            st.error("Please enrich a hypothesis in Step 2 first.")
        else:
            with st.spinner("Generating a personalized PRD..."):
                try:
                    # Build generation context
                    ctx = dict(st.session_state.get("sidebar_context", {}))
                    ctx["chosen_hypothesis"] = dict(st.session_state["chosen_hypothesis"])
                    ctx["metadata"] = {
                        "owner": owner,
                        "team": team,
                        "experiment_id": exp_id,
                        "title": f"{ctx.get('high_level_goal','Untitled')} — {ctx.get('product_type','')}"
                    }

                    # Use your existing engine function (we improved its prompt)
                    raw_plan = generate_experiment_plan(goal=ctx.get("high_level_goal", ""), context=ctx)
                    plan = extract_json_from_text(raw_plan)

                    # Fallback if parse failed
                    if not isinstance(plan, dict) or not plan:
                        plan = {}

                    # Merge metadata and chosen hypothesis if LLM omitted
                    plan.setdefault("metadata", {})
                    plan["metadata"].update(ctx["metadata"])
                    if not plan.get("hypotheses"):
                        plan["hypotheses"] = [ctx["chosen_hypothesis"]]

                    plan = sanitize_plan(plan)
                    st.session_state["experiment_plan"] = plan

                    # Optional validation
                    if PROMPT_ENGINE_AVAILABLE and "validate_experiment_plan" in globals():
                        try:
                            issues = validate_experiment_plan(plan)
                            if issues:
                                st.warning(f"Validation notes: {issues}")
                        except Exception:
                            pass

                except Exception as e:
                    st.error(f"Error generating PRD: {e}")

    # If we have a plan, show Preview + Editors
    plan = st.session_state.get("experiment_plan")
    if plan:
        with st.expander("🔎 Quick PRD Preview (JSON)", expanded=False):
            st.json(plan)

        with st.expander("🧾 PRD Document Preview (Markdown)", expanded=True):
            st.markdown(prd_to_markdown(plan))

        st.markdown("---")
        st.header("Step 4: Review & Edit")

        # Metadata Editor
        with st.expander("📇 Metadata", expanded=False):
            m = plan.get("metadata", {})
            m["title"] = st.text_input("Title", value=m.get("title", "Untitled Experiment"), key="md_title")
            m["team"] = st.text_input("Team", value=m.get("team", ""), key="md_team")
            m["owner"] = st.text_input("Owner", value=m.get("owner", ""), key="md_owner")
            m["experiment_id"] = st.text_input("Experiment ID", value=m.get("experiment_id", ""), key="md_eid")
            plan["metadata"] = m

        with st.expander("🧩 Problem Statement", expanded=False):
            plan["problem_statement"] = st.text_area(
                "Problem Statement",
                value=plan.get("problem_statement", ""),
                height=150,
                key="ps_text",
            )

        with st.expander("💡 Hypothesis", expanded=False):
            hyps = ensure_list(plan.get("hypotheses"))
            if not hyps:
                hyps = [{}]
            h0 = hyps[0] if isinstance(hyps[0], dict) else {}
            h0["hypothesis"] = st.text_area("Hypothesis", value=h0.get("hypothesis", ""), height=100, key="h_hyp")
            h0["rationale"] = st.text_area("Rationale", value=h0.get("rationale", ""), height=100, key="h_rat")
            h0["example_implementation"] = st.text_area("Example Implementation", value=h0.get("example_implementation", ""), height=100, key="h_ex")
            h0["behavioral_basis"] = st.text_input("Behavioral Basis", value=h0.get("behavioral_basis", ""), key="h_beh")
            plan["hypotheses"] = [h0]

        with st.expander("🛠️ Proposed Solution & Variants", expanded=False):
            plan["proposed_solution"] = st.text_area("Proposed Solution", value=plan.get("proposed_solution", ""), height=120, key="psol")
            vs = ensure_list(plan.get("variants"))
            if not vs:
                vs = [{"control": "", "variation": "", "notes": ""}]
            v0 = vs[0]
            v0["control"] = st.text_area("Control", value=v0.get("control", ""), height=80, key="v_ctrl")
            v0["variation"] = st.text_area("Variation", value=v0.get("variation", ""), height=80, key="v_var")
            v0["notes"] = st.text_input("Notes", value=v0.get("notes", ""), key="v_notes")
            plan["variants"] = [v0]

        with st.expander("📊 Metrics", expanded=False):
            ms = ensure_list(plan.get("metrics"))
            if not ms:
                ms = [{"name": "", "formula": "", "importance": "Primary"}]
            for i in range(len(ms)):
                ms[i]["name"] = st.text_input(f"Metric {i+1} Name", value=ms[i].get("name",""), key=f"m_name_{i}")
                ms[i]["formula"] = st.text_input(f"Metric {i+1} Formula", value=ms[i].get("formula",""), key=f"m_for_{i}")
                ms[i]["importance"] = st.selectbox(f"Metric {i+1} Importance", ["Primary", "Secondary"], index=0 if ms[i].get("importance","Primary")=="Primary" else 1, key=f"m_imp_{i}")
            plan["metrics"] = ms

        with st.expander("🛡️ Guardrail Metrics", expanded=False):
            gs = ensure_list(plan.get("guardrail_metrics"))
            if not gs:
                gs = [{"name": "", "direction": "Decrease", "threshold": ""}]
            for i in range(len(gs)):
                gs[i]["name"] = st.text_input(f"Guardrail {i+1} Name", value=gs[i].get("name",""), key=f"g_name_{i}")
                gs[i]["direction"] = st.selectbox(f"Guardrail {i+1} Direction", ["Increase","Decrease","No Change"], index=["Increase","Decrease","No Change"].index(gs[i].get("direction","Decrease")), key=f"g_dir_{i}")
                gs[i]["threshold"] = st.text_input(f"Guardrail {i+1} Threshold", value=gs[i].get("threshold",""), key=f"g_thr_{i}")
            plan["guardrail_metrics"] = gs

        with st.expander("🧪 Experiment Design & Rollout", expanded=False):
            ed = ensure_dict(plan.get("experiment_design"))
            col_ed1, col_ed2 = st.columns(2)
            with col_ed1:
                ed["traffic_allocation"] = st.selectbox("Traffic Allocation", ["50/50", "90/10", "80/20", "70/30", "60/40"], index=["50/50","90/10","80/20","70/30","60/40"].index(ed.get("traffic_allocation","50/50")), key="ed_alloc")
                ed["sample_size_per_variant"] = st.number_input("Sample Size per Variant", value=safe_int(ed.get("sample_size_per_variant", 0)), step=1, key="ed_sspv")
                ed["test_duration_days"] = st.number_input("Estimated Duration (days)", value=safe_int(ed.get("test_duration_days", 0)), step=1, key="ed_days")
                ed["dau_coverage_percent"] = st.number_input("DAU Coverage (%)", value=safe_float(ed.get("dau_coverage_percent", 0.0)), step=1.0, key="ed_dau")
            with col_ed2:
                ed["total_sample_size"] = st.number_input("Total Sample Size", value=safe_int(ed.get("total_sample_size", 0)), step=1, key="ed_tss")
                ed["power"] = st.number_input("Statistical Power (%)", value=safe_float(ed.get("power", 80.0)), step=1.0, key="ed_pow")
            plan["experiment_design"] = ed

        with st.expander("🎯 Success Criteria", expanded=False):
            sc = ensure_dict(plan.get("success_criteria"))
            sc["confidence_level"] = st.number_input("Confidence Level (%)", value=safe_float(sc.get("confidence_level", 95.0)), step=0.5, key="sc_conf")
            sc["power"] = st.number_input("Power (%)", value=safe_float(sc.get("power", 80.0)), step=0.5, key="sc_pow")
            sc["MDE"] = st.number_input("MDE (%)", value=safe_float(sc.get("MDE", 1.0)), step=0.1, key="sc_mde")
            sc["benchmark"] = st.text_input("Benchmark", value=sc.get("benchmark",""), key="sc_bench")
            sc["monitoring"] = st.text_input("Monitoring", value=sc.get("monitoring",""), key="sc_mon")
            plan["success_criteria"] = sc

        with st.expander("⚠️ Risks & Assumptions", expanded=False):
            rs = ensure_list(plan.get("risks_and_assumptions"))
            if not rs:
                rs = [{"risk":"", "severity":"Medium", "mitigation":""}]
            for i in range(len(rs)):
                rs[i]["risk"] = st.text_input(f"Risk {i+1}", value=rs[i].get("risk",""), key=f"r_risk_{i}")
                rs[i]["severity"] = st.selectbox(f"Severity {i+1}", ["High","Medium","Low"], index=["High","Medium","Low"].index(rs[i].get("severity","Medium")), key=f"r_sev_{i}")
                rs[i]["mitigation"] = st.text_input(f"Mitigation {i+1}", value=rs[i].get("mitigation",""), key=f"r_mit_{i}")
            plan["risks_and_assumptions"] = rs

        with st.expander("📘 Success & Learning Criteria", expanded=False):
            sl = ensure_dict(plan.get("success_learning_criteria"))
            sl["definition_of_success"] = st.text_input("Definition of Success", value=sl.get("definition_of_success",""), key="sl_def")
            sl["stopping_rules"] = st.text_input("Stopping Rules", value=sl.get("stopping_rules",""), key="sl_stop")
            sl["rollback_criteria"] = st.text_input("Rollback Criteria", value=sl.get("rollback_criteria",""), key="sl_roll")
            plan["success_learning_criteria"] = sl

        with st.expander("📐 Statistical Rationale", expanded=False):
            plan["statistical_rationale"] = st.text_area("Statistical Rationale", value=plan.get("statistical_rationale",""), height=120, key="sr_text")

        # Save back to session
        st.session_state["experiment_plan"] = sanitize_plan(plan)

        st.markdown("---")
        st.header("Step 5: AI Quality Check & Suggestions")

        qc_clicked = st.button("Get Final Suggestions", use_container_width=True)
        if qc_clicked and PROMPT_ENGINE_AVAILABLE:
            with st.spinner("AI is reviewing your PRD..."):
                try:
                    try_plan = st.session_state["experiment_plan"]
                    # Reuse engine validator if present; otherwise just echo a suggestion
                    if validate_experiment_plan:
                        issues = validate_experiment_plan(try_plan)
                    else:
                        issues = "No validator wired; consider clarifying guardrails."

                    st.success("Quality check complete.")
                    st.info(f"Suggestions / Findings:\n\n{issues}")
                except Exception as e:
                    st.error(f"Quality check failed: {e}")
        st.markdown("---")
        st.header("Step 6: Export")

        col_e1, col_e2, col_e3 = st.columns([1, 1, 1])
        with col_e1:
            if st.button("⬇️ Download PRD (Markdown)", use_container_width=True):
                md_text = prd_to_markdown(st.session_state["experiment_plan"])
                st.download_button(
                    "Download .md",
                    data=md_text,
                    file_name="experiment_prd.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

        with col_e2:
            if st.button("⬇️ Download PRD (JSON)", use_container_width=True):
                json_str = json.dumps(st.session_state["experiment_plan"], indent=2)
                st.download_button(
                    "Download .json",
                    data=json_str,
                    file_name="experiment_prd.json",
                    mime="application/json",
                    use_container_width=True,
                )

        with col_e3:
            if st.button("⬇️ Download PRD (PDF)", use_container_width=True):
                try:
                    pdf_bytes = generate_pdf_bytes_from_prd_dict(st.session_state["experiment_plan"])
                    st.download_button(
                        "Download .pdf",
                        data=pdf_bytes,
                        file_name="experiment_prd.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")


# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    main()
