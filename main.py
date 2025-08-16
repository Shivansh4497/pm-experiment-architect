# main.py — A/B Test Architect (full file, part 1/3)

import streamlit as st
import json
import os
import re
from typing import Any, Dict, Optional, Tuple, List, Union
from pydantic import BaseModel
from prompt_engine import (
    generate_experiment_plan,
    generate_hypothesis_details,
    run_quality_check,
)
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize
import numpy as np
import hashlib
from datetime import datetime
from io import BytesIO
import html

# Optional PDF export
REPORTLAB_AVAILABLE = False
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER
    REPORTLAB_AVAILABLE = True
except Exception as e:
    REPORTLAB_AVAILABLE = False
    print(f"[warn] ReportLab import failed: {e}")

# Optional DOCX export
DOCX_AVAILABLE = False
try:
    from docx import Document
    from docx.shared import Pt, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    DOCX_AVAILABLE = True
except Exception as e:
    DOCX_AVAILABLE = False
    print(f"[warn] python-docx import failed: {e}")


# =========================
# Pydantic Models (extended)
# =========================

class Hypothesis(BaseModel):
    hypothesis: str
    rationale: str
    example_implementation: str
    behavioral_basis: str

class Variant(BaseModel):
    control: str
    variation: str
    notes: Optional[str] = ""

class Metric(BaseModel):
    name: str
    formula: str
    importance: str  # e.g., Primary / Secondary

class Guardrail(BaseModel):
    name: str
    direction: Optional[str] = ""   # e.g., "must not decrease", "≤ threshold"
    threshold: Optional[str] = ""   # free-form threshold text

class Risk(BaseModel):
    risk: str
    severity: str
    mitigation: str

class SuccessCriteria(BaseModel):
    confidence_level: float  # %
    MDE: float               # %
    benchmark: str
    monitoring: str

class SuccessLearningCriteria(BaseModel):
    definition_of_success: str
    stopping_rules: str
    rollback_criteria: str

class ExperimentDesign(BaseModel):
    traffic_allocation: str              # e.g., "50/50", "10% holdout"
    sample_size_per_variant: Optional[int] = None
    total_sample_size: Optional[int] = None
    test_duration_days: Optional[float] = None
    dau_coverage_percent: Optional[float] = None

class Metadata(BaseModel):
    title: str
    team: str
    owner: str
    experiment_id: str

class ExperimentPlan(BaseModel):
    metadata: Metadata
    problem_statement: str
    hypotheses: List[Hypothesis]
    proposed_solution: str
    variants: List[Variant]
    metrics: List[Metric]
    guardrail_metrics: List[Guardrail]
    success_criteria: SuccessCriteria
    experiment_design: ExperimentDesign
    risks_and_assumptions: List[Risk]
    success_learning_criteria: SuccessLearningCriteria
    next_steps: List[str]
    statistical_rationale: str


# =========================
# Helpers & Sanitizers
# =========================

def sanitize_text(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip()

def html_sanitize(s: Any) -> str:
    return (
        sanitize_text(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _parse_value_from_text(text: str, default_unit: str) -> Tuple[Optional[float], str]:
    """
    Parses numeric value and unit from a free-form string.
    If unit not provided, falls back to default_unit.
    """
    text = sanitize_text(text)
    if not text:
        return None, default_unit
    # Match numeric + optional unit, e.g., "55.0 %", "60 INR", "12.4"
    m = re.match(r"^\s*([+-]?\d+(\.\d+)?)\s*([A-Za-z%$]*)\s*$", text)
    if not m:
        return None, default_unit
    val = float(m.group(1))
    unit = m.group(3) or default_unit
    return val, unit

def extract_json(raw: Union[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    # Find first JSON object in string
    json_match = re.search(r"\{.*\}", raw, re.S)
    if not json_match:
        return None
    try:
        return json.loads(json_match.group(0))
    except Exception:
        return None

def generate_experiment_id(title: str, owner: str) -> str:
    seed = f"{title}|{owner}|{datetime.utcnow().isoformat()}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12].upper()

def ensure_list(x: Any) -> List:
    return x if isinstance(x, list) else []

def ensure_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}

def sanitize_experiment_plan(raw_plan: Union[Dict[str, Any], None]) -> Dict[str, Any]:
    """
    Transform raw/malformed experiment plan data into a robust structure aligned with ExperimentPlan.
    Ensures all keys present and types are normalized.
    """
    if raw_plan is None:
        raw_plan = {}

    DEFAULT = {
        "metadata": {
            "title": "",
            "team": "",
            "owner": "",
            "experiment_id": "",
        },
        "problem_statement": "",
        "hypotheses": [],
        "proposed_solution": "",
        "variants": [],
        "metrics": [],
        "guardrail_metrics": [],
        "success_criteria": {
            "confidence_level": 95.0,
            "MDE": 5.0,
            "benchmark": "",
            "monitoring": "",
        },
        "experiment_design": {
            "traffic_allocation": "50/50",
            "sample_size_per_variant": None,
            "total_sample_size": None,
            "test_duration_days": None,
            "dau_coverage_percent": None,
        },
        "risks_and_assumptions": [],
        "success_learning_criteria": {
            "definition_of_success": "",
            "stopping_rules": "",
            "rollback_criteria": "",
        },
        "next_steps": [],
        "statistical_rationale": "",
    }

    plan = {**DEFAULT, **ensure_dict(raw_plan)}
    plan["metadata"] = {**DEFAULT["metadata"], **ensure_dict(plan.get("metadata", {}))}
    plan["success_criteria"] = {**DEFAULT["success_criteria"], **ensure_dict(plan.get("success_criteria", {}))}
    plan["experiment_design"] = {**DEFAULT["experiment_design"], **ensure_dict(plan.get("experiment_design", {}))}
    plan["success_learning_criteria"] = {**DEFAULT["success_learning_criteria"], **ensure_dict(plan.get("success_learning_criteria", {}))}

    # Normalize lists
    plan["hypotheses"] = ensure_list(plan.get("hypotheses", []))
    plan["variants"] = ensure_list(plan.get("variants", []))
    plan["metrics"] = ensure_list(plan.get("metrics", []))
    plan["guardrail_metrics"] = ensure_list(plan.get("guardrail_metrics", []))
    plan["risks_and_assumptions"] = ensure_list(plan.get("risks_and_assumptions", []))
    plan["next_steps"] = ensure_list(plan.get("next_steps", []))

    # Normalize child objects
    norm_hyps = []
    for h in plan["hypotheses"]:
        h = ensure_dict(h)
        norm_hyps.append({
            "hypothesis": sanitize_text(h.get("hypothesis", "")),
            "rationale": sanitize_text(h.get("rationale", "")),
            "example_implementation": sanitize_text(h.get("example_implementation", "")),
            "behavioral_basis": sanitize_text(h.get("behavioral_basis", "")),
        })
    plan["hypotheses"] = norm_hyps

    norm_vars = []
    for v in plan["variants"]:
        v = ensure_dict(v)
        norm_vars.append({
            "control": sanitize_text(v.get("control", "")),
            "variation": sanitize_text(v.get("variation", "")),
            "notes": sanitize_text(v.get("notes", "")),
        })
    plan["variants"] = norm_vars

    norm_metrics = []
    for m in plan["metrics"]:
        m = ensure_dict(m)
        norm_metrics.append({
            "name": sanitize_text(m.get("name", "")),
            "formula": sanitize_text(m.get("formula", "")),
            "importance": sanitize_text(m.get("importance", "")),
        })
    plan["metrics"] = norm_metrics

    norm_guardrails = []
    for g in plan["guardrail_metrics"]:
        g = ensure_dict(g)
        norm_guardrails.append({
            "name": sanitize_text(g.get("name", "")),
            "direction": sanitize_text(g.get("direction", "")),
            "threshold": sanitize_text(g.get("threshold", "")),
        })
    plan["guardrail_metrics"] = norm_guardrails

    norm_risks = []
    for r in plan["risks_and_assumptions"]:
        r = ensure_dict(r)
        sev = sanitize_text(r.get("severity", "Medium"))
        if sev.title() not in ["High", "Medium", "Low"]:
            sev = "Medium"
        norm_risks.append({
            "risk": sanitize_text(r.get("risk", "")) or sanitize_text(r.get("risks", "")),
            "severity": sev.title(),
            "mitigation": sanitize_text(r.get("mitigation", "")) or sanitize_text(r.get("mitigations", "")),
        })
    plan["risks_and_assumptions"] = norm_risks

    # Coerce types for success criteria & design
    try:
        plan["success_criteria"]["confidence_level"] = float(plan["success_criteria"].get("confidence_level", 95.0))
    except Exception:
        plan["success_criteria"]["confidence_level"] = 95.0
    try:
        plan["success_criteria"]["MDE"] = max(0.1, float(plan["success_criteria"].get("MDE", 5.0)))
    except Exception:
        plan["success_criteria"]["MDE"] = 5.0

    # Experiment Design
    ed = plan["experiment_design"]
    try:
        if ed.get("sample_size_per_variant") is not None:
            ed["sample_size_per_variant"] = int(float(ed.get("sample_size_per_variant")))
    except Exception:
        ed["sample_size_per_variant"] = None
    try:
        if ed.get("total_sample_size") is not None:
            ed["total_sample_size"] = int(float(ed.get("total_sample_size")))
    except Exception:
        ed["total_sample_size"] = None
    try:
        if ed.get("test_duration_days") is not None:
            ed["test_duration_days"] = float(ed.get("test_duration_days"))
    except Exception:
        ed["test_duration_days"] = None
    try:
        if ed.get("dau_coverage_percent") is not None:
            ed["dau_coverage_percent"] = float(ed.get("dau_coverage_percent"))
    except Exception:
        ed["dau_coverage_percent"] = None

    return plan


# =========================
# Stats & Export Utilities
# =========================

def calculate_sample_size(
    baseline: float,
    mde: float,
    alpha: float,
    power: float,
    num_variants: int,
    metric_type: str,
    std_dev: Optional[float] = None,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Returns (sample_per_variant, total_sample).
    baseline, mde expressed as percentage for conversion rate; for numeric metric baseline is absolute and std_dev is required.
    """
    try:
        mde_relative = float(mde) / 100.0
        if metric_type == "Conversion Rate":
            baseline_prop = float(baseline) / 100.0
            if baseline_prop <= 0:
                return None, None
            expected_prop = min(baseline_prop * (1 + mde_relative), 0.999)
            effect_size = proportion_effectsize(baseline_prop, expected_prop)
            if effect_size == 0:
                return None, None
            analysis = NormalIndPower()
            sample_size_per_variant = analysis.solve_power(
                effect_size=effect_size,
                alpha=alpha,
                power=power,
                alternative="two-sided",
            )
        elif metric_type == "Numeric Value":
            if std_dev is None or std_dev <= 0:
                return None, None
            mde_absolute = float(baseline) * mde_relative
            effect_size = mde_absolute / float(std_dev)
            if effect_size == 0:
                return None, None
            analysis = TTestIndPower()
            sample_size_per_variant = analysis.solve_power(
                effect_size=effect_size,
                alpha=alpha,
                power=power,
                alternative="two-sided",
            )
        else:
            return None, None

        if (
            sample_size_per_variant is None
            or sample_size_per_variant <= 0
            or not np.isfinite(sample_size_per_variant)
        ):
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


def generate_pdf_bytes_from_prd_dict(prd: Dict[str, Any], title: str = "Experiment PRD") -> Optional[bytes]:
    if not REPORTLAB_AVAILABLE:
        st.warning("PDF export requires ReportLab, which is not available.")
        return None

    prd = sanitize_experiment_plan(prd)
    buffer = BytesIO()

    try:
        doc = SimpleDocTemplate(
            buffer, pagesize=letter, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50
        )
        styles = getSampleStyleSheet()

        if "PRDTitle" not in styles:
            styles.add(
                ParagraphStyle(
                    name="PRDTitle",
                    fontSize=20,
                    leading=24,
                    spaceAfter=12,
                    alignment=TA_CENTER,
                )
            )
        if "SectionHeading" not in styles:
            styles.add(
                ParagraphStyle(
                    name="SectionHeading",
                    fontSize=14,
                    leading=18,
                    spaceBefore=12,
                    spaceAfter=6,
                    fontName="Helvetica-Bold",
                )
            )
        if "BodyText" not in styles:
            styles.add(ParagraphStyle(name="BodyText", fontSize=11, leading=14, spaceAfter=6))

        story = []
        story.append(Paragraph(pdf_sanitize(title), styles["PRDTitle"]))
        story.append(Spacer(1, 24))

        # 1. Metadata
        md = prd.get("metadata", {})
        meta_lines = [
            f"<b>Title:</b> {pdf_sanitize(md.get('title',''))}",
            f"<b>Team:</b> {pdf_sanitize(md.get('team',''))}",
            f"<b>Owner:</b> {pdf_sanitize(md.get('owner',''))}",
            f"<b>Experiment ID:</b> {pdf_sanitize(md.get('experiment_id',''))}",
        ]
        story.append(Paragraph("1. Experiment Title & Metadata", styles["SectionHeading"]))
        for ln in meta_lines:
            story.append(Paragraph(ln, styles["BodyText"]))
        story.append(Spacer(1, 12))

        # 2. Problem Statement
        story.append(Paragraph("2. Problem Statement", styles["SectionHeading"]))
        story.append(Paragraph(pdf_sanitize(prd.get("problem_statement", "")), styles["BodyText"]))
        story.append(Spacer(1, 12))

        # 3. Hypothesis (selected one or list)
        story.append(Paragraph("3. Hypothesis", styles["SectionHeading"]))
        for idx, h in enumerate(prd.get("hypotheses", []), start=1):
            story.append(Paragraph(f"<b>Hypothesis {idx}:</b> {pdf_sanitize(h.get('hypothesis',''))}", styles["BodyText"]))
            story.append(Paragraph(f"<b>Rationale:</b> {pdf_sanitize(h.get('rationale',''))}", styles["BodyText"]))
            story.append(Paragraph(f"<b>Example:</b> {pdf_sanitize(h.get('example_implementation',''))}", styles["BodyText"]))
            story.append(Paragraph(f"<b>Behavioral Basis:</b> {pdf_sanitize(h.get('behavioral_basis',''))}", styles["BodyText"]))
            story.append(Spacer(1, 8))
        story.append(Spacer(1, 12))

        # 4. Proposed Solution & Variants
        story.append(Paragraph("4. Proposed Solution & Variants", styles["SectionHeading"]))
        story.append(Paragraph(f"<b>Solution:</b> {pdf_sanitize(prd.get('proposed_solution',''))}", styles["BodyText"]))
        for v in prd.get("variants", []):
            story.append(Paragraph(f"<b>Control:</b> {pdf_sanitize(v.get('control',''))}", styles["BodyText"]))
            story.append(Paragraph(f"<b>Variation:</b> {pdf_sanitize(v.get('variation',''))}", styles["BodyText"]))
            if v.get("notes"):
                story.append(Paragraph(f"<b>Notes:</b> {pdf_sanitize(v.get('notes',''))}", styles["BodyText"]))
            story.append(Spacer(1, 6))
        story.append(Spacer(1, 12))

        # 5. Success Metrics & Guardrails
        story.append(Paragraph("5. Success Metrics & Guardrails", styles["SectionHeading"]))
        metrics_data = [["Name", "Formula", "Importance"]]
        for m in prd.get("metrics", []):
            metrics_data.append([pdf_sanitize(m.get("name","")), pdf_sanitize(m.get("formula","")), pdf_sanitize(m.get("importance",""))])
        if len(metrics_data) > 1:
            table = Table(metrics_data, colWidths=[2.2*inch, 3.0*inch, 1.3*inch])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#EEF2FF")),
                ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#052a4a")),
                ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                ("ALIGN", (0,0), (-1,0), "CENTER"),
                ("GRID", (0,0), (-1,-1), 0.25, colors.gray),
                ("VALIGN", (0,0), (-1,-1), "TOP"),
            ]))
            story.append(table)
            story.append(Spacer(1, 8))

        guard_data = [["Guardrail Metric", "Direction", "Threshold"]]
        for g in prd.get("guardrail_metrics", []):
            guard_data.append([pdf_sanitize(g.get("name","")), pdf_sanitize(g.get("direction","")), pdf_sanitize(g.get("threshold",""))])
        if len(guard_data) > 1:
            gtable = Table(guard_data, colWidths=[2.5*inch, 2.2*inch, 1.8*inch])
            gtable.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#FEE2E2")),
                ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#7f1d1d")),
                ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                ("ALIGN", (0,0), (-1,0), "CENTER"),
                ("GRID", (0,0), (-1,-1), 0.25, colors.gray),
                ("VALIGN", (0,0), (-1,-1), "TOP"),
            ]))
            story.append(gtable)
            story.append(Spacer(1, 12))

        # 6. Experiment Design & Rollout Plan
        story.append(Paragraph("6. Experiment Design & Rollout Plan", styles["SectionHeading"]))
        ed = prd.get("experiment_design", {})
        ed_lines = [
            f"<b>Traffic Allocation:</b> {pdf_sanitize(ed.get('traffic_allocation','50/50'))}",
            f"<b>Sample Size / Variant:</b> {pdf_sanitize(ed.get('sample_size_per_variant',''))}",
            f"<b>Total Sample Size:</b> {pdf_sanitize(ed.get('total_sample_size',''))}",
            f"<b>Test Duration (days):</b> {pdf_sanitize(ed.get('test_duration_days',''))}",
            f"<b>DAU Coverage (%):</b> {pdf_sanitize(ed.get('dau_coverage_percent',''))}",
        ]
        for ln in ed_lines:
            story.append(Paragraph(ln, styles["BodyText"]))
        story.append(Spacer(1, 12))

        # 7. Risks & Mitigation
        story.append(Paragraph("7. Risks & Mitigation", styles["SectionHeading"]))
        for r in prd.get("risks_and_assumptions", []):
            story.append(Paragraph(f"<b>Risk:</b> {pdf_sanitize(r.get('risk',''))}", styles["BodyText"]))
            story.append(Paragraph(f"<b>Severity:</b> {pdf_sanitize(r.get('severity',''))}", styles["BodyText"]))
            story.append(Paragraph(f"<b>Mitigation:</b> {pdf_sanitize(r.get('mitigation',''))}", styles["BodyText"]))
            story.append(Spacer(1, 6))
        story.append(Spacer(1, 12))

        # 8. Success & Learning Criteria
        story.append(Paragraph("8. Success & Learning Criteria", styles["SectionHeading"]))
        sl = prd.get("success_learning_criteria", {})
        story.append(Paragraph(f"<b>Definition of Success:</b> {pdf_sanitize(sl.get('definition_of_success',''))}", styles["BodyText"]))
        story.append(Paragraph(f"<b>Stopping Rules:</b> {pdf_sanitize(sl.get('stopping_rules',''))}", styles["BodyText"]))
        story.append(Paragraph(f"<b>Rollback Criteria:</b> {pdf_sanitize(sl.get('rollback_criteria',''))}", styles["BodyText"]))
        story.append(Spacer(1, 12))

        # Success Criteria & Rationale
        story.append(Paragraph("9. Statistical Criteria & Rationale", styles["SectionHeading"]))
        sc = prd.get("success_criteria", {})
        story.append(Paragraph(f"<b>Confidence Level:</b> {pdf_sanitize(sc.get('confidence_level',''))}%", styles["BodyText"]))
        story.append(Paragraph(f"<b>MDE:</b> {pdf_sanitize(sc.get('MDE',''))}%", styles["BodyText"]))
        story.append(Paragraph(f"<b>Benchmark:</b> {pdf_sanitize(sc.get('benchmark',''))}", styles["BodyText"]))
        story.append(Paragraph(f"<b>Monitoring:</b> {pdf_sanitize(sc.get('monitoring',''))}", styles["BodyText"]))
        story.append(Paragraph(f"<b>Rationale:</b> {pdf_sanitize(prd.get('statistical_rationale',''))}", styles["BodyText"]))
        story.append(Spacer(1, 12))

        # 10. Next Steps
        story.append(Paragraph("10. Next Steps", styles["SectionHeading"]))
        for s in prd.get("next_steps", []):
            story.append(Paragraph(f"• {pdf_sanitize(s)}", styles["BodyText"]))

        doc.build(story)
        buffer.seek(0)
        return buffer.read()
    except Exception as e:
        st.error(f"PDF generation failed: {str(e)}")
        return None


def generate_docx_bytes_from_prd_dict(prd: Dict[str, Any], title: str = "Experiment PRD") -> Optional[bytes]:
    if not DOCX_AVAILABLE:
        st.warning("DOCX export requires python-docx, which is not available.")
        return None

    prd = sanitize_experiment_plan(prd)
    bio = BytesIO()
    try:
        doc = Document()

        # Title
        h = doc.add_heading(title, level=0)
        h.alignment = WD_ALIGN_PARAGRAPH.CENTER

        def add_heading(txt, level=1):
            doc.add_heading(txt, level=level)

        def add_par(txt):
            p = doc.add_paragraph(txt)
            p_format = p.paragraph_format
            p_format.space_after = Pt(6)

        # 1. Metadata
        add_heading("1. Experiment Title & Metadata", 1)
        md = prd.get("metadata", {})
        add_par(f"Title: {md.get('title','')}")
        add_par(f"Team: {md.get('team','')}")
        add_par(f"Owner: {md.get('owner','')}")
        add_par(f"Experiment ID: {md.get('experiment_id','')}")
        # 2. Problem
        add_heading("2. Problem Statement", 1)
        add_par(prd.get("problem_statement", ""))

        # 3. Hypothesis
        add_heading("3. Hypothesis", 1)
        for i, h in enumerate(prd.get("hypotheses", []), start=1):
            add_par(f"Hypothesis {i}: {h.get('hypothesis','')}")
            add_par(f"Rationale: {h.get('rationale','')}")
            add_par(f"Example: {h.get('example_implementation','')}")
            add_par(f"Behavioral Basis: {h.get('behavioral_basis','')}")

        # 4. Proposed Solution & Variants
        add_heading("4. Proposed Solution & Variants", 1)
        add_par(f"Solution: {prd.get('proposed_solution','')}")
        for v in prd.get("variants", []):
            add_par(f"Control: {v.get('control','')}")
            add_par(f"Variation: {v.get('variation','')}")
            if v.get("notes"):
                add_par(f"Notes: {v.get('notes','')}")

        # 5. Success Metrics & Guardrails
        add_heading("5. Success Metrics & Guardrails", 1)
        if prd.get("metrics"):
            t = doc.add_table(rows=1, cols=3)
            hdr = t.rows[0].cells
            hdr[0].text = "Name"
            hdr[1].text = "Formula"
            hdr[2].text = "Importance"
            for m in prd["metrics"]:
                row = t.add_row().cells
                row[0].text = m.get("name","")
                row[1].text = m.get("formula","")
                row[2].text = m.get("importance","")
        if prd.get("guardrail_metrics"):
            doc.add_paragraph("")  # spacing
            t2 = doc.add_table(rows=1, cols=3)
            hdr2 = t2.rows[0].cells
            hdr2[0].text = "Guardrail Metric"
            hdr2[1].text = "Direction"
            hdr2[2].text = "Threshold"
            for g in prd["guardrail_metrics"]:
                row = t2.add_row().cells
                row[0].text = g.get("name","")
                row[1].text = g.get("direction","")
                row[2].text = g.get("threshold","")

        # 6. Experiment Design & Rollout Plan
        add_heading("6. Experiment Design & Rollout Plan", 1)
        ed = prd.get("experiment_design", {})
        add_par(f"Traffic Allocation: {ed.get('traffic_allocation','')}")
        add_par(f"Sample Size / Variant: {ed.get('sample_size_per_variant','')}")
        add_par(f"Total Sample Size: {ed.get('total_sample_size','')}")
        add_par(f"Test Duration (days): {ed.get('test_duration_days','')}")
        add_par(f"DAU Coverage (%): {ed.get('dau_coverage_percent','')}")

        # 7. Risks & Mitigation
        add_heading("7. Risks & Mitigation", 1)
        for r in prd.get("risks_and_assumptions", []):
            add_par(f"Risk: {r.get('risk','')}")
            add_par(f"Severity: {r.get('severity','')}")
            add_par(f"Mitigation: {r.get('mitigation','')}")

        # 8. Success & Learning Criteria
        add_heading("8. Success & Learning Criteria", 1)
        sl = prd.get("success_learning_criteria", {})
        add_par(f"Definition of Success: {sl.get('definition_of_success','')}")
        add_par(f"Stopping Rules: {sl.get('stopping_rules','')}")
        add_par(f"Rollback Criteria: {sl.get('rollback_criteria','')}")

        # 9. Statistical Criteria & Rationale
        add_heading("9. Statistical Criteria & Rationale", 1)
        sc = prd.get("success_criteria", {})
        add_par(f"Confidence Level: {sc.get('confidence_level','')}%")
        add_par(f"MDE: {sc.get('MDE','')}%")
        add_par(f"Benchmark: {sc.get('benchmark','')}")
        add_par(f"Monitoring: {sc.get('monitoring','')}")
        add_par(f"Rationale: {prd.get('statistical_rationale','')}")

        # 10. Next Steps
        add_heading("10. Next Steps", 1)
        for s in prd.get("next_steps", []):
            add_par(f"• {s}")

        doc.save(bio)
        bio.seek(0)
        return bio.read()
    except Exception as e:
        st.error(f"DOCX generation failed: {str(e)}")
        return None
# main.py — A/B Test Architect (full file, part 2/3)

# =========================
# Streamlit App — Layout & State
# =========================

st.set_page_config(
    page_title="A/B Test Architect",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
ss = st.session_state
for key, default in {
    "context": {},
    "goal": "",
    "raw_plan_json": None,
    "plan": None,
    "hypothesis_options": [],
    "selected_hypothesis_index": None,
    "custom_hypothesis": "",
    "quality_result": None,
    "export_filename_base": "experiment_prd",
}.items():
    if key not in ss:
        ss[key] = default


# =========================
# Sidebar — User Inputs (Step 1)
# =========================

with st.sidebar:
    st.markdown("### 👤 User Inputs")
    st.caption("Provide the context to generate hypotheses and the PRD.")

    # High-level business goal
    goal = st.text_input(
        "High-Level Business Goal",
        value=ss.get("goal", ""),
        placeholder="e.g., Improve user engagement on the homepage",
    )

    product_type = st.selectbox(
        "Product Type",
        options=["Mobile App", "E-commerce Website", "SaaS Dashboard", "Web App", "Other"],
        index=0 if not ss["context"].get("type") else
               ["Mobile App", "E-commerce Website", "SaaS Dashboard", "Web App", "Other"].index(
                   ss["context"].get("type", "Mobile App")
               ),
    )

    persona = st.text_input(
        "Target User Persona",
        value=ss["context"].get("user_persona", ""),
        placeholder="e.g., First-time visitor",
    )

    metric_to_impact = st.selectbox(
        "Key Metric to Impact",
        options=["Click-through Rate", "Purchase Conversion Rate", "Time on Page", "Average Session Time", "Retention", "Other"],
        index=0,
    )

    current_value = st.text_input(
        "Current Value",
        value=ss["context"].get("current_value", ""),
        placeholder='e.g., "25% CTR" or "5.2 minutes"',
    )
    target_value = st.text_input(
        "Target Value",
        value=ss["context"].get("target_value", ""),
        placeholder='e.g., "28% CTR" or "6.0 minutes"',
    )

    # Advanced metric details for stats
    metric_type = st.selectbox(
        "Metric Type (for Sample Size)",
        options=["Conversion Rate", "Numeric Value"],
        help="Choose how your key metric behaves statistically.",
        index=0 if ss["context"].get("metric_type", "Conversion Rate") == "Conversion Rate" else 1,
    )

    std_dev = st.text_input(
        "Std Dev (if Numeric Value)",
        value=str(ss["context"].get("std_dev", "")),
        placeholder="Required only for Numeric Value",
    )

    dau_coverage = st.slider(
        "Experiment DAU Coverage (%)",
        min_value=1,
        max_value=100,
        value=int(ss["context"].get("dau_coverage_percent", 50) or 50),
        step=1,
    )

    # Owner & team for metadata
    st.markdown("---")
    owner = st.text_input("Experiment Owner", value=ss["context"].get("owner", ""))
    team = st.text_input("Product Team", value=ss["context"].get("team", ""))

    st.markdown("---")
    st.caption("Quality & Stats")
    confidence = st.slider("Confidence Level (%)", 80, 99, int(ss.get("plan", {}).get("success_criteria", {}).get("confidence_level", 95) or 95))
    mde = st.slider("MDE — Minimum Detectable Effect (%)", 1, 50, int(ss.get("plan", {}).get("success_criteria", {}).get("MDE", 5) or 5))
    power = st.slider("Statistical Power (%)", 60, 99, 80)

    # Persist context in session
    ss["goal"] = goal
    ss["context"] = {
        "type": product_type,
        "user_persona": persona,
        "exact_metric": metric_to_impact,
        "metric_type": metric_type,
        "current_value": current_value,
        "target_value": target_value,
        "metric_unit": "%",
        "users": "N/A",
        "notes": "",
        "std_dev": float(std_dev) if std_dev.strip() else None,
        "dau_coverage_percent": dau_coverage,
        "team": team,
        "owner": owner,
        "confidence_level": confidence,
        "MDE": mde,
        "power": power,
        "strategic_goal": goal,
    }

st.title("🧪 A/B Test Architect")
st.caption("From idea → standardized, export-ready PRD. Fast. Consistent. Credible.")


# =========================
# Hypotheses (Step 2)
# =========================

st.subheader("Step 2: Generate and Select a Hypothesis")

col_h1, col_h2, col_h3 = st.columns([1, 1, 1])
with col_h1:
    if st.button("⚡ Generate Hypotheses", use_container_width=True):
        if not ss["goal"] or not ss["context"].get("type"):
            st.error("Please fill the sidebar: Business Goal and Product Type at minimum.")
        else:
            # Build three quick hypothesis options by calling `generate_hypothesis_details` with seeds
            seeds = [
                f"If we redesign the primary call-to-action on the {ss['context']['type']} to improve visibility, then {metric_to_impact} will increase because users notice it faster.",
                f"If we reduce friction in the key user flow for the {ss['context']['type']}, then {metric_to_impact} will improve because fewer users drop off.",
                f"If we add contextual hints or nudges for the {persona}, then {metric_to_impact} will increase because they understand the next best action.",
            ]
            options = []
            for s in seeds:
                try:
                    detail_json = generate_hypothesis_details(s, ss["context"])
                    obj = extract_json(detail_json) or {}
                    options.append(obj)
                except Exception as e:
                    options.append({
                        "hypothesis": s,
                        "rationale": f"Generation failed, fallback used. Error: {str(e)}",
                        "example_implementation": "N/A",
                        "behavioral_basis": "N/A",
                    })
            ss["hypothesis_options"] = options
            ss["selected_hypothesis_index"] = None
            ss["custom_hypothesis"] = ""

with col_h2:
    if st.button("📝 Write Custom Hypothesis", use_container_width=True):
        ss["custom_hypothesis"] = ss.get("custom_hypothesis", "") or "If we [change], then [outcome], because [reason]."
with col_h3:
    clear_h = st.button("🗑️ Clear Hypotheses", use_container_width=True)
    if clear_h:
        ss["hypothesis_options"] = []
        ss["selected_hypothesis_index"] = None
        ss["custom_hypothesis"] = ""

# Display hypothesis options
if ss["hypothesis_options"]:
    st.markdown("#### AI-Generated Options")
    cols = st.columns(3)
    for i, hyp in enumerate(ss["hypothesis_options"]):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"**Hypothesis {i+1}**")
                st.write(hyp.get("hypothesis", ""))
                st.caption(f"Behavioral Basis: {hyp.get('behavioral_basis','')}")
                if st.button(f"Select Hypothesis {i+1}", key=f"select_h_{i}", use_container_width=True):
                    ss["selected_hypothesis_index"] = i

# Custom hypothesis editor
if ss["custom_hypothesis"]:
    st.markdown("#### Custom Hypothesis")
    ss["custom_hypothesis"] = st.text_area(
        "Edit your custom hypothesis",
        value=ss["custom_hypothesis"],
        height=100,
    )
    if st.button("Expand Custom Hypothesis", type="primary"):
        expanded = generate_hypothesis_details(ss["custom_hypothesis"], ss["context"])
        expanded_obj = extract_json(expanded) or {
            "hypothesis": ss["custom_hypothesis"],
            "rationale": "",
            "example_implementation": "",
            "behavioral_basis": ""
        }
        # Put it into the options list as the first item
        ss["hypothesis_options"] = [expanded_obj] + ss["hypothesis_options"]
        ss["selected_hypothesis_index"] = 0

# Selection summary
if ss["selected_hypothesis_index"] is not None:
    st.success(f"Selected: Hypothesis {ss['selected_hypothesis_index'] + 1}")


# =========================
# PRD Generation (Step 3)
# =========================

st.subheader("Step 3: Generate the Full PRD")

col_g1, col_g2 = st.columns([1, 1])
with col_g1:
    default_title = ss["context"].get("strategic_goal", "Untitled Experiment")
    prd_title = st.text_input("Experiment Title", value=default_title)
with col_g2:
    ss["export_filename_base"] = st.text_input(
        "Export Filename (base)",
        value=ss["export_filename_base"],
        help="Used for JSON/PDF/DOCX downloads.",
    )

def _selected_hypothesis_obj() -> Optional[Dict[str, Any]]:
    if ss["selected_hypothesis_index"] is None:
        return None
    if ss["selected_hypothesis_index"] < 0 or ss["selected_hypothesis_index"] >= len(ss["hypothesis_options"]):
        return None
    return ss["hypothesis_options"][ss["selected_hypothesis_index"]]

if st.button("🧩 Generate PRD", type="primary", use_container_width=True):
    if not ss["goal"]:
        st.error("Please provide a High-Level Business Goal in the sidebar.")
    else:
        # Prepare context for model
        context = dict(ss["context"])
        context["strategic_goal"] = ss["goal"]
        context["metric_to_improve"] = ss["context"].get("exact_metric")
        # Generate
        raw_json = generate_experiment_plan(ss["goal"], context)
        obj = extract_json(raw_json) or {}

        # Insert selected hypothesis at top if exists
        sel = _selected_hypothesis_obj()
        if sel:
            obj_hyps = ensure_list(obj.get("hypotheses", []))
            obj["hypotheses"] = [sel] + obj_hyps

        # Enrich with metadata defaults
        md = obj.get("metadata", {}) or {}
        md["title"] = prd_title or md.get("title") or default_title
        md["team"] = ss["context"].get("team", "")
        md["owner"] = ss["context"].get("owner", "")
        md["experiment_id"] = md.get("experiment_id") or generate_experiment_id(md["title"], md["owner"])
        obj["metadata"] = md

        # Guardrail list default if missing
        if "guardrail_metrics" not in obj:
            obj["guardrail_metrics"] = []

        # Experiment design defaults
        ed = obj.get("experiment_design", {}) or {}
        ed.setdefault("traffic_allocation", "50/50")
        ed.setdefault("dau_coverage_percent", ss["context"].get("dau_coverage_percent", 50))
        obj["experiment_design"] = ed

        # Persist
        ss["raw_plan_json"] = obj
        ss["plan"] = sanitize_experiment_plan(obj)
        ss["quality_result"] = None

        st.success("PRD generated. Scroll down to review and edit.")
# main.py — A/B Test Architect (full file, part 3/3)

# =========================
# Step 4: Review & Edit PRD
# =========================

if ss["plan"]:
    st.subheader("Step 4: Review & Edit")

    plan = ss["plan"]

    # Experiment Metadata
    with st.expander("🆔 Experiment Metadata", expanded=True):
        plan["metadata"]["title"] = st.text_input("Title", value=plan["metadata"].get("title", ""))
        plan["metadata"]["owner"] = st.text_input("Owner", value=plan["metadata"].get("owner", ""))
        plan["metadata"]["team"] = st.text_input("Team", value=plan["metadata"].get("team", ""))
        plan["metadata"]["experiment_id"] = st.text_input("Experiment ID", value=plan["metadata"].get("experiment_id", ""))

    # Problem Statement
    with st.expander("❓ Problem Statement", expanded=True):
        plan["problem_statement"] = st.text_area("Problem Statement", value=plan.get("problem_statement", ""), height=120)

    # Hypotheses
    with st.expander("💡 Hypotheses", expanded=True):
        new_hypotheses = []
        for i, h in enumerate(plan.get("hypotheses", [])):
            st.markdown(f"**Hypothesis {i+1}**")
            hyp = {}
            hyp["hypothesis"] = st.text_area(f"Hypothesis {i+1}", value=h.get("hypothesis", ""), key=f"h_hyp_{i}")
            hyp["rationale"] = st.text_area(f"Rationale {i+1}", value=h.get("rationale", ""), key=f"h_rat_{i}")
            hyp["example_implementation"] = st.text_area(f"Example Implementation {i+1}", value=h.get("example_implementation", ""), key=f"h_impl_{i}")
            hyp["behavioral_basis"] = st.text_input(f"Behavioral Basis {i+1}", value=h.get("behavioral_basis", ""), key=f"h_beh_{i}")
            new_hypotheses.append(hyp)
        plan["hypotheses"] = new_hypotheses

    # Proposed Solution & Variants
    with st.expander("🧪 Proposed Solution & Variants", expanded=True):
        plan["variants"]["control"] = st.text_area("Control", value=plan["variants"].get("control", ""), height=80)
        plan["variants"]["treatment"] = st.text_area("Variation", value=plan["variants"].get("treatment", ""), height=80)

    # Success Metrics & Guardrails
    with st.expander("📊 Success Metrics & Guardrails", expanded=True):
        plan["metrics"]["primary"] = st.text_input("Primary Metric", value=plan["metrics"].get("primary", ""))
        plan["metrics"]["secondary"] = st.text_area("Secondary Metrics (comma-separated)", value=", ".join(plan["metrics"].get("secondary", [])))
        plan["guardrail_metrics"] = st.text_area("Guardrail Metrics (comma-separated)", value=", ".join(plan.get("guardrail_metrics", [])))

    # Experiment Design
    with st.expander("🧭 Experiment Design & Rollout Plan", expanded=True):
        plan["experiment_design"]["sample_size_required"] = st.number_input(
            "Sample Size Required",
            value=plan["experiment_design"].get("sample_size_required", 0),
            step=100,
        )
        plan["experiment_design"]["duration"] = st.text_input("Duration", value=plan["experiment_design"].get("duration", ""))
        plan["experiment_design"]["traffic_allocation"] = st.text_input("Traffic Allocation", value=plan["experiment_design"].get("traffic_allocation", "50/50"))
        plan["experiment_design"]["dau_coverage_percent"] = st.slider(
            "DAU Coverage (%)",
            1, 100,
            value=int(plan["experiment_design"].get("dau_coverage_percent", 50)),
        )

    # Risks & Mitigation
    with st.expander("⚠️ Risks & Mitigation", expanded=True):
        risks_str = ""
        for r in plan.get("risks", []):
            risks_str += f"- {r}\n"
        plan["risks"] = st.text_area("Risks & Mitigation (Markdown list)", value=risks_str, height=120).splitlines()

    # Success & Learning Criteria
    with st.expander("🎯 Success & Learning Criteria", expanded=True):
        plan.setdefault("success_learning_criteria", {})
        plan["success_learning_criteria"]["definition_of_success"] = st.text_area(
            "Definition of Success",
            value=plan["success_learning_criteria"].get("definition_of_success", ""),
            height=80,
        )
        plan["success_learning_criteria"]["stopping_rules"] = st.text_area(
            "Stopping Rules",
            value=plan["success_learning_criteria"].get("stopping_rules", ""),
            height=80,
        )
        plan["success_learning_criteria"]["rollback_criteria"] = st.text_area(
            "Rollback Criteria",
            value=plan["success_learning_criteria"].get("rollback_criteria", ""),
            height=80,
        )

    ss["plan"] = plan


# =========================
# Step 5: AI Quality Check
# =========================

if ss["plan"]:
    st.subheader("Step 5: AI Quality Check & Final Suggestions")

    if st.button("🔍 Run AI Quality Check", type="secondary", use_container_width=True):
        quality = validate_experiment_plan(ss["plan"])
        ss["quality_result"] = quality

    if ss.get("quality_result"):
        st.markdown("#### Quality Suggestions")
        st.info(ss["quality_result"])


# =========================
# Step 6: Finalize & Export
# =========================

if ss["plan"]:
    st.subheader("Step 6: Finalize & Export")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            "⬇️ Export JSON",
            data=json.dumps(ss["plan"], indent=2),
            file_name=f"{ss['export_filename_base']}.json",
            mime="application/json",
            use_container_width=True,
        )

    with col2:
        pdf_bytes = generate_pdf_bytes_from_prd_dict(ss["plan"])
        st.download_button(
            "⬇️ Export PDF",
            data=pdf_bytes,
            file_name=f"{ss['export_filename_base']}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    with col3:
        docx_bytes = generate_docx_bytes_from_prd_dict(ss["plan"])
        st.download_button(
            "⬇️ Export DOCX",
            data=docx_bytes,
            file_name=f"{ss['export_filename_base']}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )
