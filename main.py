# main.py — A/B Test Architect (Complete, robust, single-file Streamlit app)
# Part 1/5: imports, flags, models, sanitizers, and helpers

import streamlit as st
import json
import os
import re
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from io import BytesIO

# Optional heavy deps — import defensively
REPORTLAB_AVAILABLE = False
DOCX_AVAILABLE = False
STATSMODELS_AVAILABLE = False

try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

try:
    from docx import Document
    from docx.shared import Pt, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    DOCX_AVAILABLE = True
except Exception:
    DOCX_AVAILABLE = False

try:
    from statsmodels.stats.power import NormalIndPower, TTestIndPower
    from statsmodels.stats.proportion import proportion_effectsize
    import numpy as np
    STATSMODELS_AVAILABLE = True
except Exception:
    # Keep numpy available for non-stat calculations
    try:
        import numpy as np
    except Exception:
        np = None
    STATSMODELS_AVAILABLE = False

# Try to import prompt_engine functions (defensive: app should stay up if not configured)
PROMPT_ENGINE_AVAILABLE = True
try:
    from prompt_engine import (
    generate_experiment_plan,
    generate_hypothesis_details,
    validate_experiment_plan,
    generate_hypotheses

    )
except Exception as e:
    PROMPT_ENGINE_AVAILABLE = False
    _prompt_import_error = e

# Pydantic for validation (optional but useful)
try:
    from pydantic import BaseModel, ValidationError
    Pydantic_AVAILABLE = True
except Exception:
    Pydantic_AVAILABLE = False


# -------------------------
# Data models (Pydantic)
# -------------------------
if Pydantic_AVAILABLE:
    class HypothesisModel(BaseModel):
        hypothesis: str
        rationale: str
        example_implementation: str
        behavioral_basis: str

    class VariantModel(BaseModel):
        control: str
        variation: str
        notes: Optional[str] = ""

    class MetricModel(BaseModel):
        name: str
        formula: str
        importance: str

    class GuardrailModel(BaseModel):
        name: str
        direction: Optional[str] = ""
        threshold: Optional[str] = ""

    class RiskModel(BaseModel):
        risk: str
        severity: str
        mitigation: str

    class SuccessCriteriaModel(BaseModel):
        confidence_level: float
        MDE: float
        benchmark: str
        monitoring: str

    class SuccessLearningModel(BaseModel):
        definition_of_success: str
        stopping_rules: str
        rollback_criteria: str

    class ExperimentDesignModel(BaseModel):
        traffic_allocation: str
        sample_size_per_variant: Optional[int] = None
        total_sample_size: Optional[int] = None
        test_duration_days: Optional[float] = None
        dau_coverage_percent: Optional[float] = None

    class MetadataModel(BaseModel):
        title: str
        team: str
        owner: str
        experiment_id: str

    class ExperimentPlanModel(BaseModel):
        metadata: MetadataModel
        problem_statement: str
        hypotheses: List[HypothesisModel]
        proposed_solution: str
        variants: List[VariantModel]
        metrics: List[MetricModel]
        guardrail_metrics: List[GuardrailModel]
        success_criteria: SuccessCriteriaModel
        experiment_design: ExperimentDesignModel
        risks_and_assumptions: List[RiskModel]
        success_learning_criteria: SuccessLearningModel
        next_steps: List[str]
        statistical_rationale: str


# -------------------------
# Utility helpers
# -------------------------
def sanitize_text(x: Any) -> str:
    if x is None:
        return ""
    try:
        s = str(x)
    except Exception:
        return ""
    s = s.replace("\r", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ensure_list(x: Any) -> list:
    return x if isinstance(x, list) else []


def ensure_dict(x: Any) -> dict:
    return x if isinstance(x, dict) else {}


def generate_experiment_id(title: str, owner: str) -> str:
    seed = f"{title}|{owner}|{datetime.utcnow().isoformat()}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12].upper()


def extract_json_from_text(raw: Any) -> Optional[Dict[str, Any]]:
    """
    Try to extract a JSON object from raw LLM output or a dict string.
    Returns dict or None — never raises.
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    text = str(raw)
    # First try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try to extract first {...} block
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            # Last resort: replace single quotes with double quotes carefully
            t = m.group(0)
            t2 = re.sub(r"(?<=[:\[\{,\s])'([^']*?)'(?=[,\]\}\s])", r'"\1"', t)
            try:
                return json.loads(t2)
            except Exception:
                return None
    return None
# main.py — Part 2/5: statistics, sample-size, exporters (PDF/DOCX), sanitizers, and validation helpers

# -------------------------
# Statistics / sample size
# -------------------------
def calculate_sample_size(
    baseline: Optional[float],
    mde_pct: Optional[float],
    alpha: float,
    power: float,
    num_variants: int,
    metric_type: str,
    std_dev: Optional[float] = None,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Return tuple (sample_per_variant, total_sample).
    - baseline: for conversion rate (as percent, e.g., 25.0 for 25%) or numeric baseline for numeric metric.
    - mde_pct: MDE as percent (e.g., 5 for 5%).
    - alpha: significance level (e.g., 0.05)
    - power: e.g., 0.8
    - metric_type: "Conversion Rate" or "Numeric Value"
    - std_dev: required for numeric metrics (same units as baseline)
    """
    # Defensive: ensure stats libs exist
    if not STATSMODELS_AVAILABLE:
        return None, None

    try:
        if baseline is None or mde_pct is None:
            return None, None

        if metric_type == "Conversion Rate":
            # convert percents to proportions
            p1 = float(baseline) / 100.0
            if p1 <= 0 or p1 >= 1:
                return None, None
            mde_rel = float(mde_pct) / 100.0
            p2 = min(max(p1 * (1 + mde_rel), 1e-6), 0.999999)
            effect_size = proportion_effectsize(p1, p2)
            if not np.isfinite(effect_size) or effect_size == 0:
                return None, None
            analysis = NormalIndPower()
            per_variant = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative="two-sided")
        elif metric_type == "Numeric Value":
            if std_dev is None or std_dev <= 0:
                return None, None
            mde_rel = float(mde_pct) / 100.0
            mde_abs = float(baseline) * mde_rel
            if mde_abs == 0:
                return None, None
            effect_size = mde_abs / float(std_dev)
            analysis = TTestIndPower()
            per_variant = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative="two-sided")
        else:
            return None, None

        if per_variant is None or not np.isfinite(per_variant) or per_variant <= 0:
            return None, None

        per_variant_int = int(np.ceil(per_variant))
        total = int(np.ceil(per_variant_int * int(num_variants)))
        return per_variant_int, total
    except Exception:
        return None, None


# -------------------------
# PDF export
# -------------------------
def generate_pdf_bytes_from_plan(plan: Dict[str, Any]) -> Optional[bytes]:
    """
    Create a PDF from a sanitized plan. Returns bytes or None.
    Requires ReportLab.
    """
    if not REPORTLAB_AVAILABLE:
        return None
    plan = sanitize_plan(plan)

    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
        styles = getSampleStyleSheet()
        # Add custom styles if absent
        if "PRDTitle" not in styles:
            styles.add(ParagraphStyle(name="PRDTitle", fontSize=18, leading=22, spaceAfter=12, alignment=TA_CENTER))
        if "SectionHeading" not in styles:
            styles.add(ParagraphStyle(name="SectionHeading", fontSize=13, leading=16, spaceBefore=10, spaceAfter=6))
        if "BodyText" not in styles:
            styles.add(ParagraphStyle(name="BodyText", fontSize=10, leading=14, spaceAfter=6))

        story = []
        md = plan.get("metadata", {})
        title = md.get("title") or "Experiment PRD"
        story.append(Paragraph(title, styles["PRDTitle"]))
        story.append(Spacer(1, 8))

        # Metadata
        story.append(Paragraph("Experiment Metadata", styles["SectionHeading"]))
        story.append(Paragraph(f"Team: {md.get('team','')}", styles["BodyText"]))
        story.append(Paragraph(f"Owner: {md.get('owner','')}", styles["BodyText"]))
        story.append(Paragraph(f"Experiment ID: {md.get('experiment_id','')}", styles["BodyText"]))
        story.append(Spacer(1, 6))

        # Problem statement
        story.append(Paragraph("Problem Statement", styles["SectionHeading"]))
        story.append(Paragraph(pdf_safe(plan.get("problem_statement","")), styles["BodyText"]))
        story.append(Spacer(1, 6))

        # Proposed solution
        story.append(Paragraph("Proposed Solution", styles["SectionHeading"]))
        story.append(Paragraph(pdf_safe(plan.get("proposed_solution","")), styles["BodyText"]))
        story.append(Spacer(1, 6))

        # Hypotheses
        story.append(Paragraph("Hypotheses", styles["SectionHeading"]))
        for i, h in enumerate(plan.get("hypotheses", []), start=1):
            story.append(Paragraph(f"Hypothesis {i}: {pdf_safe(h.get('hypothesis',''))}", styles["BodyText"]))
            story.append(Paragraph(f"Rationale: {pdf_safe(h.get('rationale',''))}", styles["BodyText"]))
            story.append(Paragraph(f"Example: {pdf_safe(h.get('example_implementation',''))}", styles["BodyText"]))
            story.append(Paragraph(f"Behavioral Basis: {pdf_safe(h.get('behavioral_basis',''))}", styles["BodyText"]))
            story.append(Spacer(1, 4))

        # Variants
        story.append(Paragraph("Variants", styles["SectionHeading"]))
        for v in plan.get("variants", []):
            story.append(Paragraph(f"Control: {pdf_safe(v.get('control',''))}", styles["BodyText"]))
            story.append(Paragraph(f"Variation: {pdf_safe(v.get('variation',''))}", styles["BodyText"]))
            if v.get("notes"):
                story.append(Paragraph(f"Notes: {pdf_safe(v.get('notes',''))}", styles["BodyText"]))
            story.append(Spacer(1, 4))

        # Metrics + Guardrails
        story.append(Paragraph("Metrics", styles["SectionHeading"]))
        metrics = plan.get("metrics", [])
        if metrics:
            data = [["Name", "Formula", "Importance"]]
            for m in metrics:
                data.append([pdf_safe(m.get("name","")), pdf_safe(m.get("formula","")), pdf_safe(m.get("importance",""))])
            tbl = Table(data, colWidths=[2.2*inch, 3.0*inch, 1.2*inch])
            tbl.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.25, colors.gray)]))
            story.append(tbl)
        story.append(Spacer(1, 6))

        story.append(Paragraph("Guardrail Metrics", styles["SectionHeading"]))
        guards = plan.get("guardrail_metrics", [])
        for g in guards:
            story.append(Paragraph(f"{pdf_safe(g.get('name',''))} — {pdf_safe(g.get('direction',''))} {pdf_safe(g.get('threshold',''))}", styles["BodyText"]))
        story.append(Spacer(1, 6))

        # Experiment design
        story.append(Paragraph("Experiment Design & Rollout", styles["SectionHeading"]))
        ed = plan.get("experiment_design", {})
        story.append(Paragraph(f"Traffic Allocation: {pdf_safe(ed.get('traffic_allocation',''))}", styles["BodyText"]))
        story.append(Paragraph(f"Sample Size per Variant: {pdf_safe(ed.get('sample_size_per_variant',''))}", styles["BodyText"]))
        story.append(Paragraph(f"Total Sample Size: {pdf_safe(ed.get('total_sample_size',''))}", styles["BodyText"]))
        story.append(Paragraph(f"Estimated Duration (days): {pdf_safe(ed.get('test_duration_days',''))}", styles["BodyText"]))
        story.append(Spacer(1, 6))

        # Risks
        story.append(Paragraph("Risks & Mitigations", styles["SectionHeading"]))
        for r in plan.get("risks_and_assumptions", []):
            story.append(Paragraph(f"Risk: {pdf_safe(r.get('risk',''))}", styles["BodyText"]))
            story.append(Paragraph(f"Severity: {pdf_safe(r.get('severity',''))}", styles["BodyText"]))
            story.append(Paragraph(f"Mitigation: {pdf_safe(r.get('mitigation',''))}", styles["BodyText"]))
            story.append(Spacer(1, 4))

        # Success & learning
        story.append(Paragraph("Success & Learning Criteria", styles["SectionHeading"]))
        sl = plan.get("success_learning_criteria", {})
        story.append(Paragraph(f"Definition of Success: {pdf_safe(sl.get('definition_of_success',''))}", styles["BodyText"]))
        story.append(Paragraph(f"Stopping Rules: {pdf_safe(sl.get('stopping_rules',''))}", styles["BodyText"]))
        story.append(Paragraph(f"Rollback Criteria: {pdf_safe(sl.get('rollback_criteria',''))}", styles["BodyText"]))
        story.append(Spacer(1, 6))

        # Statistical rationale
        story.append(Paragraph("Statistical Rationale & Success Criteria", styles["SectionHeading"]))
        sc = plan.get("success_criteria", {})
        story.append(Paragraph(f"Confidence Level: {pdf_safe(sc.get('confidence_level',''))}%", styles["BodyText"]))
        story.append(Paragraph(f"MDE: {pdf_safe(sc.get('MDE',''))}%", styles["BodyText"]))
        story.append(Paragraph(f"Benchmark: {pdf_safe(sc.get('benchmark',''))}", styles["BodyText"]))
        story.append(Paragraph(f"Monitoring: {pdf_safe(sc.get('monitoring',''))}", styles["BodyText"]))
        story.append(Paragraph(pdf_safe(plan.get("statistical_rationale","")), styles["BodyText"]))

        doc.build(story)
        buffer.seek(0)
        return buffer.read()
    except Exception:
        return None


def pdf_safe(x: Any) -> str:
    if x is None:
        return ""
    try:
        s = str(x)
    except Exception:
        return ""
    # Minimal escaping for reportlab flowables
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# -------------------------
# DOCX export
# -------------------------
def generate_docx_bytes_from_plan(plan: Dict[str, Any]) -> Optional[bytes]:
    """
    Generate a DOCX document from the plan. Returns bytes or None.
    Requires python-docx.
    """
    if not DOCX_AVAILABLE:
        return None
    plan = sanitize_plan(plan)
    try:
        bio = BytesIO()
        doc = Document()
        md = plan.get("metadata", {})
        doc.add_heading(md.get("title", "Experiment PRD"), level=0)

        def add_h(txt):
            doc.add_heading(txt, level=1)

        def add_p(txt):
            p = doc.add_paragraph(txt)
            p_format = p.paragraph_format
            p_format.space_after = Pt(6)

        add_h("Experiment Metadata")
        add_p(f"Team: {md.get('team','')}")
        add_p(f"Owner: {md.get('owner','')}")
        add_p(f"Experiment ID: {md.get('experiment_id','')}")

        add_h("Problem Statement")
        add_p(plan.get("problem_statement",""))

        add_h("Proposed Solution")
        add_p(plan.get("proposed_solution",""))

        add_h("Hypotheses")
        for i, h in enumerate(plan.get("hypotheses", []), start=1):
            add_p(f"Hypothesis {i}: {h.get('hypothesis','')}")
            add_p(f"Rationale: {h.get('rationale','')}")
            add_p(f"Example: {h.get('example_implementation','')}")
            add_p(f"Behavioral Basis: {h.get('behavioral_basis','')}")

        add_h("Variants")
        for v in plan.get("variants", []):
            add_p(f"Control: {v.get('control','')}")
            add_p(f"Variation: {v.get('variation','')}")
            if v.get("notes"):
                add_p(f"Notes: {v.get('notes','')}")

        add_h("Metrics")
        if plan.get("metrics"):
            table = doc.add_table(rows=1, cols=3)
            hdr = table.rows[0].cells
            hdr[0].text = "Name"
            hdr[1].text = "Formula"
            hdr[2].text = "Importance"
            for m in plan["metrics"]:
                row = table.add_row().cells
                row[0].text = m.get("name","")
                row[1].text = m.get("formula","")
                row[2].text = m.get("importance","")

        add_h("Guardrail Metrics")
        for g in plan.get("guardrail_metrics", []):
            add_p(f"{g.get('name','')} — {g.get('direction','')} {g.get('threshold','')}")

        add_h("Experiment Design & Rollout")
        ed = plan.get("experiment_design", {})
        add_p(f"Traffic Allocation: {ed.get('traffic_allocation','')}")
        add_p(f"Sample Size / Variant: {ed.get('sample_size_per_variant','')}")
        add_p(f"Total Sample Size: {ed.get('total_sample_size','')}")
        add_p(f"Estimated Duration (days): {ed.get('test_duration_days','')}")

        add_h("Risks & Mitigation")
        for r in plan.get("risks_and_assumptions", []):
            add_p(f"Risk: {r.get('risk','')}")
            add_p(f"Severity: {r.get('severity','')}")
            add_p(f"Mitigation: {r.get('mitigation','')}")

        add_h("Success & Learning Criteria")
        sl = plan.get("success_learning_criteria", {})
        add_p(f"Definition of Success: {sl.get('definition_of_success','')}")
        add_p(f"Stopping Rules: {sl.get('stopping_rules','')}")
        add_p(f"Rollback Criteria: {sl.get('rollback_criteria','')}")

        add_h("Statistical Rationale & Success Criteria")
        sc = plan.get("success_criteria", {})
        add_p(f"Confidence Level: {sc.get('confidence_level','')}%")
        add_p(f"MDE: {sc.get('MDE','')}%")
        add_p(f"Benchmark: {sc.get('benchmark','')}")
        add_p(f"Monitoring: {sc.get('monitoring','')}")
        add_p(plan.get("statistical_rationale",""))

        doc.save(bio)
        bio.seek(0)
        return bio.read()
    except Exception:
        return None


# -------------------------
# Sanitization / normalization
# -------------------------
def sanitize_plan(raw_plan: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normalize incoming plan into the app's canonical schema (dict).
    Returns a safe, fully populated dict that exporters and UI expect.
    """
    if raw_plan is None:
        raw_plan = {}
    p = ensure_dict(raw_plan)

    # Metadata
    meta = ensure_dict(p.get("metadata", {}))
    metadata = {
        "title": sanitize_text(meta.get("title", "")),
        "team": sanitize_text(meta.get("team", "")),
        "owner": sanitize_text(meta.get("owner", "")),
        "experiment_id": sanitize_text(meta.get("experiment_id", "")),
    }
    if not metadata["experiment_id"]:
        metadata["experiment_id"] = generate_experiment_id(metadata["title"] or "Untitled", metadata["owner"] or "unknown")

    # Problem statement & proposed solution
    problem_statement = sanitize_text(p.get("problem_statement", ""))
    proposed_solution = sanitize_text(p.get("proposed_solution", ""))

    # Hypotheses
    hyps_raw = ensure_list(p.get("hypotheses", []))
    hypotheses = []
    for h in hyps_raw:
        d = ensure_dict(h)
        hypotheses.append({
            "hypothesis": sanitize_text(d.get("hypothesis", "")),
            "rationale": sanitize_text(d.get("rationale", "")),
            "example_implementation": sanitize_text(d.get("example_implementation", "")),
            "behavioral_basis": sanitize_text(d.get("behavioral_basis", "")),
        })

    # Variants
    vars_raw = ensure_list(p.get("variants", []))
    variants = []
    if vars_raw:
        for v in vars_raw:
            d = ensure_dict(v)
            variants.append({
                "control": sanitize_text(d.get("control", "")),
                "variation": sanitize_text(d.get("variation", "")),
                "notes": sanitize_text(d.get("notes", "")),
            })
    else:
        # default placeholders
        variants = [{"control": "", "variation": "", "notes": ""}]

    # Metrics
    metrics_raw = ensure_list(p.get("metrics", []))
    metrics = []
    if metrics_raw:
        for m in metrics_raw:
            md = ensure_dict(m)
            metrics.append({
                "name": sanitize_text(md.get("name", "")),
                "formula": sanitize_text(md.get("formula", "")),
                "importance": sanitize_text(md.get("importance", "Primary")),
            })
    else:
        metrics = []

    # Guardrails
    guards_raw = ensure_list(p.get("guardrail_metrics", []))
    guardrail_metrics = []
    for g in guards_raw:
        gd = ensure_dict(g)
        guardrail_metrics.append({
            "name": sanitize_text(gd.get("name", "")),
            "direction": sanitize_text(gd.get("direction", "")),
            "threshold": sanitize_text(gd.get("threshold", "")),
        })

    # Success criteria
    sc_raw = ensure_dict(p.get("success_criteria", {}))
    try:
        confidence_level = float(sc_raw.get("confidence_level", 95.0))
    except Exception:
        confidence_level = 95.0
    try:
        MDE = float(sc_raw.get("MDE", 5.0))
        MDE = max(0.1, MDE)
    except Exception:
        MDE = 5.0
    success_criteria = {
        "confidence_level": confidence_level,
        "MDE": MDE,
        "benchmark": sanitize_text(sc_raw.get("benchmark", "")),
        "monitoring": sanitize_text(sc_raw.get("monitoring", "")),
    }

    # Experiment design
    ed_raw = ensure_dict(p.get("experiment_design", {}))
    ed = {
        "traffic_allocation": sanitize_text(ed_raw.get("traffic_allocation", "50/50")),
        "sample_size_per_variant": try_int(ed_raw.get("sample_size_per_variant")),
        "total_sample_size": try_int(ed_raw.get("total_sample_size")),
        "test_duration_days": try_float(ed_raw.get("test_duration_days")),
        "dau_coverage_percent": try_float(ed_raw.get("dau_coverage_percent")),
    }

    # Risks
    risks_raw = ensure_list(p.get("risks_and_assumptions", []))
    risks = []
    for r in risks_raw:
        rd = ensure_dict(r)
        sev = sanitize_text(rd.get("severity", "Medium")).title()
        if sev not in ["High", "Medium", "Low"]:
            sev = "Medium"
        risks.append({
            "risk": sanitize_text(rd.get("risk", rd.get("risks", ""))),
            "severity": sev,
            "mitigation": sanitize_text(rd.get("mitigation", rd.get("mitigations", ""))),
        })

    # Success & learning
    sl_raw = ensure_dict(p.get("success_learning_criteria", {}))
    success_learning = {
        "definition_of_success": sanitize_text(sl_raw.get("definition_of_success", "")),
        "stopping_rules": sanitize_text(sl_raw.get("stopping_rules", "")),
        "rollback_criteria": sanitize_text(sl_raw.get("rollback_criteria", "")),
    }

    # Next steps & statistical rationale
    next_steps = [sanitize_text(x) for x in ensure_list(p.get("next_steps", []))]
    statistical_rationale = sanitize_text(p.get("statistical_rationale", ""))

    canonical = {
        "metadata": metadata,
        "problem_statement": problem_statement,
        "proposed_solution": proposed_solution,
        "hypotheses": hypotheses,
        "variants": variants,
        "metrics": metrics,
        "guardrail_metrics": guardrail_metrics,
        "success_criteria": success_criteria,
        "experiment_design": ed,
        "risks_and_assumptions": risks,
        "success_learning_criteria": success_learning,
        "next_steps": next_steps,
        "statistical_rationale": statistical_rationale,
    }

    # If sample sizes absent but inputs present, attempt to compute basic sample size
    try_auto_compute_sample_size(canonical)

    return canonical


# -------------------------
# small helpers for coercion
# -------------------------
def try_int(x: Any) -> Optional[int]:
    try:
        if x is None or x == "":
            return None
        return int(float(x))
    except Exception:
        return None


def try_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def try_auto_compute_sample_size(plan: Dict[str, Any]) -> None:
    """
    If plan lacks sample sizes but has enough inputs (baseline, MDE, metric type), compute and fill experiment_design fields.
    """
    ed = plan.get("experiment_design", {})
    sc = plan.get("success_criteria", {})
    # Only compute if stats libs present and values needed exist
    if STATSMODELS_AVAILABLE:
        # baseline attempt: try infer from problem_statement or metrics — best effort
        baseline = None
        # check if a metric has a formula like "25%" or first metric name has a number — best-effort parse
        for m in plan.get("metrics", []):
            # look for numeric in metric name or formula
            nm = m.get("name","")
            fm = m.get("formula","")
            for text in [nm, fm]:
                found = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%?", text)
                if found:
                    baseline = float(found.group(1))
                    break
            if baseline is not None:
                break

        mde = sc.get("MDE", 5.0)
        conf = sc.get("confidence_level", 95.0)
        power = sc.get("power", 80.0) if sc.get("power") else 80.0

        metric_type = "Conversion Rate"
        # check indicators in plan
        if any("rate" in (m.get("name","").lower()) or "%" in (m.get("formula","")) for m in plan.get("metrics", [])):
            metric_type = "Conversion Rate"
        else:
            # fallback: if statistical_rationale mentions "mean" or "std", assume numeric
            if "mean" in plan.get("statistical_rationale","").lower() or "standard deviation" in plan.get("statistical_rationale","").lower():
                metric_type = "Numeric Value"

        if ed.get("sample_size_per_variant") is None and baseline is not None:
            per_var, total = calculate_sample_size(
                baseline=baseline,
                mde_pct=mde,
                alpha=1 - (conf / 100.0),
                power=(power / 100.0),
                num_variants=max(2, len(plan.get("variants", [])) or 2),
                metric_type=metric_type,
                std_dev=None,
            )
            if per_var:
                ed["sample_size_per_variant"] = per_var
                ed["total_sample_size"] = total

    plan["experiment_design"] = ed


# -------------------------
# Validation (Local + LLM-assisted)
# -------------------------
def local_validate_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform a quick local validation and return a dict with keys:
    - is_valid: bool
    - issues: list[str]
    - suggestions: list[str]
    """
    issues = []
    suggestions = []
    p = sanitize_plan(plan)

    # Basic checks
    if not p.get("metadata", {}).get("title"):
        issues.append("Missing experiment title.")
        suggestions.append("Add a concise experiment title in metadata.")
    if not p.get("problem_statement"):
        issues.append("Problem statement is empty.")
        suggestions.append("Clarify the metric-driven problem the experiment addresses.")
    if not p.get("hypotheses"):
        issues.append("No hypotheses found.")
        suggestions.append("Create at least one clear hypothesis in the 'If X then Y because Z' format.")
    if not p.get("metrics"):
        issues.append("No metrics defined.")
        suggestions.append("Define a primary metric and at least one secondary or guardrail metric.")
    sc = p.get("success_criteria", {})
    if sc.get("MDE", None) is None:
        issues.append("MDE not provided.")
        suggestions.append("Provide an MDE (minimum detectable effect) to compute sample sizes.")
    # sample size checks
    ed = p.get("experiment_design", {})
    if ed.get("sample_size_per_variant") is None:
        suggestions.append("Sample size per variant could not be computed automatically; provide inputs or check metric formatting.")
    # return result
    return {"is_valid": len(issues) == 0, "issues": issues, "suggestions": suggestions}


def validate_plan_with_llm(plan: Dict[str, Any]) -> str:
    """
    If prompt_engine is available, call its validate_experiment_plan; otherwise return a synthesized local validation message.
    """
    if PROMPT_ENGINE_AVAILABLE:
        try:
            return validate_experiment_plan(plan)
        except Exception:
            # fallback to local message
            local = local_validate_plan(plan)
            return json.dumps(local, indent=2)
    else:
        # provide local validation summary
        local = local_validate_plan(plan)
        return json.dumps(local, indent=2)
# main.py — Part 3/5: Streamlit UI (layout, inputs, Step 1–2 UI)

# -------------------------
# Streamlit App
# -------------------------

def main():
    st.set_page_config(page_title="Experiment PRD Architect", layout="wide")
    st.title("🧪 Experiment PRD Architect")
    st.write(
        "Generate standardized, detailed A/B test PRDs with AI assistance. "
        "This tool streamlines the process from idea to finalized document."
    )

    # -------------------------
    # Sidebar: User Inputs
    # -------------------------
    st.sidebar.header("Step 1: Provide Experiment Context")
    high_level_goal = st.sidebar.text_input("High-Level Business Goal", "")
    product_type = st.sidebar.selectbox(
        "Product Type",
        ["", "Mobile App", "E-commerce Website", "SaaS Dashboard", "Other"]
    )
    if product_type == "Other":
        product_type = st.sidebar.text_input("Specify Product Type", "")

    target_user = st.sidebar.text_input("Target User Persona", "")
    key_metric = st.sidebar.text_input("Key Metric to Impact", "")

    current_val = st.sidebar.text_input("Current Value (e.g., 25% CTR or 5.2 minutes)", "")
    target_val = st.sidebar.text_input("Target Value (e.g., 28% CTR or 6.0 minutes)", "")

    exp_owner = st.sidebar.text_input("Your Name (Experiment Owner)", "")
    exp_team = st.sidebar.text_input("Team/Org", "")

    # Sidebar advanced experiment design inputs
    with st.sidebar.expander("Advanced Experiment Design Inputs", expanded=False):
        confidence_level = st.slider("Confidence Level (%)", 80, 99, 95)
        power = st.slider("Statistical Power (%)", 50, 99, 80)
        mde = st.number_input("Minimum Detectable Effect (MDE %)", value=5.0, step=0.1)
        dau_coverage = st.number_input("Experiment DAU Coverage (%)", value=50.0, step=1.0)

    # Shared experiment context
    experiment_context = {
        "high_level_goal": high_level_goal,
        "product_type": product_type,
        "target_user": target_user,
        "key_metric": key_metric,
        "current_value": current_val,
        "target_value": target_val,
        "metadata": {"owner": exp_owner, "team": exp_team},
        "success_criteria": {
            "confidence_level": confidence_level,
            "power": power,
            "MDE": mde,
        },
        "experiment_design": {"dau_coverage_percent": dau_coverage},
    }

    # -------------------------
    # Step 2: Generate Hypotheses
    # -------------------------
    st.header("Step 2: Generate & Select a Hypothesis")
    if st.button("Generate Hypotheses", use_container_width=True):
        with st.spinner("Generating hypotheses..."):
            if PROMPT_ENGINE_AVAILABLE:
                try:
                    hyps = generate_hypotheses(experiment_context)
                except Exception as e:
                    st.error(f"Error generating hypotheses: {e}")
                    hyps = []
            else:
                hyps = []
        if not hyps:
            st.warning("No hypotheses generated. Please refine your inputs.")
        else:
            st.session_state["generated_hypotheses"] = hyps

    if "generated_hypotheses" in st.session_state:
        st.subheader("Choose or Edit a Hypothesis")
        hyps = st.session_state["generated_hypotheses"]
        options = [h.get("hypothesis", f"Hypothesis {i+1}") for i, h in enumerate(hyps)]
        selected_idx = st.radio("Select Hypothesis", list(range(len(options))), format_func=lambda i: options[i])
        chosen_hypothesis = hyps[selected_idx]
        st.session_state["chosen_hypothesis"] = chosen_hypothesis

        st.text_area(
            "Edit Hypothesis",
            value=chosen_hypothesis.get("hypothesis", ""),
            key="edited_hypothesis",
            height=100,
        )

        st.text_area(
            "Rationale",
            value=chosen_hypothesis.get("rationale", ""),
            key="edited_rationale",
            height=80,
        )
        st.text_area(
            "Example Implementation",
            value=chosen_hypothesis.get("example_implementation", ""),
            key="edited_example",
            height=80,
        )
        st.text_area(
            "Behavioral Basis",
            value=chosen_hypothesis.get("behavioral_basis", ""),
            key="edited_behavioral",
            height=80,
        )
# main.py — Part 4/5: Step 3–4 UI (Generate PRD, Review & Edit)

    # -------------------------
    # Step 3: Generate PRD
    # -------------------------
    st.header("Step 3: Generate Full PRD")
    if st.button("Generate PRD", use_container_width=True):
        with st.spinner("Generating PRD..."):
            if PROMPT_ENGINE_AVAILABLE:
                try:
                    edited_hypothesis = {
                        "hypothesis": st.session_state.get("edited_hypothesis", ""),
                        "rationale": st.session_state.get("edited_rationale", ""),
                        "example_implementation": st.session_state.get("edited_example", ""),
                        "behavioral_basis": st.session_state.get("edited_behavioral", ""),
                    }
                    exp_id = generate_experiment_id(high_level_goal, exp_owner)
                    context_with_hyp = {**experiment_context, "chosen_hypothesis": edited_hypothesis}
                    raw = generate_experiment_plan(context_with_hyp)
                    plan = extract_json_from_text(raw)
                    if not plan:
                        st.error("Could not parse generated PRD JSON.")
                        plan = {}
                    plan.setdefault("metadata", {})
                    plan["metadata"].update({
                        "title": sanitize_text(high_level_goal),
                        "team": sanitize_text(exp_team),
                        "owner": sanitize_text(exp_owner),
                        "experiment_id": exp_id,
                    })
                    st.session_state["experiment_plan"] = sanitize_experiment_plan(plan)
                except Exception as e:
                    st.error(f"Error generating PRD: {e}")
            else:
                st.error("Prompt engine not available. Cannot generate PRD.")

    # -------------------------
    # Step 4: Review & Edit
    # -------------------------
    st.header("Step 4: Review & Edit PRD")
    if "experiment_plan" in st.session_state:
        plan = st.session_state["experiment_plan"]

        st.subheader("Metadata")
        plan["metadata"]["title"] = st.text_input("Experiment Title", plan["metadata"].get("title", ""))
        plan["metadata"]["team"] = st.text_input("Team", plan["metadata"].get("team", ""))
        plan["metadata"]["owner"] = st.text_input("Owner", plan["metadata"].get("owner", ""))
        st.write(f"Experiment ID: `{plan['metadata'].get('experiment_id','')}`")

        st.subheader("Problem Statement")
        plan["problem_statement"] = st.text_area("Problem Statement", plan.get("problem_statement", ""), height=120)

        st.subheader("Hypothesis")
        if plan.get("hypotheses"):
            hyp = plan["hypotheses"][0]
            hyp["hypothesis"] = st.text_area("Hypothesis", hyp.get("hypothesis", ""), height=80)
            hyp["rationale"] = st.text_area("Rationale", hyp.get("rationale", ""), height=80)
            hyp["example_implementation"] = st.text_area("Example Implementation", hyp.get("example_implementation", ""), height=80)
            hyp["behavioral_basis"] = st.text_area("Behavioral Basis", hyp.get("behavioral_basis", ""), height=80)

        st.subheader("Proposed Solution")
        plan["proposed_solution"] = st.text_area("Proposed Solution", plan.get("proposed_solution", ""), height=100)

        st.subheader("Variants")
        new_variants = []
        for i, var in enumerate(plan.get("variants", [])):
            st.markdown(f"**Variant {i+1}**")
            control = st.text_area(f"Control {i+1}", var.get("control",""), key=f"variant_control_{i}")
            variation = st.text_area(f"Variation {i+1}", var.get("variation",""), key=f"variant_variation_{i}")
            notes = st.text_area(f"Notes {i+1}", var.get("notes",""), key=f"variant_notes_{i}")
            new_variants.append({"control": control, "variation": variation, "notes": notes})
        plan["variants"] = new_variants

        st.subheader("Metrics")
        new_metrics = []
        for i, m in enumerate(plan.get("metrics", [])):
            name = st.text_input(f"Metric {i+1} Name", m.get("name",""), key=f"metric_name_{i}")
            formula = st.text_input(f"Metric {i+1} Formula", m.get("formula",""), key=f"metric_formula_{i}")
            importance = st.selectbox(
                f"Metric {i+1} Importance",
                ["Primary","Secondary","Diagnostic"],
                index=0 if m.get("importance","Primary")=="Primary" else (1 if m.get("importance")=="Secondary" else 2),
                key=f"metric_importance_{i}"
            )
            new_metrics.append({"name": name, "formula": formula, "importance": importance})
        plan["metrics"] = new_metrics

        st.subheader("Guardrail Metrics")
        new_guardrails = []
        for i, g in enumerate(plan.get("guardrail_metrics", [])):
            name = st.text_input(f"Guardrail {i+1} Name", g.get("name",""), key=f"guard_name_{i}")
            direction = st.text_input(f"Guardrail {i+1} Direction", g.get("direction",""), key=f"guard_dir_{i}")
            threshold = st.text_input(f"Guardrail {i+1} Threshold", g.get("threshold",""), key=f"guard_thresh_{i}")
            new_guardrails.append({"name": name, "direction": direction, "threshold": threshold})
        plan["guardrail_metrics"] = new_guardrails

        st.subheader("Risks & Assumptions")
        new_risks = []
        for i, r in enumerate(plan.get("risks_and_assumptions", [])):
            risk = st.text_input(f"Risk {i+1}", r.get("risk",""), key=f"risk_risk_{i}")
            severity = st.selectbox(
                f"Risk {i+1} Severity",
                ["High","Medium","Low"],
                index=0 if r.get("severity","High")=="High" else (1 if r.get("severity")=="Medium" else 2),
                key=f"risk_severity_{i}"
            )
            mitigation = st.text_input(f"Risk {i+1} Mitigation", r.get("mitigation",""), key=f"risk_mit_{i}")
            new_risks.append({"risk": risk, "severity": severity, "mitigation": mitigation})
        plan["risks_and_assumptions"] = new_risks
# main.py — Part 5/5: Step 5–6 UI (Quality Check, Export) + Entrypoint

        st.subheader("Success Criteria")
        sc = plan.get("success_criteria", {})
        sc["confidence_level"] = st.number_input(
            "Confidence Level (%)", value=float(sc.get("confidence_level", 95)), step=1.0
        )
        sc["MDE"] = st.number_input(
            "Minimum Detectable Effect (%)", value=float(sc.get("MDE", 5.0)), step=0.1
        )
        sc["benchmark"] = st.text_input("Benchmark", sc.get("benchmark", ""))
        sc["monitoring"] = st.text_input("Monitoring", sc.get("monitoring", ""))
        plan["success_criteria"] = sc

        st.subheader("Experiment Design")
        ed = plan.get("experiment_design", {})
        ed["traffic_allocation"] = st.text_input("Traffic Allocation", ed.get("traffic_allocation","50/50"))
        ed["sample_size_per_variant"] = st.number_input(
            "Sample Size per Variant", value=int(ed.get("sample_size_per_variant", 0)), step=1
        )
        ed["total_sample_size"] = st.number_input(
            "Total Sample Size", value=int(ed.get("total_sample_size", 0)), step=1
        )
        ed["test_duration_days"] = st.number_input(
            "Test Duration (days)", value=float(ed.get("test_duration_days", 14)), step=0.5
        )
        ed["dau_coverage_percent"] = st.number_input(
            "DAU Coverage (%)", value=float(ed.get("dau_coverage_percent", 50)), step=1.0
        )
        plan["experiment_design"] = ed

        st.subheader("Success & Learning Criteria")
        slc = plan.get("success_learning_criteria", {})
        slc["definition_of_success"] = st.text_area(
            "Definition of Success", slc.get("definition_of_success",""), height=80
        )
        slc["stopping_rules"] = st.text_area(
            "Stopping Rules", slc.get("stopping_rules",""), height=80
        )
        slc["rollback_criteria"] = st.text_area(
            "Rollback Criteria", slc.get("rollback_criteria",""), height=80
        )
        plan["success_learning_criteria"] = slc

        st.subheader("Next Steps")
        steps = ensure_list(plan.get("next_steps"))
        new_steps = []
        for i, step in enumerate(steps):
            new_steps.append(st.text_input(f"Next Step {i+1}", step, key=f"next_step_{i}"))
        new_step_add = st.text_input("Add Another Step", "")
        if new_step_add:
            new_steps.append(new_step_add)
        plan["next_steps"] = new_steps

        st.subheader("Statistical Rationale")
        plan["statistical_rationale"] = st.text_area(
            "Statistical Rationale", plan.get("statistical_rationale",""), height=100
        )

        # Save updates
        st.session_state["experiment_plan"] = plan

    # -------------------------
    # Step 5: AI Quality Check
    # -------------------------
    st.header("Step 5: AI Quality Check & Suggestions")
    if "experiment_plan" in st.session_state:
        if st.button("Get AI Feedback", use_container_width=True):
            with st.spinner("Evaluating PRD quality..."):
                if PROMPT_ENGINE_AVAILABLE:
                    try:
                        feedback = validate_experiment_plan(st.session_state["experiment_plan"])
                        st.markdown("### Feedback Suggestions")
                        st.write(feedback)
                    except Exception as e:
                        st.error(f"Error during validation: {e}")
                else:
                    st.warning("Prompt engine not available.")

    # -------------------------
    # Step 6: Finalize & Export
    # -------------------------
    st.header("Step 6: Finalize & Export")
    if "experiment_plan" in st.session_state:
        plan = st.session_state["experiment_plan"]

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Export to JSON", use_container_width=True):
                data = json.dumps(plan, indent=2).encode("utf-8")
                st.download_button(
                    "Download JSON", data, file_name="experiment_prd.json", mime="application/json"
                )
        with col2:
            if REPORTLAB_AVAILABLE and st.button("Export to PDF", use_container_width=True):
                pdf_bytes = generate_pdf_bytes_from_prd_dict(plan)
                st.download_button(
                    "Download PDF", pdf_bytes, file_name="experiment_prd.pdf", mime="application/pdf"
                )
            elif not REPORTLAB_AVAILABLE:
                st.caption("📄 PDF export unavailable (ReportLab not installed).")
        with col3:
            if DOCX_AVAILABLE and st.button("Export to DOCX", use_container_width=True):
                docx_bytes = generate_docx_bytes_from_prd_dict(plan)
                st.download_button(
                    "Download DOCX", docx_bytes, file_name="experiment_prd.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            elif not DOCX_AVAILABLE:
                st.caption("📄 DOCX export unavailable (python-docx not installed).")

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    main()
