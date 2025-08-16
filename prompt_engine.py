# prompt_engine.py

# This file contains the prompts and logic for calling the LLM to generate
# experiment plans and hypothesis details.

import os
import json
import textwrap
from typing import Dict, Any, Optional

# Attempt to import Groq client
try:
    from groq import Groq  # type: ignore
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False
    print("Warning: Groq client not available. Please ensure the groq library is installed and the GROQ_API_KEY environment variable is set for full functionality.")

def _build_main_prompt(goal: str, context: Dict[str, Any]) -> str:
    """
    Build the main system prompt for generating a full A/B Test PRD.
    Includes metadata, hypotheses, variants, metrics, design, risks, and success criteria.
    """

    schema = {
        "metadata": {
            "title": "string — clear title for the experiment",
            "owner": "string — experiment owner",
            "team": "string — product team",
            "experiment_id": "string — unique experiment ID",
        },
        "problem_statement": "string — detailed explanation of the problem or opportunity",
        "hypotheses": [
            {
                "hypothesis": "string",
                "rationale": "string",
                "example_implementation": "string",
                "behavioral_basis": "string"
            }
        ],
        "variants": {
            "control": "string — description of control",
            "treatment": "string — description of variant"
        },
        "metrics": {
            "primary": "string",
            "secondary": ["string"]
        },
        "guardrail_metrics": ["string — metrics to ensure no negative side effects"],
        "experiment_design": {
            "sample_size_required": "int — minimum sample size required",
            "duration": "string — estimated duration",
            "traffic_allocation": "string — e.g., 50/50",
            "dau_coverage_percent": "int — % of DAU covered",
        },
        "risks": ["string — risk description + mitigation"],
        "success_learning_criteria": {
            "definition_of_success": "string — what defines success",
            "stopping_rules": "string — when to stop or pause",
            "rollback_criteria": "string — rollback conditions",
        },
        "success_criteria": {
            "confidence_level": "int — confidence level (e.g., 95)",
            "MDE": "float — minimum detectable effect (%)",
            "power": "int — statistical power (e.g., 80)",
        }
    }

    return f"""
You are an expert experimentation product manager and data scientist.
Generate a **complete PRD (Product Requirements Document)** for an A/B test.

## Input Context
Business Goal: {goal}
Context: {json.dumps(context, indent=2)}

## Instructions
- Output must be **valid JSON only** (no extra commentary).
- Populate ALL fields according to this schema:
{json.dumps(schema, indent=2)}

- Be specific and professional. Avoid placeholders like "TBD".
- Ensure hypotheses follow the structure: "If we [change], then [outcome], because [reason]."
- Provide realistic sample size, risks, and mitigation strategies.
- Success and learning criteria must include a clear definition of success, stopping rules, and rollback criteria.
    """


def _build_validation_prompt(plan: Dict[str, Any]) -> str:
    """
    Build a validation prompt to check quality of a PRD.
    """

    return f"""
You are reviewing the following A/B Test PRD for quality and rigor.

PRD JSON:
{json.dumps(plan, indent=2)}

Evaluate and suggest improvements for:
- Clarity of problem statement and hypothesis
- Statistical rigor of metrics and experiment design
- Completeness of success & learning criteria
- Risks & mitigations coverage
- Overall consistency

Respond with a short professional feedback note (markdown list).
    """


def _build_hypothesis_prompt(hypothesis: str, context: Dict[str, Any]) -> str:
    """
    Build a prompt to expand and detail a hypothesis.
    """

    return f"""
You are helping expand a hypothesis into a structured, detailed form.

Context:
{json.dumps(context, indent=2)}

Hypothesis:
{hypothesis}

Output JSON with this structure:
{{
  "hypothesis": "...",
  "rationale": "...",
  "example_implementation": "...",
  "behavioral_basis": "..."
}}
    """
# prompt_engine.py — A/B Test Architect (full file, part 2/2)

# =========================
# LLM Call Functions
# =========================

def generate_experiment_plan(goal: str, context: Dict[str, Any]) -> str:
    """
    Call the LLM to generate a full experiment PRD JSON.
    """
    prompt = _build_main_prompt(goal, context)
    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.2,
        max_tokens=2000,
    )
    return response.choices[0].message["content"]


def validate_experiment_plan(plan: Dict[str, Any]) -> str:
    """
    Call the LLM to validate and give suggestions for a PRD.
    """
    prompt = _build_validation_prompt(plan)
    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.0,
        max_tokens=800,
    )
    return response.choices[0].message["content"]


def generate_hypothesis_details(hypothesis: str, context: Dict[str, Any]) -> str:
    """
    Call the LLM to expand a hypothesis into detailed structured form.
    """
    prompt = _build_hypothesis_prompt(hypothesis, context)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.4,
        max_tokens=600,
    )
    return response.choices[0].message["content"]


# =========================
# Helpers
# =========================

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to extract JSON object from a text string (LLM output).
    """
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try regex extraction
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
    return None
