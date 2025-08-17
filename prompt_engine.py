# prompt_engine.py — Part 1/4 (Fixed)
# Full updated version with all fixes and compatibility improvements

import os
import json
import re
from typing import Dict, Any, List, Optional
from io import BytesIO

# ============ LLM Client Setup ============
try:
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: OpenAI package not available. LLM features will be disabled.")

# ============ Constants ============
DEFAULT_MODEL = "gpt-4"  # Updated from invalid "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.7

# ============ Prompt Templates ============
PROMPTS = {
    "hypotheses": """You are an expert Product Manager. Based on the user inputs, generate 3 highly relevant A/B test hypotheses.
Each hypothesis must be structured JSON with:
- hypothesis: a single clear testable statement
- rationale: why this change could work
- example_implementation: one concrete way to run this test
- behavioral_basis: psychological principle backing this

User Inputs:
Business Goal: {business_goal}
Product Type: {product_type}
Target Persona: {user_persona}
Key Metric: {key_metric}
Current Value: {current_value}
Target Value: {target_value}

Return JSON only in this structure:
{{
  "hypotheses": [
    {{
      "hypothesis": "...",
      "rationale": "...",
      "example_implementation": "...",
      "behavioral_basis": "..."
    }}
  ]
}}""",

    "hypothesis_details": """You are an expert Product Manager. Expand the provided hypothesis with more detail.
Add sections for:
- Behavioral Basis: A psychological or user behavior principle that explains the hypothesis.
- Example Implementation: A concrete, technical-level example of how the test would be implemented.

User Inputs:
Business Goal: {business_goal}
Product Type: {product_type}
Target Persona: {user_persona}
Hypothesis: {hypothesis}
Rationale: {rationale}

Return JSON only in this structure:
{{
  "hypothesis": "{hypothesis}",
  "rationale": "{rationale}",
  "example_implementation": "...",
  "behavioral_basis": "..."
}}""",

    "prd": """You are a highly analytical and detail-oriented Product Manager. Based on the user's business context and a chosen hypothesis, generate a complete experiment plan (PRD).
The output MUST be a single JSON object.

Business Context:
Business Goal: {business_goal}
Product Type: {product_type}
Target Persona: {user_persona}
Key Metric: {key_metric}
Current Value: {current_value}
Target Value: {target_value}

Chosen Hypothesis:
Hypothesis: {hypothesis}
Rationale: {rationale}
Example Implementation: {example_implementation}
Behavioral Basis: {behavioral_basis}

Generate the full PRD JSON object with these sections. Ensure all fields are filled.
1.  metadata:
    - title: A concise title for the experiment (e.g., "Checkout Button Color Test")
    - team: [Your Team Name]
    - owner: [Your Name]
2.  problem_statement: A short description of the user problem to be solved.
3.  hypotheses: An array with the single chosen hypothesis.
4.  proposed_solution: A detailed description of the proposed solution and how it will address the problem.
5.  variants: An array describing the control and variation(s).
    - control: A description of the current experience.
    - variation: A description of the new experience to be tested.
6.  metrics: An array of primary and secondary success metrics.
    - name: e.g., "Click-through Rate"
    - formula: A clear formula, e.g., "Clicks / Impressions"
    - importance: "Primary" or "Secondary"
7.  guardrail_metrics: An array of metrics to monitor for negative side effects.
    - name: e.g., "Page Load Time"
    - direction: "Decrease" (for negative side effects)
    - threshold: e.g., "+5%"
8.  experiment_design:
    - traffic_allocation: e.g., "50/50"
    - sample_size_per_variant: a calculated number
    - total_sample_size: a calculated number
    - test_duration_days: a calculated number
    - dau_coverage_percent: % of DAU covered by test
    - power: statistical power of the test
9.  risks_and_assumptions: An array of potential risks and how to mitigate them.
    - risk: e.g., "Cannibalization of other products"
    - severity: "High", "Medium", or "Low"
    - mitigation: e.g., "Monitor funnel metrics"
10. statistical_rationale: A detailed rationale for the experiment design, including calculations for sample size, duration, and rationale for MDE/power/confidence level.

Return JSON only, with no additional text or markdown outside the JSON object.
""",
    "tips": """You are an expert product manager. The user is currently in the process of building an experiment plan. Provide concise, actionable tips based on their current step and context. The tips should be a JSON array of short strings.

Context:
User is on the '{current_step}' step.
Their current inputs are:
{context}

Provide a JSON array of up to 5 tips to help them improve their current work.
Return JSON array only, like this: ["Tip 1", "Tip 2", ...].
"""
}
# prompt_engine.py — Part 2/4 (Fixed)

# ============ LLM Interaction ============
def extract_json_from_text(text: Optional[str]) -> Dict[str, Any]:
    """Robust JSON extraction from potentially messy LLM output."""
    if not text:
        return {}
    
    # Try to find a JSON object or array block
    match = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass # Fall through to other extraction methods
            
    # Try more lenient extraction
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass
    
    return {}

def safe_call_llm(prompt_template: str, inputs: Dict[str, Any], model: str = DEFAULT_MODEL) -> Any:
    """Safely calls the LLM, handles errors, and returns parsed JSON."""
    if not LLM_AVAILABLE:
        return {}

    prompt = prompt_template.format(**inputs)
    
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=DEFAULT_TEMPERATURE,
        )
        content = response.choices[0].message.content
        return extract_json_from_text(content)
    except openai.APIError as e:
        print(f"API Error: {e}")
        return {"error": str(e)}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {"error": str(e)}

# ============ PRD Schema Validation ============
def _validate_experiment_plan(plan: Dict[str, Any]) -> List[str]:
    """
    Validates the experiment plan JSON against common best practices.
    Returns a list of issues found.
    """
    issues = []
    
    # Check for empty fields
    for key, value in plan.items():
        if isinstance(value, str) and not value.strip():
            issues.append(f"Missing value for field: {key}")
        elif isinstance(value, list) and not value:
            issues.append(f"List is empty for field: {key}")
            
    # Check for metrics
    metrics = plan.get("metrics", [])
    if not metrics:
        issues.append("No success metrics defined.")
    else:
        primary_count = sum(1 for m in metrics if m.get("importance", "").lower() == "primary")
        if primary_count == 0:
            issues.append("No primary success metric defined.")
        if primary_count > 1:
            issues.append(f"Multiple primary metrics defined ({primary_count}). Best practice is to have a single primary metric.")
            
    # Check for sample size consistency
    design = plan.get("experiment_design", {})
    total_size = design.get("total_sample_size", 0)
    per_variant_size = design.get("sample_size_per_variant", 0)
    if total_size > 0 and per_variant_size > 0 and total_size != per_variant_size * 2:
        issues.append(f"Total sample size ({total_size}) does not match per-variant size ({per_variant_size} * 2).")

    # Check for risks
    if not plan.get("risks_and_assumptions"):
        issues.append("No risks and assumptions defined. Consider potential pitfalls.")
        
    return issues

# ============ Sanitization Helpers ============
def sanitize_plan(raw_plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitizes and normalizes the raw LLM output to a predefined schema.
    This is a critical step to handle LLM quirks and ensure backward compatibility.
    """
    if not isinstance(raw_plan, dict):
        return {}

    sanitized: Dict[str, Any] = {
        "metadata": {"title": "", "team": "", "owner": ""},
        "problem_statement": "",
        "hypotheses": [],
        "proposed_solution": "",
        "variants": [],
        "metrics": [],
        "guardrail_metrics": [],
        "experiment_design": {},
        "success_criteria": {},
        "success_learning_criteria": {},
        "risks_and_assumptions": [],
        "statistical_rationale": "",
    }

    # Scalars
    for key in ["problem_statement", "proposed_solution", "statistical_rationale"]:
        if raw_plan.get(key) is not None:
            sanitized[key] = str(raw_plan.get(key))

    # Metadata
    meta = raw_plan.get("metadata", {})
    if isinstance(meta, dict):
        sanitized["metadata"]["title"] = str(meta.get("title", ""))
        sanitized["metadata"]["team"] = str(meta.get("team", ""))
        sanitized["metadata"]["owner"] = str(meta.get("owner", ""))

    # Hypotheses
    hyps = raw_plan.get("hypotheses", [])
    if isinstance(hyps, list):
        for h in hyps:
            if isinstance(h, dict):
                sanitized["hypotheses"].append({
                    "hypothesis": str(h.get("hypothesis", "")),
                    "rationale": str(h.get("rationale", "")),
                    "example_implementation": str(h.get("example_implementation", "")),
                    "behavioral_basis": str(h.get("behavioral_basis", ""))
                })
# prompt_engine.py — Part 3/4 (Fixed)

# Sanitize list-based fields
    for v in raw_plan.get("variants", []):
        if isinstance(v, dict):
            sanitized["variants"].append({
                "control": str(v.get("control", "")),
                "variation": str(v.get("variation", "")),
                "notes": str(v.get("notes", ""))
            })

    for m in raw_plan.get("metrics", []):
        if isinstance(m, dict):
            importance = str(m.get("importance", "")).lower()
            importance = "Primary" if importance == "primary" else "Secondary"
            sanitized["metrics"].append({
                "name": str(m.get("name", "")),
                "formula": str(m.get("formula", "")),
                "importance": importance
            })

    for g in raw_plan.get("guardrail_metrics", []):
        if isinstance(g, dict):
            direction = g.get("direction", "Decrease")
            if direction not in ["Increase", "Decrease", "No Change"]:
                direction = "Decrease"
            sanitized["guardrail_metrics"].append({
                "name": str(g.get("name", "")),
                "direction": direction,
                "threshold": str(g.get("threshold", ""))
            })

    for r in raw_plan.get("risks_and_assumptions", []):
        if isinstance(r, dict):
            severity = r.get("severity", "Medium")
            if severity not in ["High", "Medium", "Low"]:
                severity = "Medium"
            sanitized["risks_and_assumptions"].append({
                "risk": str(r.get("risk", "")),
                "severity": severity,
                "mitigation": str(r.get("mitigation", ""))
            })

    # Sanitize dictionary fields with type coercion
    ed = raw_plan.get("experiment_design", {})
    if isinstance(ed, dict):
        sanitized["experiment_design"] = {
            "traffic_allocation": str(ed.get("traffic_allocation", "")),
            "sample_size_per_variant": int(ed.get("sample_size_per_variant", 0)),
            "total_sample_size": int(ed.get("total_sample_size", 0)),
            "test_duration_days": int(ed.get("test_duration_days", 0)),
            "dau_coverage_percent": float(ed.get("dau_coverage_percent", 0.0)),
            "power": float(ed.get("power", 80.0)),
        }

    sc = raw_plan.get("success_criteria", {})
    if isinstance(sc, dict):
        sanitized["success_criteria"] = {
            "confidence_level": float(sc.get("confidence_level", 95.0)),
            "power": float(sc.get("power", 80.0)),
            "MDE": float(sc.get("MDE", 1.0)),
            "benchmark": str(sc.get("benchmark", "")),
            "monitoring": str(sc.get("monitoring", "")),
        }

    slc = raw_plan.get("success_learning_criteria", {})
    if isinstance(slc, dict):
        sanitized["success_learning_criteria"] = {
            "definition_of_success": str(slc.get("definition_of_success", "")),
            "stopping_rules": str(slc.get("stopping_rules", "")),
            "rollback_criteria": str(slc.get("rollback_criteria", "")),
        }

    return sanitized


# ============ Public Functions (Main API) ============

def generate_hypotheses(inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generates a set of A/B test hypotheses based on business context."""
    raw_output = safe_call_llm(PROMPTS["hypotheses"], inputs)
    hyps_list = raw_output.get("hypotheses", [])
    if isinstance(hyps_list, list):
        return hyps_list
    return []

def expand_hypothesis_with_details(business_context: Dict[str, Any], hypothesis_info: Dict[str, Any]) -> Dict[str, Any]:
    """Expands a single hypothesis with behavioral basis and implementation details."""
    inputs = {**business_context, **hypothesis_info}
    raw_output = safe_call_llm(PROMPTS["hypothesis_details"], inputs)
    return raw_output if isinstance(raw_output, dict) else {}

def generate_experiment_plan(business_context: Dict[str, Any], hypothesis_details: Dict[str, Any]) -> Dict[str, Any]:
    """Generates a complete experiment plan (PRD) JSON from a hypothesis."""
    inputs = {**business_context, **hypothesis_details}
    raw_output = safe_call_llm(PROMPTS["prd"], inputs)
    return sanitize_plan(raw_output)

def validate_experiment_plan(plan: Dict[str, Any]) -> List[str]:
    """Validates a given experiment plan against best practices."""
    return _validate_experiment_plan(plan)

def generate_tips(current_step: str, context: Dict[str, Any]) -> List[str]:
    """Generates contextual tips for the user based on their current progress."""
    raw_output = safe_call_llm(PROMPTS["tips"], {"current_step": current_step, "context": json.dumps(context, indent=2)})
    return raw_output if isinstance(raw_output, list) else []

# ============ Backward Compatibility Aliases ============
# These functions ensure that older main.py versions still work without modification.
def generate_prd(business_context: Dict[str, Any], hypothesis_details: Dict[str, Any]) -> Dict[str, Any]:
    """Alias for generate_experiment_plan for backward compatibility."""
    return generate_experiment_plan(business_context, hypothesis_details)

def sanitize_plan_for_legacy(raw_plan: Dict[str, Any]) -> Dict[str, Any]:
    """Alias for sanitize_plan for backward compatibility."""
    return sanitize_plan(raw_plan)

def generate_hypothesis_details(business_context: Dict[str, Any], hypothesis_info: Dict[str, Any]) -> Dict[str, Any]:
    """Alias for expand_hypothesis_with_details."""
    return expand_hypothesis_with_details(business_context, hypothesis_info)

def generate_dynamic_tips(current_step: str, context: Dict[str, Any]) -> List[str]:
    """Alias for generate_tips."""
    return generate_tips(current_step, context)
