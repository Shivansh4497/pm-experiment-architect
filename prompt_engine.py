# prompt_engine.py — Part 1/2
# Prompts, schema, and helpers for generating experiment PRDs and hypothesis details.
# Defensive: supports GROQ client if present, otherwise provides informative stubs.

import os
import json
import re
import textwrap
from typing import Any, Dict, Optional

# Try to import a Groq client (or any other provider you prefer)
GROQ_AVAILABLE = False
_client = None
try:
    # Import guarded so module still loads if Groq not installed
    from groq import Groq  # type: ignore
    GROQ_AVAILABLE = True
    # instantiate safely (expect user to set GROQ_API_KEY in env)
    try:
        _client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    except Exception:
        # keep _client None if init failed
        _client = None
except Exception:
    GROQ_AVAILABLE = False
    _client = None

# You can add other providers here (OpenAI, Anthropic) as needed.
# If none are available, functions will return deterministic stubs (safe fallback).

# -------------------------
# Canonical PRD schema (for prompt authors)
# -------------------------
_CANONICAL_SCHEMA = {
    "metadata": {
        "title": "string",
        "team": "string",
        "owner": "string",
        "experiment_id": "string"
    },
    "problem_statement": "string",
    "proposed_solution": "string",
    "hypotheses": [
        {
            "hypothesis": "string",
            "rationale": "string",
            "example_implementation": "string",
            "behavioral_basis": "string"
        }
    ],
    "variants": [
        {"control": "string", "variation": "string", "notes": "string"}
    ],
    "metrics": [
        {"name": "string", "formula": "string", "importance": "Primary|Secondary|Diagnostic"}
    ],
    "guardrail_metrics": [
        {"name": "string", "direction": "string", "threshold": "string"}
    ],
    "success_criteria": {
        "confidence_level": "float",
        "MDE": "float",
        "benchmark": "string",
        "monitoring": "string",
        "power": "float (optional)"
    },
    "experiment_design": {
        "traffic_allocation": "string",
        "sample_size_per_variant": "int or null",
        "total_sample_size": "int or null",
        "test_duration_days": "float or null",
        "dau_coverage_percent": "float or null"
    },
    "risks_and_assumptions": [
        {"risk": "string", "severity": "High|Medium|Low", "mitigation": "string"}
    ],
    "success_learning_criteria": {
        "definition_of_success": "string",
        "stopping_rules": "string",
        "rollback_criteria": "string"
    },
    "next_steps": ["string"],
    "statistical_rationale": "string"
}

# -------------------------
# JSON extraction helper
# -------------------------
def extract_json(text: Any) -> Optional[Dict[str, Any]]:
    """
    Robust JSON extractor:
    - If input is already a dict, return it.
    - If input is a JSON string, parse and return.
    - If input contains a JSON block, extract and parse.
    - If parsing fails, return None.
    """
    if text is None:
        return None
    if isinstance(text, dict):
        return text
    s = str(text)

    # Fast path: clean string and try direct parse
    try:
        return json.loads(s)
    except Exception:
        pass

    # Try to extract first {...} block
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    candidate = m.group(0)

    # Try direct parse
    try:
        return json.loads(candidate)
    except Exception:
        pass

    # Last-resort: naive single-quote -> double-quote conversion for simple cases
    # This is conservative and won't fix deeply malformed JSON
    try:
        cand2 = re.sub(r"(?<=[:\{\[,]\s*)'([^']*?)'(?=\s*[,}\]])", r'"\1"', candidate)
        cand2 = cand2.replace("\n", " ")
        return json.loads(cand2)
    except Exception:
        return None

# -------------------------
# Prompt builders
# -------------------------
def _build_main_prompt(goal: str, context: Dict[str, Any]) -> str:
    """
    Create an LLM system/user prompt to generate a full PRD following the canonical schema.
    This prompt is explicit about the exact JSON structure required.
    """
    ctx = {k: ("" if v is None else str(v)) for k, v in context.items()}

    schema_text = json.dumps(_CANONICAL_SCHEMA, indent=2)
    prompt = textwrap.dedent(f"""
    You are an expert Principal Product Manager and experimentation lead.
    Your task: generate a single, valid JSON object that is a complete A/B experiment PRD.
    The JSON MUST follow the schema exactly (keys present). If a section has no content, use an empty string or empty list.

    CONTEXT:
    - Business goal: {ctx.get('high_level_goal', '')}
    - Product Type: {ctx.get('product_type', '')}
    - Target User Persona: {ctx.get('target_user', '')}
    - Key Metric: {ctx.get('key_metric', '')}
    - Current Value: {ctx.get('current_value', '')}
    - Target Value: {ctx.get('target_value', '')}
    - Additional notes: {ctx.get('notes', '')}

    OUTPUT SCHEMA (must be valid JSON, no extra text):
    {schema_text}

    GUIDELINES:
    - Provide at least 1 and up to 3 hypotheses in the specified format "If [change] then [outcome] because [rationale]".
    - For each hypothesis include: hypothesis, rationale (mention any credible source or reasoning), example_implementation (step-by-step UI or flow changes), behavioral_basis (named principle).
    - Variants: clearly and unambiguously describe control and variation.
    - Metrics: include one Primary metric and any Secondary or Diagnostic metrics as needed; provide formulas if applicable.
    - Guardrails: include metrics and thresholds to prevent regressions.
    - Success & Learning Criteria: include definition_of_success, stopping_rules, and rollback_criteria—be concrete and actionable.
    - Experiment design: provide an initial sample size if possible (use provided current/target/MDE/confidence); if you cannot compute numeric values, use null and explain briefly inside 'statistical_rationale'.
    - Risks: list likely technical or user risks and a short mitigation for each.
    - Next steps: give 3-6 verb-first, assignable next steps (e.g., 'Assign: Data - compute event funnel by 2025-08-20').
    - Statistical rationale: explain why chosen confidence/MDE/sample-size is appropriate, or state what additional inputs are needed.

    Return only valid JSON that adheres to the schema.
    """).strip()

    return prompt


def _build_hypothesis_prompt(hypothesis_text: str, context: Dict[str, Any]) -> str:
    """
    Prompt that asks the model to expand a one-line hypothesis into the structured hypothesis object.
    """
    ctx = {k: ("" if v is None else str(v)) for k, v in context.items()}
    prompt = textwrap.dedent(f"""
    You are an expert Product Manager. Expand the following hypothesis into a JSON object with keys:
    hypothesis, rationale, example_implementation, behavioral_basis.

    Context:
    - Business goal: {ctx.get('high_level_goal', '')}
    - Product Type: {ctx.get('product_type', '')}
    - Key Metric: {ctx.get('key_metric', '')}
    - Target Persona: {ctx.get('target_user', '')}

    INPUT HYPOTHESIS:
    \"\"\"{hypothesis_text}\"\"\"

    OUTPUT (valid JSON only, single object):
    {{
      "hypothesis": "...",
      "rationale": "...",
      "example_implementation": "...",
      "behavioral_basis": "..."
    }}
    """).strip()

    return prompt


def generate_hypotheses(context: Dict[str, Any]) -> str:
    """
    Generate 3 alternative hypotheses given the context.
    Returns a JSON string: {"hypotheses": [list of hypothesis strings]}.
    """
    prompt = f"""
    You are an expert Product Manager.
    Given this experiment context:

    {json.dumps(context, indent=2)}

    Generate 3 distinct, concise hypotheses in the format:
    "If [change], then [outcome], because [rationale]."

    Return ONLY valid JSON with this shape:
    {{
      "hypotheses": [
        "hypothesis 1",
        "hypothesis 2",
        "hypothesis 3"
      ]
    }}
    """

    llm_resp = _call_llm(prompt, max_tokens=600, temperature=0.7)
    if llm_resp:
        parsed = extract_json(llm_resp)
        if parsed and "hypotheses" in parsed:
            return _safe_json_dumps(parsed)
        else:
            return llm_resp

    # Fallback stub
    return _safe_json_dumps({
        "hypotheses": [
            "If we simplify the signup flow, then completion rate will improve because friction is reduced.",
            "If we personalize homepage content, then engagement will rise because relevance increases.",
            "If we highlight discounts earlier in checkout, then conversions will improve because users feel higher urgency."
        ]
    })

def _build_validation_prompt(plan: Dict[str, Any]) -> str:
    """
    Prompt for LLM to validate the PRD and give suggestions.
    Returns instructions expecting a concise JSON or markdown text summary.
    """
    short_plan = json.dumps(plan, indent=2) if isinstance(plan, dict) else str(plan)
    prompt = textwrap.dedent(f"""
    You are an expert Experimentation Lead. Review the following experiment PRD JSON for:
      - Clarity of problem statement and hypothesis
      - Statistical rigor and sample-size sanity
      - Completeness of success & learning criteria
      - Risks & mitigations coverage
      - Actionability of next steps

    PRD:
    {short_plan}

    Provide:
    - A short JSON object with keys: is_valid (true/false), critical_issues (list), suggested_improvements (list), pro_tips (list).
    Return valid JSON only.
    """).strip()
    return prompt
# prompt_engine.py — Part 2/2
# LLM call wrappers, safe fallbacks, and utilities.
# Exported functions:
#   - generate_experiment_plan(...) -> str (LLM raw output or JSON string)
#   - generate_hypothesis_details(hypothesis_text, context) -> str (LLM raw output or JSON string)
#   - validate_experiment_plan(plan) -> str (LLM raw feedback or JSON string)

import json
import time
from typing import Any, Dict, Optional, Tuple

# Import objects from part 1
try:
    # _client, GROQ_AVAILABLE, extract_json, _build_main_prompt, _build_hypothesis_prompt, _build_validation_prompt are defined in part1
    _ = _client  # type: ignore
except NameError:
    _client = None
    GROQ_AVAILABLE = False  # type: ignore

# -------------------------
# Utilities
# -------------------------
def _safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        try:
            return json.dumps(obj, default=str, indent=2, ensure_ascii=False)
        except Exception:
            return "{}"


def _call_llm(prompt: str, max_tokens: int = 1500, temperature: float = 0.2) -> Optional[str]:
    """
    Generic LLM invocation wrapper.
    - If a Groq client (_client) is present and callable in expected shape, attempt to call it.
    - If call fails or no client available, return None so caller can fallback to stub.
    """
    if _client is None:
        return None

    # Attempt multiple reasonable invocation patterns to be robust to client shapes
    try:
        # Preferred: Groq-like chat completions
        if hasattr(_client, "chat") and hasattr(_client.chat, "completions"):
            resp = _client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[{"role": "system", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # Try to pull content consistently
            try:
                return resp.choices[0].message["content"]
            except Exception:
                try:
                    return resp.choices[0].message.content
                except Exception:
                    return str(resp)
        # Fallback: if client has "completions.create"
        elif hasattr(_client, "completions") and hasattr(_client.completions, "create"):
            resp = _client.completions.create(model="default", prompt=prompt, max_tokens=max_tokens, temperature=temperature)
            # Try common response shapes
            if isinstance(resp, dict) and "choices" in resp and resp["choices"]:
                return resp["choices"][0].get("text") or resp["choices"][0].get("message") or str(resp["choices"][0])
            return str(resp)
        else:
            # As last resort, try calling the client as a function
            resp = _client(prompt)
            return str(resp)
    except Exception:
        # avoid raising — let callers fallback
        return None


# -------------------------
# Mock / deterministic fallbacks
# -------------------------
def _stub_hypothesis_object(seed_hypothesis: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a conservative, useful stub for an expanded hypothesis object.
    """
    hyp_text = seed_hypothesis if seed_hypothesis else "If we make a small UX change, then the primary metric will improve because friction is reduced."
    rationale = "Based on common conversion optimization patterns: reducing friction and increasing prominence typically raises the metric."
    example_impl = "Change the primary CTA text and increase button size by 20%. Implement AB routing for 50/50 split."
    behavioral = "Hick's Law / Salience"
    return {
        "hypothesis": hyp_text,
        "rationale": rationale,
        "example_implementation": example_impl,
        "behavioral_basis": behavioral,
    }


def _stub_full_prd(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a conservative, minimal but valid PRD structure when LLM is unavailable.
    The content is intentionally generic but useful as a placeholder for editing in the UI.
    """
    title = context.get("high_level_goal") or context.get("title") or "Experiment — " + (context.get("key_metric") or "Unnamed Metric")
    owner = (context.get("metadata") or {}).get("owner") or context.get("owner") or "Unknown"
    team = (context.get("metadata") or {}).get("team") or context.get("team") or "Product"
    exp_id = context.get("metadata", {}).get("experiment_id") or f"STUB-{int(time.time())}"

    hypothesis_seed = ""
    chosen_h = context.get("chosen_hypothesis")
    if isinstance(chosen_h, dict):
        hypothesis_seed = chosen_h.get("hypothesis", "")
    elif isinstance(chosen_h, str):
        hypothesis_seed = chosen_h

    hyp_obj = _stub_hypothesis_object(hypothesis_seed, context)

    metric_name = context.get("key_metric") or "Primary Metric"
    current_val = context.get("current_value") or ""
    target_val = context.get("target_value") or ""

    plan = {
        "metadata": {"title": title, "team": team, "owner": owner, "experiment_id": exp_id},
        "problem_statement": f"Current: {current_val}. Target: {target_val}. The experiment aims to move {metric_name}.",
        "proposed_solution": "Implement the chosen variation described in Variants.",
        "hypotheses": [hyp_obj],
        "variants": [{"control": "Current experience", "variation": "Proposed change", "notes": ""}],
        "metrics": [{"name": metric_name, "formula": current_val + " -> " + target_val, "importance": "Primary"}],
        "guardrail_metrics": [{"name": "Conversion Rate", "direction": "must not decrease", "threshold": "no more than 1% drop"}],
        "success_criteria": {"confidence_level": float(context.get("success_criteria", {}).get("confidence_level", 95.0)), "MDE": float(context.get("success_criteria", {}).get("MDE", 5.0)), "benchmark": "", "monitoring": ""},
        "experiment_design": {"traffic_allocation": "50/50", "sample_size_per_variant": None, "total_sample_size": None, "test_duration_days": None, "dau_coverage_percent": float((context.get("experiment_design") or {}).get("dau_coverage_percent", 50.0))},
        "risks_and_assumptions": [{"risk": "Novelty effect", "severity": "Medium", "mitigation": "Monitor uplift over time and run longer if needed."}],
        "success_learning_criteria": {"definition_of_success": f"{metric_name} improves by at least {context.get('success_criteria', {}).get('MDE', 5)}% with p < 0.05.", "stopping_rules": "Stop if major regression on guardrails", "rollback_criteria": "Rollback if guardrail breached beyond threshold."},
        "next_steps": ["Assign: Data - validate instrumentation", "Assign: Eng - implement variant", "Assign: Design - provide assets"],
        "statistical_rationale": "Insufficient inputs to compute exact sample size. Provide baseline conversion and std dev for numeric metrics to calculate sample size."
    }
    return plan


# -------------------------
# Exported functions
# -------------------------
def generate_experiment_plan(*args, **kwargs) -> str:
    """
    Flexible wrapper:
    - If called as generate_experiment_plan(goal_str, context_dict) it uses those.
    - If called as generate_experiment_plan(context_dict) it uses the dict.
    Returns the LLM raw output string if available, else returns JSON string of a conservative stub.
    """
    # Normalize inputs
    goal = ""
    context = {}
    if len(args) == 1 and isinstance(args[0], dict):
        context = args[0]
        goal = context.get("high_level_goal", "") or context.get("goal", "")
    elif len(args) >= 2:
        goal = args[0] or ""
        context = args[1] or {}
    else:
        goal = kwargs.get("goal", "") or ""
        context = kwargs.get("context", {}) or {}

    # Ensure context keys exist to help prompt
    ctx_for_prompt = dict(context)
    if goal and not ctx_for_prompt.get("high_level_goal"):
        ctx_for_prompt["high_level_goal"] = goal

    prompt = _build_main_prompt(goal or ctx_for_prompt.get("high_level_goal", ""), ctx_for_prompt)

    # Try LLM
    llm_resp = _call_llm(prompt, max_tokens=2500, temperature=0.15)
    if llm_resp:
        # If LLM returned text, try to extract JSON block.
        parsed = extract_json(llm_resp)
        if parsed:
            # Return JSON string to keep deterministic shape for main app
            return _safe_json_dumps(parsed)
        else:
            # If unable to parse LLM output as JSON, still return raw text so calling code can inspect
            return llm_resp

    # Fallback: mock PRD
    stub = _stub_full_prd(ctx_for_prompt)
    return _safe_json_dumps(stub)


def generate_hypothesis_details(hypothesis_text: str, context: Dict[str, Any]) -> str:
    """
    Expand a single-line hypothesis into a structured hypothesis object.
    Returns JSON string or raw LLM text if extraction fails.
    """
    if not hypothesis_text:
        hypothesis_text = "If we make a small UX change, then the key metric will improve because friction is reduced."

    prompt = _build_hypothesis_prompt(hypothesis_text, context or {})

    llm_resp = _call_llm(prompt, max_tokens=800, temperature=0.4)
    if llm_resp:
        parsed = extract_json(llm_resp)
        if parsed:
            return _safe_json_dumps(parsed)
        else:
            return llm_resp

    # Fallback
    stub = _stub_hypothesis_object(hypothesis_text, context or {})
    return _safe_json_dumps(stub)


def validate_experiment_plan(plan: Dict[str, Any]) -> str:
    """
    Validate the given plan using the LLM if available. Returns LLM response string or a local JSON summary.
    """
    # Local quick validation
    local = None
    try:
        from pprint import pformat
    except Exception:
        pformat = lambda x: str(x)

    # If LLM available, call it
    prompt = _build_validation_prompt(plan or {})
    llm_resp = _call_llm(prompt, max_tokens=800, temperature=0.0)
    if llm_resp:
        # Try parse JSON; if not JSON, return text
        parsed = extract_json(llm_resp)
        if parsed:
            return _safe_json_dumps(parsed)
        else:
            return llm_resp

    # Fallback: produce local summary
    local_issues = []
    local_suggestions = []
    p = plan or {}
    if not p.get("metadata", {}).get("title"):
        local_issues.append("Missing title in metadata.")
        local_suggestions.append("Add a concise title that summarizes the experiment's objective.")
    if not p.get("problem_statement"):
        local_issues.append("Missing problem statement.")
        local_suggestions.append("Describe the user pain and the metric gap quantitatively.")
    if not p.get("hypotheses"):
        local_issues.append("No hypotheses present.")
        local_suggestions.append("Add at least one clear hypothesis in the 'If X then Y because Z' format.")
    if not p.get("metrics"):
        local_issues.append("No metrics specified.")
        local_suggestions.append("Define the primary metric and one or two secondary/guardrail metrics.")

    local = {
        "is_valid": len(local_issues) == 0,
        "critical_issues": local_issues,
        "suggested_improvements": local_suggestions,
        "pro_tips": ["Ensure instrumentation for primary metric before launch.", "Prefer guardrails that are measurable and actionable."]
    }
    return _safe_json_dumps(local)


# -------------------------
# Expose public API in module namespace
# -------------------------
__all__ = [
    "generate_experiment_plan",
    "generate_hypothesis_details",
    "validate_experiment_plan",
    "extract_json",
]
