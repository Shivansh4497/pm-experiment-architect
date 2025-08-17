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
def sanitize_experiment_plan(plan: dict) -> dict:
    """Ensure experiment plan dict is safe for UI and export."""
    if not isinstance(plan, dict):
        return {}

    safe = {}
    for k, v in plan.items():
        if isinstance(v, dict):
            safe[k] = sanitize_experiment_plan(v)
        elif isinstance(v, list):
            safe[k] = [sanitize_experiment_plan(i) if isinstance(i, dict) else str(i) for i in v]
        else:
            safe[k] = str(v) if v is not None else ""
    return safe
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
    Create a strongly personalized prompt to generate a full PRD following the canonical schema.
    Forces the model to explicitly reference ALL user inputs in each relevant section.
    """
    ctx = {k: ("" if v is None else str(v)) for k, v in context.items()}

    schema_text = json.dumps(_CANONICAL_SCHEMA, indent=2)
    prompt = textwrap.dedent(f"""
    You are an expert Principal Product Manager and experimentation lead.

    TASK:
    Generate a single, valid JSON object for an A/B experiment PRD that STRICTLY adheres to this schema:
    {schema_text}

    PERSONALIZATION CONSTRAINTS (MANDATORY):
    - You MUST explicitly reference ALL provided inputs in the most relevant sections:
      • Business goal: {ctx.get('high_level_goal', '')}
      • Product type: {ctx.get('product_type', '')}
      • Target persona: {ctx.get('target_user', '')}
      • Key metric: {ctx.get('key_metric', '')}
      • Baseline (current value): {ctx.get('current_value', '')}
      • Target value: {ctx.get('target_value', '')}
      • Chosen hypothesis (if present): {ctx.get('chosen_hypothesis', '')}
      • Confidence / Power / MDE (if present): {ctx.get('success_criteria', '')}
      • DAU coverage / traffic allocation (if present): {ctx.get('experiment_design', '')}

    FORMATTING:
    - Return ONLY a single JSON object (no markdown fences, no commentary).
    - All keys from the schema must be present. Use empty strings/lists only when truly unknown.
    - Keep language concrete, specific to the product type/persona, and avoid generic phrases.

    SECTION GUIDANCE (QUALITY BAR):
    - "metadata": Title should combine the business goal + product type. Owner/team come from inputs if present.
    - "problem_statement": Explain WHY improving {ctx.get('key_metric','')} for {ctx.get('target_user','')} matters NOW; reference the baseline {ctx.get('current_value','')} vs target {ctx.get('target_value','')}.
    - "hypotheses": If context contains a chosen hypothesis object, include it as the first item with fields (hypothesis, rationale, example_implementation, behavioral_basis). Do NOT invent a different one unless needed as alternates.
    - "variants": Control = current {ctx.get('product_type','')} experience; Variation = change directly tied to the hypothesis.
    - "metrics": Include the key metric as Primary and add 2–3 Secondary/Diagnostic that make sense for {ctx.get('product_type','')} and {ctx.get('target_user','')}. Include formulas when obvious.
    - "guardrail_metrics": Include at least 2 guardrails (e.g., retention, latency, error rate, uninstalls) with direction/thresholds.
    - "success_criteria": Fill confidence_level / power / MDE using inputs when present; otherwise set defaults and state assumptions in statistical_rationale.
    - "experiment_design": Propose sample_size_per_variant and total_sample_size if possible (based on current→target). If not enough info, set nulls and justify in statistical_rationale. Include traffic_allocation and dau_coverage_percent guidance.
    - "risks_and_assumptions": 3–6 risks with severity + mitigation; avoid generic risks that don’t fit the product type.
    - "success_learning_criteria": Be decisive about stopping rules and rollback, aligned to the key metric and guardrails.
    - "next_steps": 3–6 verb-first, assignable items with discipline tags (e.g., [Design], [Data], [Eng]).

    IMPORTANT:
    - Do NOT hallucinate numbers; when missing, keep numeric fields null but provide a short justification in statistical_rationale.
    - Make every section feel written for the supplied product type and persona—not a template.

    Return only valid JSON.
    """).strip()

    return prompt



def _build_hypothesis_prompt(context: dict) -> str:
    """
    Strongly personalized hypothesis generator.
    Forces the LLM to tie every hypothesis to ALL user inputs.
    """
    business_goal = context.get("business_goal", "")
    product_type = context.get("product_type", "")
    user_persona = context.get("user_persona", "")
    key_metric = context.get("key_metric", "")
    current_value = context.get("current_value", "")
    target_value = context.get("target_value", "")

    return f"""
You are acting as a Senior Product Manager designing A/B tests.

The user has provided the following context:
- Business Goal: {business_goal}
- Product Type: {product_type}
- Target User Persona: {user_persona}
- Key Metric: {key_metric}
- Current Value: {current_value}
- Target Value: {target_value}

TASK:
Generate 3 distinct, *highly personalized* hypotheses for an A/B test.
Every hypothesis MUST explicitly connect to ALL of the following:
1. The **Business Goal** ({business_goal})
2. The **Product Type** ({product_type})
3. The **Target Persona** ({user_persona})
4. The **Key Metric** ({key_metric}), including its **Current Value** ({current_value}) and **Target Value** ({target_value})

FORMAT REQUIREMENTS:
- Each hypothesis must follow this format:  
  "We believe that [specific product change for this product type]  
   will result in [measurable impact tied to the key metric]  
   for [the described persona]."
- Avoid generic or vague hypotheses. Tailor them to this exact product and persona.
- Always explain *why* this hypothesis matters in relation to the baseline and target values.

OUTPUT JSON SCHEMA:
{{
  "hypotheses": [
    {{
      "hypothesis": "string",
      "rationale": "Explain why this hypothesis is important and how it ties to the goal/metric/target persona",
      "example_implementation": "Give a concrete example of how this hypothesis could be tested in {product_type}",
      "behavioral_basis": "Describe the behavioral or psychological principle that supports this hypothesis"
    }},
    ...
  ]
}}

CONSTRAINTS:
- Each hypothesis must be unique, non-overlapping, and specific.
- Avoid boilerplate language like "increase engagement" without details — be concrete.
- Ensure hypotheses are testable within an A/B framework.

RETURN ONLY VALID JSON.
"""



def generate_hypotheses(context: Dict[str, Any]) -> str:
    """
    Generate 3 alternative hypotheses given the user input context.
    Returns a JSON string with this shape:
    {
      "hypotheses": [
        {"hypothesis": "..."},
        {"hypothesis": "..."},
        {"hypothesis": "..."}
      ]
    }
    """
    prompt = f"""
    You are an expert Product Manager.
    Based on this experiment context:

    {json.dumps(context, indent=2)}

    Generate 3 distinct, concise hypotheses in the format:
    "If [change], then [outcome], because [rationale]."

    Return ONLY valid JSON with this shape:
    {{
      "hypotheses": [
        {{"hypothesis": "Hypothesis 1"}},
        {{"hypothesis": "Hypothesis 2"}},
        {{"hypothesis": "Hypothesis 3"}}
      ]
    }}
    """

    llm_resp = _call_llm(prompt, max_tokens=600, temperature=0.7)
    if llm_resp:
        parsed = extract_json(llm_resp)
        if parsed and "hypotheses" in parsed:
            # Ensure each hypothesis is a dict
            fixed = []
            for h in parsed["hypotheses"]:
                if isinstance(h, str):
                    fixed.append({"hypothesis": h})
                elif isinstance(h, dict) and "hypothesis" in h:
                    fixed.append(h)
            return _safe_json_dumps({"hypotheses": fixed})
        else:
            return llm_resp

    # Fallback stub
    return _safe_json_dumps({
        "hypotheses": [
            {"hypothesis": "If we simplify the signup flow, then completion rate will improve because friction is reduced."},
            {"hypothesis": "If we personalize homepage content, then engagement will rise because relevance increases."},
            {"hypothesis": "If we highlight discounts earlier in checkout, then conversions will improve because users feel higher urgency."}
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


def _build_prd_prompt(context: dict, hypothesis: dict) -> str:
    """
    Strongly personalized PRD generator.
    Forces LLM to use ALL provided user inputs and selected hypothesis
    to generate a detailed, professional-grade PRD.
    """
    business_goal = context.get("business_goal", "")
    product_type = context.get("product_type", "")
    user_persona = context.get("user_persona", "")
    key_metric = context.get("key_metric", "")
    current_value = context.get("current_value", "")
    target_value = context.get("target_value", "")
    hypothesis_text = hypothesis.get("hypothesis", "")
    rationale = hypothesis.get("rationale", "")
    example = hypothesis.get("example_implementation", "")
    basis = hypothesis.get("behavioral_basis", "")

    return f"""
You are an expert Product Manager writing an A/B test Product Requirements Document (PRD).

The user has provided the following inputs:
- Business Goal: {business_goal}
- Product Type: {product_type}
- Target User Persona: {user_persona}
- Key Metric: {key_metric}
- Current Value: {current_value}
- Target Value: {target_value}
- Selected Hypothesis: {hypothesis_text}
- Rationale: {rationale}
- Example Implementation: {example}
- Behavioral Basis: {basis}

TASK:
Write a complete, structured PRD for this experiment. The PRD must be deeply personalized to these inputs. 
Do not generate generic templates — all sections should directly connect to the provided context.

PRD STRUCTURE (JSON ONLY):
{{
  "Experiment Title & Metadata": {{
    "title": "Short, specific title tied to {business_goal} in {product_type}",
    "owner": "Auto-fill as 'PM User'",
    "team": "Product Growth / Experimentation",
    "id": "Generate a unique experiment ID"
  }},
  "Problem Statement": "Detailed problem/opportunity statement that explains why improving {key_metric} for {user_persona} matters. Must reference {current_value} and {target_value}.",
  "Hypothesis": "{hypothesis_text}",
  "Proposed Solution & Variants": {{
    "control": "Describe the current {product_type} experience baseline",
    "variant": "Describe the new variation being tested, explicitly tied to the hypothesis"
  }},
  "Success Metrics & Guardrails": {{
    "primary_metric": "{key_metric}",
    "secondary_metrics": [
      "At least 2 other metrics relevant to {business_goal} and {product_type}"
    ],
    "guardrail_metrics": [
      "Metrics that ensure no harm to retention, performance, or user trust"
    ]
  }},
  "Experiment Design & Rollout Plan": {{
    "sample_size_per_variant": "Estimate with rationale, tied to {current_value} → {target_value}",
    "total_sample_size": "Total across all variants",
    "confidence_level": "Default 95%",
    "statistical_power": "Default 80%",
    "mde": "Estimate based on {current_value} → {target_value}",
    "dau_coverage": "Suggested rollout percentage with justification"
  }},
  "Risks & Mitigation": [
    "List at least 3 experiment-specific risks and their mitigations"
  ],
  "Success & Learning Criteria": "Define what success looks like, including required statistical confidence, and how learnings will apply even if the hypothesis is disproved"
}}

CONSTRAINTS:
- All content must reference the actual inputs. 
- Do not return placeholders like 'TBD'.
- Keep the tone professional and PRD-ready.
- Return only valid JSON.
"""


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
    "generate_hypotheses",   # <-- add this line
    "extract_json",
]
