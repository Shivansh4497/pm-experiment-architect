file_manager.write(
    path='prompt_engine.py',
    content='''# prompt_engine.py — Full corrected version with Groq API integration

import os
import json
import re
from typing import Dict, Any, List, Optional
from io import BytesIO

# ============ LLM Client Setup ============
try:
    # Import guarded so module still loads if Groq not installed
    from groq import Groq  # type: ignore
    GROQ_AVAILABLE = True
    # instantiate safely (expect user to set GROQ_API_KEY in env)
    try:
        _client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    except Exception as e:
        print(f"Groq client failed to initialize: {e}")
        _client = None
except ImportError:
    GROQ_AVAILABLE = False
    _client = None
    print("Warning: Groq package not available. LLM features will be disabled.")

# ============ Constants ============
DEFAULT_MODEL = "mixtral-8x7b-32768"
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

    "prd": """You are a senior product manager. Generate a complete A/B Test PRD in structured JSON matching this exact structure:
{{
  "metadata": {{
    "title": "...",
    "team": "...",
    "owner": "...",
    "experiment_id": "..."
  }},
  "problem_statement": "...",
  "hypotheses": [
    {{
      "hypothesis": "...",
      "rationale": "...",
      "example_implementation": "...",
      "behavioral_basis": "..."
    }}
  ],
  "proposed_solution": "...",
  "variants": [
    {{
      "control": "...",
      "variation": "...",
      "notes": "..."
    }}
  ],
  "metrics": [
    {{
      "name": "...",
      "formula": "...",
      "importance": "Primary/Secondary"
    }}
  ],
  "guardrail_metrics": [
    {{
      "name": "...",
      "direction": "Increase/Decrease/No Change",
      "threshold": "..."
    }}
  ],
  "experiment_design": {{
    "traffic_allocation": "...",
    "sample_size_per_variant": 0,
    "total_sample_size": 0,
    "test_duration_days": 0,
    "dau_coverage_percent": 0.0,
    "power": 80.0
  }},
  "success_criteria": {{
    "confidence_level": 95.0,
    "power": 80.0,
    "MDE": 1.0,
    "benchmark": "...",
    "monitoring": "..."
  }},
  "success_learning_criteria": {{
    "definition_of_success": "...",
    "stopping_rules": "...",
    "rollback_criteria": "..."
    }},
  "risks_and_assumptions": [
    {{
      "risk": "...",
      "severity": "High/Medium/Low",
      "mitigation": "..."
    }}
  ],
  "statistical_rationale": "..."
}}

Input Context:
{context}

Hypothesis Details:
{hypothesis}""",

    "tips": """You are a PM mentor providing 3-5 short, practical tips for A/B testing at step: {step}.
Context: {context}
Format as a JSON list: ["Tip 1", "Tip 2", "Tip 3"]""",

    "validate": """Review this experiment plan for quality issues:
{plan}
Return JSON with: {{
  "missing_fields": ["..."],
  "inconsistencies": ["..."],
  "suggestions": ["..."]
}}"""
}

# ============ Core Utilities ============
def extract_json_from_text(text: str) -> dict:
    """Robust JSON extraction from LLM output with multiple fallbacks."""
    if not text:
        return {}
    
    text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to find outermost JSON block
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

def safe_call_llm(prompt: str, model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE) -> str:
    """Wrapper for Groq API with comprehensive error handling."""
    if not GROQ_AVAILABLE or _client is None:
        return ""
    
    try:
        completion = _client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"LLM call failed: {str(e)}")
        return ""

# ============ Hypothesis Generation ============
def generate_hypotheses(context: dict) -> List[Dict[str, str]]:
    """Generate 3 hypotheses based on user context, always return list of dicts."""
    if not isinstance(context, dict):
        context = {}
    
    prompt = PROMPTS["hypotheses"].format(
        business_goal=context.get("business_goal", ""),
        product_type=context.get("product_type", ""),
        user_persona=context.get("user_persona", ""),
        key_metric=context.get("key_metric", ""),
        current_value=context.get("current_value", ""),
        target_value=context.get("target_value", "")
    )
    
    raw = safe_call_llm(prompt, temperature=0.7)
    parsed = extract_json_from_text(raw)

    hyps = []
    if parsed and "hypotheses" in parsed:
        for h in parsed["hypotheses"]:
            if isinstance(h, dict):
                hyps.append({
                    "hypothesis": h.get("hypothesis", "").strip(),
                    "rationale": h.get("rationale", "").strip(),
                    "example_implementation": h.get("example_implementation", "").strip(),
                    "behavioral_basis": h.get("behavioral_basis", "").strip()
                })
    
    # Ensure we always return at least one hypothesis
    if not hyps:
        hyps.append({
            "hypothesis": "If we change [variable], then [metric] will improve because [rationale]",
            "rationale": "",
            "example_implementation": "",
            "behavioral_basis": ""
        })
    
    return hyps

def generate_hypothesis_details(hypothesis: str, context: dict) -> dict:
    """Enrich a hypothesis with details - main.py expects this exact function name."""
    return expand_hypothesis_with_details(hypothesis, context)

def expand_hypothesis_with_details(hypothesis: str, context: dict) -> dict:
    """Enrich a hypothesis with rationale, example, and behavioral basis."""
    if not hypothesis or not isinstance(context, dict):
        return {
            "hypothesis": hypothesis or "",
            "rationale": "",
            "example_implementation": "",
            "behavioral_basis": ""
        }
    
    prompt = f"""Expand this hypothesis into full PRD-ready details:
    
Hypothesis: {hypothesis}

Context:
- Business Goal: {context.get('business_goal', '')}
- Product: {context.get('product_type', '')}
- User: {context.get('user_persona', '')}
- Metric: {context.get('key_metric', '')}

Return JSON with:
{{
  "hypothesis": "...", 
  "rationale": "...",
  "example_implementation": "...",
  "behavioral_basis": "..."
}}"""
    
    raw = safe_call_llm(prompt, temperature=0.5)
    parsed = extract_json_from_text(raw)
    
    if not parsed:
        parsed = {}
    
    return {
        "hypothesis": parsed.get("hypothesis", hypothesis).strip(),
        "rationale": parsed.get("rationale", "").strip(),
        "example_implementation": parsed.get("example_implementation", "").strip(),
        "behavioral_basis": parsed.get("behavioral_basis", "").strip()
    }

# ============ PRD Generation ============
def generate_experiment_plan(context: dict, hypothesis: dict) -> dict:
    """Generate full PRD from context and hypothesis - main.py expects this exact function."""
    if not isinstance(context, dict):
        context = {}
    if not isinstance(hypothesis, dict):
        hypothesis = {}
    
    prompt = PROMPTS["prd"].format(
        context=json.dumps(context, indent=2),
        hypothesis=json.dumps(hypothesis, indent=2)
    )
    
    raw = safe_call_llm(prompt, temperature=0.5)
    parsed = extract_json_from_text(raw)
    
    return sanitize_experiment_plan(parsed)

def validate_experiment_plan(plan: dict) -> dict:
    """Validate PRD for completeness and quality - main.py expects this exact function."""
    if not isinstance(plan, dict):
        return {
            "missing_fields": [],
            "inconsistencies": ["Plan is not a valid dictionary"],
            "suggestions": ["Please provide a valid experiment plan"]
        }
    
    prompt = PROMPTS["validate"].format(
        plan=json.dumps(plan, indent=2)
    )
    
    raw = safe_call_llm(prompt, temperature=0.3)
    parsed = extract_json_from_text(raw)
    
    if not parsed:
        return {
            "missing_fields": [],
            "inconsistencies": [],
            "suggestions": ["Could not validate plan - validation service unavailable"]
        }
    
    return {
        "missing_fields": parsed.get("missing_fields", []),
        "inconsistencies": parsed.get("inconsistencies", []),
        "suggestions": parsed.get("suggestions", [])
    }

def generate_prd(context: dict, hypothesis: dict) -> dict:
    """Alias for generate_experiment_plan for backward compatibility."""
    return generate_experiment_plan(context, hypothesis)

# ============ Tips Generation ============
def generate_dynamic_tips(context: dict, current_step: str) -> List[str]:
    """Generate contextual tips - main.py expects this exact function signature."""
    return generate_tips(current_step, context)

def generate_tips(step: str, context: dict) -> List[str]:
    """Generate contextual PM tips for a given step."""
    if not step or not isinstance(context, dict):
        return get_fallback_tips(step)
    
    prompt = PROMPTS["tips"].format(
        step=step,
        context=json.dumps(context, indent=2)
    )
    
    raw = safe_call_llm(prompt, temperature=0.6)
    parsed = extract_json_from_text(raw)
    
    if isinstance(parsed, list):
        return [str(tip) for tip in parsed][:5]
    elif isinstance(parsed, dict) and "tips" in parsed:
        return [str(tip) for tip in parsed["tips"]][:5]
    elif isinstance(raw, str):
        return [line.strip() for line in raw.split("\\n") if line.strip()][:5]
    
    return get_fallback_tips(step)

def get_fallback_tips(step: str) -> List[str]:
    """Static fallback tips when LLM is unavailable."""
    base_tips = {
        "inputs": [
            "🎯 Keep business goals specific and measurable",
            "📊 Focus on one primary metric",
            "🔢 Include baseline metrics for better estimates"
        ],
        "hypothesis": [
            "✍️ Make hypotheses falsifiable",
            "📌 Connect to user personas",
            "⚡ Start with small changes"
        ],
        "prd": [
            "🛡️ Add guardrail metrics",
            "📐 Verify sample size feasibility",
            "🔁 Define clear rollback criteria"
        ]
    }
    return base_tips.get(step, base_tips["prd"])

# ============ Sanitization ============
def sanitize_experiment_plan(raw_plan: dict) -> dict:
    """Ensure PRD has all required fields with proper types."""
    if not isinstance(raw_plan, dict):
        raw_plan = {}
    
    # Start with empty structure
    sanitized = {
        "metadata": {
            "title": str(raw_plan.get("metadata", {}).get("title", "Untitled Experiment")),
            "team": str(raw_plan.get("metadata", {}).get("team", "")),
            "owner": str(raw_plan.get("metadata", {}).get("owner", "")),
            "experiment_id": str(raw_plan.get("metadata", {}).get("experiment_id", ""))
        },
        "problem_statement": str(raw_plan.get("problem_statement", "")),
        "hypotheses": [],
        "proposed_solution": str(raw_plan.get("proposed_solution", "")),
        "variants": [],
        "metrics": [],
        "guardrail_metrics": [],
        "experiment_design": {
            "traffic_allocation": str(raw_plan.get("experiment_design", {}).get("traffic_allocation", "50/50")),
            "sample_size_per_variant": int(raw_plan.get("experiment_design", {}).get("sample_size_per_variant", 0)),
            "total_sample_size": int(raw_plan.get("experiment_design", {}).get("total_sample_size", 0)),
            "test_duration_days": int(raw_plan.get("experiment_design", {}).get("test_duration_days", 0)),
            "dau_coverage_percent": float(raw_plan.get("experiment_design", {}).get("dau_coverage_percent", 0.0)),
            "power": float(raw_plan.get("experiment_design", {}).get("power", 80.0))
        },
        "success_criteria": {
            "confidence_level": float(raw_plan.get("success_criteria", {}).get("confidence_level", 95.0)),
            "power": float(raw_plan.get("success_criteria", {}).get("power", 80.0)),
            "MDE": float(raw_plan.get("success_criteria", {}).get("MDE", 1.0)),
            "benchmark": str(raw_plan.get("success_criteria", {}).get("benchmark", "")),
            "monitoring": str(raw_plan.get("success_criteria", {}).get("monitoring", ""))
        },
        "success_learning_criteria": {
            "definition_of_success": str(raw_plan.get("success_learning_criteria", {}).get("definition_of_success", "")),
            "stopping_rules": str(raw_plan.get("success_learning_criteria", {}).get("stopping_rules", "")),
            "rollback_criteria": str(raw_plan.get("success_learning_criteria", {}).get("rollback_criteria", ""))
        },
        "risks_and_assumptions": [],
        "statistical_rationale": str(raw_plan.get("statistical_rationale", ""))
    }

    # Process lists with type checking
    for h in raw_plan.get("hypotheses", []):
        if isinstance(h, dict):
            sanitized["hypotheses"].append({
                "hypothesis": str(h.get("hypothesis", "")),
                "rationale": str(h.get("rationale", "")),
                "example_implementation": str(h.get("example_implementation", "")),
                "behavioral_basis": str(h.get("behavioral_basis", ""))
            })

    for v in raw_plan.get("variants", []):
        if isinstance(v, dict):
            sanitized["variants"].append({
                "control": str(v.get("control", "")),
                "variation": str(v.get("variation", "")),
                "notes": str(v.get("notes", ""))
            })

    for m in raw_plan.get("metrics", []):
        if isinstance(m, dict):
            sanitized["metrics"].append({
                "name": str(m.get("name", "")),
                "formula": str(m.get("formula", "")),
                "importance": "Primary" if str(m.get("importance", "")).lower() == "primary" else "Secondary"
            })

    for g in raw_plan.get("guardrail_metrics", []):
        if isinstance(g, dict):
            sanitized["guardrail_metrics"].append({
                "name": str(g.get("name", "")),
                "direction": g.get("direction", "Decrease") if g.get("direction") in ["Increase", "Decrease", "No Change"] else "Decrease",
                "threshold": str(g.get("threshold", ""))
            })

    for r in raw_plan.get("risks_and_assumptions", []):
        if isinstance(r, dict):
            sanitized["risks_and_assumptions"].append({
                "risk": str(r.get("risk", "")),
                "severity": r.get("severity", "Medium") if r.get("severity") in ["High", "Medium", "Low"] else "Medium",
                "mitigation": str(r.get("mitigation", ""))
            })

    return sanitized

def sanitize_plan(plan: dict) -> dict:
    """Alias for sanitize_experiment_plan for compatibility."""
    return sanitize_experiment_plan(plan)'''
)
