# prompt_engine.py - Complete Fixed Version (Part 1/2)

import os
import json
import re
from typing import Dict, Any, List, Optional
from io import BytesIO
from groq import Groq
import streamlit as st


# ============ LLM Client Setup ============
try:
    from groq import Groq
    GROQ_AVAILABLE = True
    try:
        _client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        if not _client:
            print("🔴 Groq client failed: No API key found")
    except Exception as e:
        print(f"🔴 Groq client failed: {str(e)}")
        _client = None
except ImportError:
    GROQ_AVAILABLE = False
    _client = None
    print("⚠️ Groq package not available. LLM features disabled.")

# ============ Constants ============
DEFAULT_MODEL = "mixtral-8x7b-32768"
DEFAULT_TEMPERATURE = 0.7

# ============ Enhanced Prompt Templates ============
PROMPTS = {
    "hypotheses": """You are an expert Product Manager specializing in behavioral psychology. Generate 3 A/B test hypotheses with:

1. SPECIFIC CHANGE: The exact UI/feature to modify (e.g. "purchase button color")
2. METRIC IMPACT: Quantified prediction (e.g. "increase ARPU by 12%")
3. RATIONALE: Data/psychology basis (e.g. "red triggers urgency per color psychology studies")
4. IMPLEMENTATION: Technical details (e.g. "Change #buy-btn from blue to #FF0000")
5. BEHAVIORAL BASIS: Academic theory (e.g. "Color Affordance Theory (Norman, 1988)")

Format each EXACTLY like this:
{
  "variable": "specific_element",
  "prediction": "metric_impact",
  "rationale": "scientific_basis", 
  "implementation": "technical_steps",
  "behavioral_basis": "theory(reference)"
}

Business Context:
- Goal: {business_goal}
- Product: {product_type}
- Persona: {user_persona}
- Metric: {key_metric} (Current: {current_value} → Target: {target_value})""",

    "prd": """You are a senior product manager. Generate a complete A/B Test PRD in structured JSON matching this exact structure:
{
  "metadata": {
    "title": "...",
    "team": "...",
    "owner": "...",
    "experiment_id": "..."
  },
  "problem_statement": "...",
  "hypotheses": [
    {
      "hypothesis": "...",
      "rationale": "...",
      "example_implementation": "...",
      "behavioral_basis": "..."
    }
  ],
  "proposed_solution": "...",
  "variants": [
    {
      "control": "...",
      "variation": "...",
      "notes": "..."
    }
  ],
  "metrics": [
    {
      "name": "...",
      "formula": "...",
      "importance": "Primary/Secondary"
    }
  ],
  "guardrail_metrics": [
    {
      "name": "...",
      "direction": "Increase/Decrease/No Change",
      "threshold": "..."
    }
  ],
  "experiment_design": {
    "traffic_allocation": "...",
    "sample_size_per_variant": 0,
    "total_sample_size": 0,
    "test_duration_days": 0,
    "dau_coverage_percent": 0.0,
    "power": 80.0
  },
  "success_criteria": {
    "confidence_level": 95.0,
    "power": 80.0,
    "MDE": 1.0,
    "benchmark": "...",
    "monitoring": "..."
  },
  "success_learning_criteria": {
    "definition_of_success": "...",
    "stopping_rules": "...",
    "rollback_criteria": "..."
  },
  "risks_and_assumptions": [
    {
      "risk": "...",
      "severity": "High/Medium/Low",
      "mitigation": "..."
    }
  ],
  "statistical_rationale": "..."
}""",

    "tips": """You are a PM mentor providing 3-5 short, practical tips for A/B testing at step: {step}.
Context: {context}
Format as a JSON list: ["Tip 1", "Tip 2", "Tip 3"]""",

    "validate": """Review this experiment plan for quality issues:
{plan}
Return JSON with: {
  "missing_fields": ["..."],
  "inconsistencies": ["..."],
  "suggestions": ["..."]
}"""
}

# ============ Robust JSON Extraction ============
def extract_json_from_text(text: str) -> dict:
    """Improved JSON extraction with multiple fallback methods"""
    if not text:
        return {}
    
    text = text.strip()
    
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Handle code block formatting
    json_pattern = r'```(?:json)?\n([\s\S]*?)\n```'
    matches = re.findall(json_pattern, text)
    if matches:
        try:
            return json.loads(matches[0])
        except json.JSONDecodeError:
            pass
    
    # Try to find outermost JSON block
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass
    
    # Try array format
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start:end+1])
            return {"list": parsed} if isinstance(parsed, list) else {}
        except json.JSONDecodeError:
            pass
    
    # If all else fails, try to parse as a single JSON object even if malformed
    try:
        # Try to fix common issues like trailing commas
        fixed = re.sub(r',\s*([}\]])', r'\1', text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        return {}

# ============ Enhanced LLM Calling ============
def safe_call_llm(prompt: str, model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE) -> str:
    if not GROQ_AVAILABLE or _client is None:
        return json.dumps({"error": "LLM service not configured"})  # Return structured error
    
    try:
        completion = _client.chat.completions.create(
            # ... existing code ...
        )
        if not completion.choices:
            return json.dumps({"error": "Empty response from LLM"})
        return completion.choices[0].message.content
    except Exception as e:
        return json.dumps({"error": str(e)})

# ============ Fixed Hypothesis Generation ============
def generate_hypotheses(context: dict) -> List[Dict[str, str]]:
    """Generate 3 complete hypotheses with all required components"""
    if not isinstance(context, dict):
        context = {}
    
    # Validate required fields
    required_fields = ['business_goal', 'product_type', 'key_metric']
    if any(field not in context for field in required_fields):
        return [{
            "error": "Missing required fields",
            "hypothesis": "Please provide business_goal, product_type and key_metric",
            "rationale": "",
            "example_implementation": "",
            "behavioral_basis": ""
        }]
    
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

    # Transform response to expected format
    hypotheses = []
    if parsed:
        # Handle different response formats
        items = []
        if isinstance(parsed, list):
            items = parsed
        elif "hypotheses" in parsed:
            items = parsed["hypotheses"]
        elif isinstance(parsed, dict):
            # Check if it's a single hypothesis in the expected format
            if all(k in parsed for k in ["variable", "prediction", "rationale"]):
                items = [parsed]
        
        for item in items[:3]:  # Only take first 3
            if not isinstance(item, dict):
                continue
                
            hypotheses.append({
                "hypothesis": f"If we change {item.get('variable', '[variable]')}, then {item.get('prediction', '[metric impact]')}",
                "rationale": item.get("rationale", ""),
                "example_implementation": item.get("implementation", ""),
                "behavioral_basis": item.get("behavioral_basis", "")
            })
    
    # Fallback if generation failed
    if not hypotheses:
        hypotheses.append({
            "hypothesis": "If we change [variable], then [metric] will improve",
            "rationale": "AI generation failed - check API key and inputs",
            "example_implementation": "Try providing more specific context",
            "behavioral_basis": "N/A"
        })
    
    return hypotheses

def generate_hypothesis_details(hypothesis: str, context: dict) -> dict:
    """Wrapper for backward compatibility"""
    return expand_hypothesis_with_details(hypothesis, context)

def expand_hypothesis_with_details(hypothesis: str, context: dict) -> dict:
    """Enrich a hypothesis with rationale, example, and behavioral basis"""
    if not hypothesis or not isinstance(context, dict):
        return {
            "hypothesis": hypothesis or "",
            "rationale": "",
            "example_implementation": "",
            "behavioral_basis": ""
        }
    
    prompt = f"""Expand this hypothesis into full PRD-ready details:
    
Original Hypothesis: {hypothesis}

Business Context:
- Goal: {context.get('business_goal', '')}
- Product: {context.get('product_type', '')}
- Users: {context.get('user_persona', '')}
- Metric: {context.get('key_metric', '')}

Provide:
1. Detailed technical implementation
2. Behavioral psychology basis with academic reference
3. Expected impact range

Return JSON with:
{{
  "hypothesis": "refined_statement",
  "rationale": "detailed_explanation",
  "example_implementation": "technical_steps",
  "behavioral_basis": "theory(reference)"
}}"""
    
    raw = safe_call_llm(prompt, temperature=0.5)
    parsed = extract_json_from_text(raw)
    
    return {
        "hypothesis": parsed.get("hypothesis", hypothesis).strip(),
        "rationale": parsed.get("rationale", "").strip(),
        "example_implementation": parsed.get("example_implementation", "").strip(),
        "behavioral_basis": parsed.get("behavioral_basis", "").strip()
    } if parsed else {
        "hypothesis": hypothesis,
        "rationale": "Expansion failed - check API",
        "example_implementation": "",
        "behavioral_basis": ""
    }
# prompt_engine.py - Complete Fixed Version (Part 2/2)

# ============ PRD Generation ============
def generate_experiment_plan(context: dict, hypothesis: dict) -> dict:
    """Generate full PRD from context and hypothesis"""
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
    """Validate PRD for completeness and quality"""
    if not isinstance(plan, dict):
        return {
            "missing_fields": ["Plan is not a valid dictionary"],
            "inconsistencies": ["Invalid plan structure"],
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
            "inconsistencies": ["Could not validate plan - validation service unavailable"],
            "suggestions": []
        }
    
    return {
        "missing_fields": parsed.get("missing_fields", []),
        "inconsistencies": parsed.get("inconsistencies", []),
        "suggestions": parsed.get("suggestions", [])
    }

def generate_prd(context: dict, hypothesis: dict) -> dict:
    """Alias for generate_experiment_plan for backward compatibility"""
    return generate_experiment_plan(context, hypothesis)

# ============ Tips Generation ============
def generate_dynamic_tips(context: dict, current_step: str) -> List[str]:
    """Generate contextual tips - main.py expects this exact function signature"""
    return generate_tips(current_step, context)

def generate_tips(step: str, context: dict) -> List[str]:
    """Generate contextual PM tips for a given step"""
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
        return [line.strip() for line in raw.split("\n") if line.strip()][:5]
    
    return get_fallback_tips(step)

def get_fallback_tips(step: str) -> List[str]:
    """Static fallback tips when LLM is unavailable"""
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

# ============ Sanitization Utilities ============
def sanitize_experiment_plan(raw_plan: dict) -> dict:
    """Ensure PRD has all required fields with proper types"""
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
    """Alias for sanitize_experiment_plan for compatibility"""
    return sanitize_experiment_plan(plan)
