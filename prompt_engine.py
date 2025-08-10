# prompt_engine.py
"""
Enhanced prompt engine with:
1. Google-style personality and rigor
2. Output validation layer
3. Maintains exact same schema and functionality
"""

import os
import json
import textwrap
from typing import Dict, Any

# Attempt to import Groq client
try:
    from groq import Groq  # type: ignore
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

def _build_validation_prompt(prd: Dict[str, Any], context: Dict[str, Any]) -> str:
    """Create a rigorous validation prompt for the generated PRD"""
    return f"""
You are a Principal Product Manager at Google reviewing an experiment plan.
Perform a brutal quality check on this PRD:

CONTEXT:
- Product Type: {context.get('type', '')}
- User Base: {context.get('users', '')}
- Metric: {context.get('exact_metric', '')} ({context.get('current_value', '')} → {context.get('target_value', '')})
- Strategic Goal: {context.get('strategic_metric', '')}

VALIDATION CRITERIA:
1. Statistical Soundness:
   - Is the MDE realistic for this product type? (SaaS: 5-15%, Gaming: 10-20%)
   - Does the sample size account for novelty effects?
   
2. Hypothesis Quality:
   - Are hypotheses specific and testable?
   - Do they connect clearly to the target metric?
   
3. Risk Coverage:
   - Are major edge cases considered?
   - Are assumptions explicitly called out?

4. Google Standards:
   - Would this pass a Google PRD review?
   - Are success criteria ambitious but achievable?

PRD TO REVIEW:
{json.dumps(prd, indent=2)}

INSTRUCTIONS:
Return JSON with validation results and fixes:
{{
    "is_valid": boolean,
    "critical_issues": [str],
    "suggested_improvements": [str],
    "google_pro_tips": [str]
}}
"""

def _build_main_prompt(goal: str, context: Dict[str, Any]) -> str:
    """Build the main LLM instruction prompt with Google-style rigor"""
    ctx = {k: ("" if v is None else v) for k, v in context.items()}

    return textwrap.dedent(f"""
    You are a Principal Product Manager at Google with 12 years of A/B testing experience.
    Your task: Create a flawless experiment PRD that would pass Google's rigorous review process.

    STYLE GUIDE:
    - Tone: Confident but precise (like Sundar Pichai explaining ML)
    - Depth: Include Google-level insights (e.g., "For DAU <10K, consider sequential testing")
    - Structure: Mirror Google's PRD format exactly

    CONTEXT (USE VERBATIM WHERE RELEVANT):
    - Product: {ctx.get('type', '')} ({ctx.get('users', '')} DAU)
    - Persona: {ctx.get('user_persona', '')}
    - Metric: {ctx.get('exact_metric', '')} ({ctx.get('current_value', '')}{ctx.get('metric_unit', '')} → {ctx.get('target_value', '')}{ctx.get('metric_unit', '')})
    - Goal: {ctx.get('strategic_goal', '')}
    - Notes: {ctx.get('notes', '')}

    OUTPUT REQUIREMENTS:
    1. For each hypothesis, include:
       - "google_insight": Why this works at Google-scale
       - "example_impact": Concrete scenario with numbers
       - "failure_case": How we'd detect if it's not working

    2. In success_criteria:
       - Add "google_benchmark": Typical results for similar products
       - Include "monitoring_plan": How we'd track post-launch

    3. In risks_and_assumptions:
       - List "google_red_flags": What would make us pause the test
       - Add "rollback_plan": Concrete steps if metrics drop

    SCHEMA (MUST FOLLOW EXACTLY):
    {{
      "problem_statement": str,
      "hypotheses": [
        {{
          "hypothesis": str,
          "google_insight": str,
          "example_impact": str,
          "failure_case": str
        }}
      ],
      "variants": [{{"control": str, "variation": str}}],
      "metrics": [{{"name": str, "formula": str, "importance": str}}],
      "success_criteria": {{
        "confidence_level": num,
        "MDE": num,
        "google_benchmark": str,
        "monitoring_plan": str
      }},
      "risks_and_assumptions": {{
    "risk": "User fatigue from frequent changes",
    "severity_indicator": "High",
    "mitigation": "Limit to 1 test per user per week"
}},
      "statistical_rationale": str
    }}

    FINAL INSTRUCTION:
    Imagine this PRD will be reviewed by Sundar Pichai - make it bulletproof.
    """).strip()

def _validate_prd(raw_prd: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the PRD against Google standards"""
    try:
        prd = json.loads(raw_prd)
    except Exception:
        return {"error": "PRD parsing failed during validation"}
    
    if GROQ_AVAILABLE:
        try:
            client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{
                    "role": "system",
                    "content": "You are a brutal Google PRD reviewer"
                }, {
                    "role": "user",
                    "content": _build_validation_prompt(prd, context)
                }],
                temperature=0.1,
                max_tokens=2000
            )
            return json.loads(response.choices[0].message.content.strip())
        except Exception:
            return {"warning": "Validation failed - using original PRD"}
    return {}

def generate_experiment_plan(goal: str, context: Dict[str, Any]) -> str:
    """
    Generate an experiment plan with Google-level quality and validation
    Maintains EXACT same return signature and schema as original
    """
    # Step 1: Generate initial PRD
    if GROQ_AVAILABLE:
        try:
            client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{
                    "role": "system",
                    "content": "You are a Google Principal PM"
                }, {
                    "role": "user",
                    "content": _build_main_prompt(goal, context)
                }],
                temperature=0.2,
                max_tokens=2500
            )
            raw_prd = response.choices[0].message.content.strip()
        except Exception as e:
            return json.dumps({
                "error": f"Generation failed: {str(e)}",
                "schema_fallback": True  # Ensures main.py can still parse
            })
    else:
        return json.dumps({
            "error": "Groq client not available",
            "schema_fallback": True
        })

    # Step 2: Validate (but don't block on failure)
    validation_results = _validate_prd(raw_prd, context)
    
    # Step 3: Return in original expected format
    if validation_results.get("is_valid", True):
        return raw_prd
    else:
        # Merge validation feedback into PRD without breaking schema
        try:
            prd = json.loads(raw_prd)
            prd["validation_feedback"] = validation_results
            return json.dumps(prd)
        except Exception:
            return raw_prd  # Fallback to original if merge fails
