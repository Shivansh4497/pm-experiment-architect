# prompt_engine.py
"""
Ultimate prompt engine with:
1. Hyper-contextual responses using all user inputs
2. Three detailed hypotheses with implementation examples
3. Google-level rigor without desperate naming
4. Built-in validation layer
5. Maintained schema compatibility
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
You are a Principal Product Manager reviewing an experiment plan.
Perform a quality check with these lenses:

CONTEXT:
- Product: {context.get('type', '')} ({context.get('users', '')} DAU)
- Persona: {context.get('user_persona', '')}
- Metric: {context.get('exact_metric', '')} ({context.get('current_value', '')} → {context.get('target_value', '')})
- Goal: {context.get('strategic_goal', '')}

VALIDATION CRITERIA:
1. Statistical Soundness:
   - Is MDE {prd.get('success_criteria', {}).get('MDE', '')}% realistic for {context.get('type', '')}?
   - Does sample size account for novelty effects?

2. Hypothesis Quality (3 required):
   - Are all hypotheses testable and specific to the persona?
   - Do examples show concrete product implementations?

3. Risk Coverage:
   - Are severity indicators properly calibrated?
   - Do mitigations match risk levels?

4. Professional Standards:
   - Would this pass an enterprise PRD review?
   - Are success criteria ambitious but achievable?

PRD TO REVIEW:
{json.dumps(prd, indent=2)}

Return JSON with:
{{
    "is_valid": boolean,
    "critical_issues": [str],
    "suggested_improvements": [str],
    "pro_tips": [str]  # No "Google" naming
}}
"""

def _build_main_prompt(goal: str, context: Dict[str, Any]) -> str:
    """Build the main LLM instruction prompt with maximum context utilization"""
    ctx = {k: ("" if v is None else str(v)) for k, v in context.items()}

    return textwrap.dedent(f"""
    You are a Principal Product Manager crafting an enterprise-grade experiment plan.
    Use ALL provided context to create hyper-personalized outputs.

    CONTEXT DEEP DIVE:
    - Product Type: {ctx.get('type', '')}
    - User Base: {ctx.get('users', '')}
    - Persona Traits: {ctx.get('user_persona', '')}
    - Current Metric: {ctx.get('current_value', '')}{ctx.get('metric_unit', '')}
    - Target Metric: {ctx.get('target_value', '')}{ctx.get('metric_unit', '')}
    - Strategic Goal: {ctx.get('strategic_goal', '')}
    - Metric Type: {ctx.get('metric_type', '')}
    - Data Notes: {ctx.get('notes', '')}

    REQUIREMENTS:
    1. Problem Statement:
       - Start with a data-driven insight
       - Explain persona-specific impact
       - Connect to business goal

    2. Three Hypotheses (each):
       - Mechanism: "If [change], then [outcome] because [rationale]"
       - Behavioral Basis: Psychology/data reference
       - Implementation: Exactly how to build it
       - Example: "For {ctx.get('type', '')}, this would look like [concrete example]"

    3. Professional Enhancements:
       - Add benchmark data for similar products
       - Include monitoring protocols
       - Specify severity-based mitigations

    OUTPUT SCHEMA:
    {{
      "problem_statement": str,
      "hypotheses": [
        {{
          "hypothesis": str,
          "rationale": str,
          "example_implementation": str,
          "behavioral_basis": str
        }},
        {{...}}  # 3 total
      ],
      "variants": [{{"control": str, "variation": str}}],
      "metrics": [{{"name": str, "formula": str, "importance": str}}],
      "success_criteria": {{
        "confidence_level": num,
        "MDE": num,
        "benchmark": str,
        "monitoring": str
      }},
      "risks_and_assumptions": [
        {{
          "risk": str,
          "severity": "High/Medium/Low",
          "mitigation": str
        }}
      ],
      "statistical_rationale": str
    }}

    EXAMPLE HYPOTHESIS:
    {{
      "hypothesis": "If we add progress indicators, onboarding completion will increase by 15%",
      "rationale": "Progress bars reduce abandonment by managing expectations",
      "example_implementation": "Add 3-step tracker with % complete",
      "behavioral_basis": "Zeigarnik effect (unfinished tasks create mental tension)"
    }}
    """).strip()

def _enrich_output(raw_prd: str, context: Dict[str, Any]) -> str:
    """Ensure output quality standards"""
    try:
        prd = json.loads(raw_prd)
        
        # Guarantee 3 hypotheses
        hypotheses = prd.get("hypotheses", [])
        while len(hypotheses) < 3:
            hypotheses.append({
                "hypothesis": f"Secondary lever for {context.get('exact_metric', 'metric')} improvement",
                "rationale": "Complements primary hypotheses through [mechanism]",
                "example_implementation": f"Example: {context.get('type', 'Product')} could [action]",
                "behavioral_basis": "Supported by [principle]"
            })
        prd["hypotheses"] = hypotheses

        # Enhance problem statement
        current = f"{context.get('current_value', '')}{context.get('metric_unit', '')}"
        target = f"{context.get('target_value', '')}{context.get('metric_unit', '')}"
        prd["problem_statement"] = (
            f"OPPORTUNITY: {prd.get('problem_statement', '')}\n\n"
            f"Current {context.get('exact_metric', 'metric')}: {current} → "
            f"Target: {target}\n"
            f"Strategic Impact: {context.get('strategic_goal', '')}"
        )

        return json.dumps(prd)
    except Exception:
        return raw_prd

def _validate_prd(raw_prd: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the PRD against professional standards"""
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
                    "content": "You are a meticulous PRD reviewer"
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
    """Generate a professional experiment plan with validation"""
    if not GROQ_AVAILABLE:
        return json.dumps({
            "error": "Groq client not available",
            "schema_fallback": True
        })
    
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{
                "role": "system",
                "content": "You are a principal PM who explains complex concepts simply"
            }, {
                "role": "user",
                "content": _build_main_prompt(goal, context)
            }],
            temperature=0.3,
            max_tokens=3000
        )
        raw_prd = response.choices[0].message.content.strip()
        enriched_prd = _enrich_output(raw_prd, context)
        validation = _validate_prd(enriched_prd, context)
        
        if not validation.get("is_valid", True):
            try:
                prd = json.loads(enriched_prd)
                prd["validation_feedback"] = validation
                return json.dumps(prd)
            except Exception:
                return enriched_prd
        return enriched_prd
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "schema_fallback": True
        })
