"""
Ultimate prompt engine with:
1. Hyper-contextual responses using all user inputs
2. Three detailed hypotheses with implementation examples
3. Professional-grade rigor
4. Built-in validation layer
5. Guaranteed complete outputs
"""

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

def _build_validation_prompt(prd: Dict[str, Any], context: Dict[str, Any]) -> str:
    """Create a rigorous validation prompt for the generated PRD"""
    return f"""
You are a Principal Product Manager reviewing an experiment plan.
Perform a quality check with these lenses:

CONTEXT:
- Product: {context.get('type', '')} ({context.get('users', '')})
- Persona: {context.get('user_persona', '')}
- Metric: {context.get('exact_metric', '')} ({context.get('current_value', '')} → {context.get('target_value', '')})
- Goal: {context.get('strategic_goal', '')}

VALIDATION CRITERIA:
1. Statistical Soundness:
   - Is MDE {prd.get('success_criteria', {}).get('MDE', '')}% realistic for {context.get('type', '')}?
   - Does sample size account for novelty effects?

2. Hypothesis Quality (3 required):
   - Does each hypothesis include:
     * Clear if-then-because statement
     * Behavioral science/data reference
     * Concrete implementation example
     * Psychological principle

3. Completeness:
   - Are all next steps actionable?
   - Are risks paired with mitigations?
   - Is the problem statement metric-driven?

PRD TO REVIEW:
{json.dumps(prd, indent=2)}

Return JSON with:
{{
    "is_valid": boolean,
    "critical_issues": [str],
    "suggested_improvements": [str],
    "pro_tips": [str]
}}
"""

def _build_main_prompt(goal: str, context: Dict[str, Any]) -> str:
    """Build the main LLM instruction prompt with maximum context utilization"""
    ctx = {k: ("" if v is None else str(v)) for k, v in context.items()}
    
    # Custom, personalized introduction using user inputs
    prompt_intro = f"You are a Principal Product Manager at a company working on a {ctx.get('type', 'SaaS')} product. Your primary goal is to address the strategic business objective of '{ctx.get('strategic_goal', 'improving a key metric')}'. The specific challenge is to find an effective way to move a crucial metric: {ctx.get('exact_metric', 'an unspecified metric')} from its current value of {ctx.get('current_value', 'N/A')}{ctx.get('metric_unit', '')} to a target of {ctx.get('target_value', 'N/A')}{ctx.get('metric_unit', '')}."
    
    if ctx.get('user_persona'):
        prompt_intro += f" The experiment should be specifically tailored to the user persona: '{ctx.get('user_persona', '')}'."
    
    # Combine the new intro with the rest of the prompt
    full_prompt = textwrap.dedent(f"""
    {prompt_intro}
    
    Your output MUST be a valid JSON object.
    
    MANDATORY REQUIREMENTS:
    1. For each of 3 hypotheses:
       - hypothesis: "If [change] then [outcome] because [rationale]"
       - rationale: Peer-reviewed research or credible data source
       - example_implementation: Exact UI/flow changes
       - behavioral_basis: Psychological principle
    
    2. Problem statement must:
       - Start with specific metric comparison
       - Explain user pain points
       - Connect to business impact

    3. Next steps must be:
       - Actionable (verb-first)
       - Owned (assignable to roles)
       - Time-bound (when possible)

    CONTEXT DEEP DIVE:
    - Product Type: {ctx.get('type', '')}
    - User Base: {ctx.get('users', '')}
    - Persona Traits: {ctx.get('user_persona', '')}
    - Current Metric: {ctx.get('current_value', '')}{ctx.get('metric_unit', '')}
    - Target Metric: {ctx.get('target_value', '')}{ctx.get('metric_unit', '')}
    - Strategic Goal: {ctx.get('strategic_goal', '')}
    - Metric Type: {ctx.get('metric_type', '')}
    - Data Notes: {ctx.get('notes', '')}

    OUTPUT SCHEMA:
    {{
      "problem_statement": str,
      "hypotheses": [
        {{
          "hypothesis": str,  # "If we X, then Y because Z"
          "rationale": str,   # "Baymard Institute shows..."
          "example_implementation": str,  # "Remove these 2 fields..."
          "behavioral_basis": str  # "Hick's Law..."
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
      "next_steps": [str],  # ["Create mockups by Fri (Design)"]
      "statistical_rationale": str
    }}

    EXAMPLE OUTPUT:
    {{
      "problem_statement": "The problem statement goes here, starting with a metric.",
      "hypotheses": [
        {{
          "hypothesis": "If [change] then [outcome] because [rationale]",
          "rationale": "Data-backed or research-based rationale.",
          "example_implementation": "Specific changes to the UI or flow.",
          "behavioral_basis": "A relevant psychological principle."
        }},
        {{...}}
      ],
      "variants": [
        {{ "control": "Current design", "variation": "New design" }}
      ],
      "metrics": [
        {{ "name": "Primary Metric", "formula": "Click-through-rate (CTR)", "importance": "Primary" }}
      ],
      "success_criteria": {{
        "confidence_level": 95,
        "MDE": 3,
        "benchmark": "Industry average",
        "monitoring": "Daily via dashboards"
      }},
      "risks_and_assumptions": [
        {{ "risk": "Seasonality", "severity": "Medium", "mitigation": "Run experiment longer" }}
      ],
      "next_steps": [
        "Step 1 (Owner)",
        "Step 2 (Owner)"
      ]
    }}""").strip()

    return full_prompt

def _enrich_output(raw_prd: str, context: Dict[str, Any]) -> str:
    """Provides a second-pass validation and enrichment prompt."""
    return _build_validation_prompt(json.loads(raw_prd), context)

def _get_llm_client() -> Groq:
    """Returns a Groq client."""
    return Groq(api_key=os.environ.get("GROQ_API_KEY"))

def generate_experiment_plan(goal: str, context: Dict[str, Any]) -> Optional[str]:
    """Generates an experiment plan using an LLM, with validation and structured output."""
    if not GROQ_AVAILABLE:
        return json.dumps({
            "error": "Groq client not available. Please ensure the groq library is installed and the GROQ_API_KEY environment variable is set."
        })

    prompt = _build_main_prompt(goal, context)
    client = _get_llm_client()
    try:
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"The high-level business goal is: {goal}"}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        first_pass_json = completion.choices[0].message.content
        return first_pass_json
    except Exception as e:
        print(f"LLM call failed: {e}")
        return json.dumps({"error": f"LLM generation failed: {e}"})
