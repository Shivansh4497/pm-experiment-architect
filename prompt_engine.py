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

def _build_validation_prompt(prd: Dict[str, Any], context: Dict[str, Any]) -> str:
    """Create a rigorous validation prompt for the generated PRD."""
    return textwrap.dedent(f"""
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
    """).strip()

def _build_main_prompt(goal: str, context: Dict[str, Any]) -> str:
    """Build the main LLM instruction prompt with maximum context utilization."""
    ctx = {k: ("" if v is None else str(v)) for k, v in context.items()}
    
    prompt_intro = f"You are a Principal Product Manager at a company working on a {ctx.get('type', 'SaaS')} product. Your primary goal is to address the strategic business objective of '{ctx.get('strategic_goal', 'improving a key metric')}'. The specific challenge is to find an effective way to move a crucial metric: {ctx.get('exact_metric', 'an unspecified metric')} from its current value of {ctx.get('current_value', 'N/A')}{ctx.get('metric_unit', '')} to a target of {ctx.get('target_value', 'N/A')}{ctx.get('metric_unit', '')}."
    
    if ctx.get('user_persona'):
        prompt_intro += f" The experiment should be specifically tailored to the user persona: '{ctx.get('user_persona', '')}'."
    
    # A new line has been added to the prompt to explicitly enforce the complete JSON structure.
    # This prevents the LLM from omitting sections like "risks_and_assumptions" or "next_steps".
    full_prompt = textwrap.dedent(f"""
    {prompt_intro}
    
    Your output MUST be a valid JSON object. All keys in the schema below must be present in the final output. If a section has no content, use an empty list or an empty string.
    
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
    - Standard Deviation: {ctx.get('std_dev', 'N/A')}

    OUTPUT SCHEMA:
    {{
      "problem_statement": str,
      "hypotheses": [
        {{
          "hypothesis": str,
          "rationale": str,
          "example_implementation": str,
          "behavioral_basis": str
        }}
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
      "next_steps": [str],
      "statistical_rationale": str
    }}

    Based on the provided context, please generate the JSON output for a complete experiment plan.
    """).strip()

    return full_prompt

def _build_hypothesis_prompt(hypothesis_text: str, context: Dict[str, Any]) -> str:
    """Builds a prompt specifically for detailing a single hypothesis."""
    ctx = {k: ("" if v is None else str(v)) for k, v in context.items()}
    return textwrap.dedent(f"""
    You are a Principal Product Manager. Your task is to expand a one-sentence hypothesis into a detailed, structured format for an A/B test. The test is for a {ctx.get('type', 'SaaS')} product.

    You MUST return a valid JSON object following the schema below.

    INPUT HYPOTHESIS: "{hypothesis_text}"

    CONTEXT:
    - Strategic Goal: "{ctx.get('strategic_goal', '')}"
    - Metric to Improve: "{ctx.get('metric_to_improve', '')}"
    - Problem Statement: "{ctx.get('problem_statement', '')}"
    - User Persona: "{ctx.get('user_persona', '')}"

    OUTPUT SCHEMA:
    {{
      "hypothesis": str,
      "rationale": str,
      "example_implementation": str,
      "behavioral_basis": str
    }}

    REQUIREMENTS:
    - Rationale: Explain the data or logic behind the hypothesis in a professional tone.
    - Example Implementation: Provide a concrete, actionable example of how the test would be set up.
    - Behavioral Basis: Name a recognized psychological or behavioral science principle that supports the hypothesis (e.g., 'Hick's Law', 'Loss Aversion', 'Social Proof').
    """)

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

def generate_hypothesis_details(hypothesis_text: str, context: Dict[str, Any]) -> Optional[str]:
    """
    Generates detailed, structured information for a single hypothesis.
    """
    if not GROQ_AVAILABLE:
        return json.dumps({
            "error": "Groq client not available. Please ensure the groq library is installed and the GROQ_API_KEY environment variable is set."
        })

    prompt = _build_hypothesis_prompt(hypothesis_text, context)
    client = _get_llm_client()
    try:
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Hypothesis to expand: {hypothesis_text}"}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"LLM call failed: {e}")
        return json.dumps({"error": f"LLM generation failed: {e}"})
