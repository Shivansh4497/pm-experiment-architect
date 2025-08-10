# prompt_engine.py
"""
Enhanced prompt engine for the A/B Test Architect app.

Upgrades:
- Forces usage of all context fields, even if empty (with impact explained in risks_and_assumptions).
- Hypotheses now include a concrete "example_impact" field with a realistic product scenario.
- Rationales, field_importance, and risks are more targeted and persona-aware.
- Maintains EXACT SAME SCHEMA as before for main.py compatibility.
"""

import os
import json
import textwrap

# Attempt to import Groq client
try:
    from groq import Groq  # type: ignore
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False


def _build_prompt(goal: str, context: dict) -> str:
    """
    Build the main LLM instruction prompt.
    """
    ctx = {k: ("" if v is None else v) for k, v in context.items()}

    unit = ctx.get("metric_unit", "")
    expected_lift = f"{ctx.get('expected_lift', '')}{unit}".strip()
    mde = ctx.get("minimum_detectable_effect", "")
    notes = ctx.get("notes", "")
    strategic_goal = ctx.get("strategic_goal", "")
    persona = ctx.get("user_persona", "")
    metric_type = ctx.get("metric_type", "Conversion Rate")
    std_dev = ctx.get("std_dev", None)
    users = ctx.get("users", "")
    exact_metric = ctx.get("exact_metric", "")
    current_value = ctx.get("current_value", "")
    target_value = ctx.get("target_value", "")

    prompt = f"""
You are an expert Senior Product Manager and Data Scientist.
Your task: Produce a production-ready A/B test plan as a STRICT JSON object following the schema below.

CRITICAL:
1. Use EVERY provided context field — if any field is missing or blank, explain its absence in `risks_and_assumptions` with specific impact.
2. All hypotheses must be persona-aware, metric-linked, and scenario-driven.
3. For EACH hypothesis, also include an `"example_impact"` field inside the same hypothesis object:
   - This is a short, concrete product example showing how the change could improve the product, grounded in this context.
4. Rationales must tie back to the strategic goal, target persona, and provided metrics.
5. Include actual numbers (current, target, lift, unit) wherever possible in descriptions and rationales.
6. Field importance must be precise: say why it matters for decision-making.
7. Output must be valid JSON only — no markdown, commentary, or trailing commas.
8. All numeric fields in success_criteria must be numbers, not strings.

CONTEXT (verbatim when useful):
- High-level business objective: {strategic_goal}
- Product type: {ctx.get('type','')}
- Target user persona: {persona}
- Metric type: {metric_type}
- Standard deviation (if numeric metric): {std_dev}
- User base size (DAU): {users}
- Primary metric category: {ctx.get('metric','')}
- Exact metric to improve: {exact_metric}
- Current value: {current_value}
- Target value: {target_value}
- Expected lift: {expected_lift}
- Minimum detectable effect (MDE): {mde}
- Notes: {notes}

SCHEMA: Return a JSON object with EXACTLY these keys:

{{
  "problem_statement": string,              
  "field_importance": {{ key: {{"level":"High|Medium|Low","reason":"..."}} }},
  "hypotheses": [                            
    {{
      "hypothesis": string,
      "description": string,
      "example_impact": string
    }}
  ],
  "variants": [                              
    {{
      "hypothesis": string,
      "control": string,
      "variation": string
    }}
  ],
  "hypothesis_rationale": [                  
    {{ "rationale": string }}
  ],
  "metrics": [                               
    {{ "name": string, "formula": string, "importance": "High|Medium|Low" }}
  ],
  "segments": [ string ],
  "success_criteria": {{
    "confidence_level": number,
    "expected_lift": number,
    "MDE": number,
    "sample_size_required": null|number,
    "users_per_variant": null|number,
    "estimated_test_duration_days": null|number
  }},
  "effort": [ {{ "hypothesis": string, "effort": "Low|Medium|High" }} ],
  "team_involved": [ string ],
  "risks_and_assumptions": [ string ],       
  "next_steps": [ string ],                  
  "statistical_rationale": string           
}}

ADDITIONAL REQUIREMENTS:
- Each hypothesis must clearly connect to the metric and persona.
- In `example_impact`, provide a specific, short story or scenario.
- In `hypothesis_rationale`, explicitly link to strategic goal, metric type, and user behavior patterns.
- If metric data is missing, explain in `risks_and_assumptions` what effect that has.
"""

    return textwrap.dedent(prompt).strip()


def generate_experiment_plan(goal: str, context: dict) -> str:
    """
    Generate an experiment plan using Groq or return placeholder JSON if unavailable.
    """
    prompt = _build_prompt(goal, context)

    if GROQ_AVAILABLE:
        api_key = os.environ.get("GROQ_API_KEY", "") or os.environ.get("GROQ_KEY", "")
        if not api_key:
            return json.dumps({
                "error": "GROQ_API_KEY not found. Set it in your environment."
            })
        try:
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are a structured, execution-focused product strategist."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.18,
                max_tokens=2000,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return json.dumps({
                "error": "Groq request failed",
                "exception": str(e),
                "prompt_excerpt": prompt[:800]
            })
    else:
        return json.dumps({
            "error": "Groq client not available.",
            "note": "This is a placeholder output when Groq isn't installed."
        })
