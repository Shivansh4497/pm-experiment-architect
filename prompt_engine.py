# prompt_engine.py
"""
Polished prompt engine for the A/B Test Architect app.

Provides:
- generate_experiment_plan(goal, context)
  -> returns a string (LLM text) containing a JSON object (the experiment plan)

Notes:
- This function will attempt to call Groq if the Groq client is available and
  GROQ_API_KEY is present in the environment. If Groq isn't available or key
  missing, it will return a nicely formatted error string for display in the UI.
- The prompt strongly emphasises using all context fields, producing personalized
  and contextual output, strict JSON-only response, and unquoted numeric fields.
"""
import os
import json
import textwrap

# Attempt to import Groq client; if not present we'll surface an error message back to the UI.
try:
    from groq import Groq  # type: ignore
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

def _build_prompt(goal: str, context: dict) -> str:
    """
    Returns the main instruction prompt for the model.

    This prompt:
    - instructs the LLM to use ALL provided context fields
    - requests personalization and context-awareness
    - demands strict JSON output according to the schema
    - enforces JSON-only response (no markdown or commentary)
    - asks the model to indicate which input fields it considered most important
      (so downstream users can see field importance)
    """
    # Safely extract context values for inline prompt clarity
    # Default safe formatting for absent keys
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
    exact_metric = ctx.get("exact_metric", ctx.get("exact_metric", ""))
    current_value = ctx.get("current_value", "")
    target_value = ctx.get("target_value", "")

    # The prompt below is intentionally explicit and prescriptive.
    prompt = f"""
You are an expert Senior Product Manager and Data Scientist tasked with producing a production-ready A/B test plan.
Use every piece of context provided below to produce a single, valid JSON object that fully implements the schema and fields requested.

IMPORTANT:
- Use ALL fields in the context when producing hypotheses, rationale, metrics, segments and the statistical rationale. If a field is missing, explain briefly in the 'risks_and_assumptions' field why that matters (but still return valid JSON).
- Personalize each hypothesis and rationale to the target user persona and product context when available.
- For every output field, consider its relative importance and include an explicit `field_importance` object mapping key input fields to an importance level ("High"/"Medium"/"Low") and a one-line justification.
- Output STRICT JSON only. No markdown, no commentary, no surrounding text, no code fences, and no trailing commas.
- Numeric fields (confidence_level, MDE, estimated_test_duration, sample sizes) must be numbers (not strings). Percent signs may be included where appropriate but do not quote numbers as strings.
- Where lists are requested, return arrays. Where objects are requested, return objects.
- Compose concise but concrete hypotheses (10-20 words) and a 2–4 line rationale per hypothesis grounded in user behavior or data signals in the context.
- For the `statistical_rationale` field, explicitly reference the metric type (Conversion Rate vs Numeric Value), the test method (e.g., two-sample proportions test or t-test), and show how the MDE & user base inform sample size/duration qualitatively.
- When you refer to the metric, include the metric unit (e.g., % or minutes) and current vs target values in the rationale when relevant.
- If you cannot compute a numerical estimate (e.g., sample size) due to missing inputs, set those fields to null and explain which inputs are missing in `risks_and_assumptions`.

CONTEXT (use this verbatim when helpful):
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

SCHEMA: Return a JSON object with these keys (fill thoughtfully):

{{
  "problem_statement": string,              // 2-3 sentences. Include metric with current and target values and a risk of not improving.
  "field_importance": {{                     // map of input fields -> "High"/"Medium"/"Low" and a 1-line reason
     "strategic_goal": {{"level":"High","reason":"..."}},
     "...": {{"level":"Medium","reason":"..."}}
  }},
  "hypotheses": [                            // 2-3 distinct hypothesis objects
    {{
      "hypothesis": string,                 // 10–20 words summary of the change and why
      "description": string                 // 1–2 sentences with persona-specific detail if available
    }}
  ],
  "variants": [                              // same length as hypotheses
    {{
      "hypothesis": string,
      "control": string,
      "variation": string
    }}
  ],
  "hypothesis_rationale": [                  // same order as hypotheses; must be objects
    {{ "rationale": string }}
  ],
  "metrics": [                               // 2–4 metrics (primary + secondaries)
    {{ "name": string, "formula": string, "importance": "High|Medium|Low" }}
  ],
  "segments": [ string ],                    // include persona as one segment if provided
  "success_criteria": {{
    "confidence_level": number,              // e.g., 95
    "expected_lift": number,                  // numeric percent (not string)
    "MDE": number,                            // numeric percent (not string)
    "sample_size_required": null|number,      // numeric total users required (or null if cannot compute)
    "users_per_variant": null|number,
    "estimated_test_duration_days": null|number
  }},
  "effort": [ {{ "hypothesis": string, "effort": "Low|Medium|High" }} ],
  "team_involved": [ string ],
  "risks_and_assumptions": [ string ],       // 2–4 items
  "next_steps": [ string ],                  // 3–6 actionable steps to start
  "statistical_rationale": string           // 2–3 sentences, reference metric type and method
}}

ADDITIONAL CONSTRAINTS:
- Return exactly the keys in the schema above. Additional keys are allowed only if they add value (e.g., a small `notes` object), but avoid needless verbosity.
- Keep text concise and professional. Use the target persona where relevant.
- Make the output high-quality and interview-ready.

Now produce the JSON plan for this product goal and context.
"""

    # keep indentation clean
    return textwrap.dedent(prompt).strip()

def generate_experiment_plan(goal: str, context: dict) -> str:
    """
    Generate an experiment plan using an LLM.

    - goal: short natural language description of the product goal.
    - context: dictionary containing fields used by the LLM.

    Returns: the raw string output from the model (expected to be JSON).
    """
    # Build prompt
    prompt = _build_prompt(goal, context)

    # If Groq available and an API key in environment, call it
    if GROQ_AVAILABLE:
        api_key = os.environ.get("GROQ_API_KEY", "") or os.environ.get("GROQ_KEY", "")
        if not api_key:
            # Return instructive error so the UI can show it
            return json.dumps({
                "error": "GROQ_API_KEY not found in environment. To generate plans using Groq, set GROQ_API_KEY."
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
            content = response.choices[0].message.content.strip()
            # If Groq's output sometimes prefixes things, try to strip leading/trailing non-json
            return content
        except Exception as e:
            # Return a clear diagnostic JSON allowing the UI to show the error
            return json.dumps({
                "error": "Groq request failed",
                "exception": str(e),
                "prompt_excerpt": prompt[:800]
            })
    else:
        # Groq isn't installed — return an instructional placeholder so the UI can show it to the user.
        return json.dumps({
            "error": "Groq client not available. Install the Groq SDK or set up your provider integration.",
            "note": "This placeholder is returned by prompt_engine.generate_experiment_plan when Groq isn't installed."
        })
