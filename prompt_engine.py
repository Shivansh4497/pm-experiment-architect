def generate_experiment_plan(goal, context):
    import streamlit as st
    from groq import Groq
    import os

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    unit = context.get("metric_unit", "")
    context["expected_lift_with_unit"] = f"{context['expected_lift']}{unit}"
    context["mde_with_unit"] = f"{context['minimum_detectable_effect']}%"
    strategic_goal = context.get('strategic_goal', '')
    
    prompt = f"""
You are an expert product manager. Your primary objective is to generate an A/B test plan that directly supports the following high-level business objective:
High-level business objective: {strategic_goal}
Product type: {context['type']}
User base size (DAU): {context['users']}
Primary metric category: {context['metric']}
Exact metric to improve: {context['exact_metric']}
Current value: {context['current_value']}
Target value: {context['target_value']}
Expected lift: {context['expected_lift']}{unit}
Minimum detectable effect (MDE): {context['minimum_detectable_effect']}%

Notes: {context['notes']}
Target user persona: {user_persona}

Product goal: "{goal}"

Return a JSON with the following keys:

- problem_statement: exactly 2–3 clear sentences. Must frame the problem as a barrier to achieving the high-level business objective. Include the metric name, current value, and target value*. Include the risk of not improving. DO NOT return placeholder text, special symbols, or markdown.

- hypotheses: list of 2–3 actionable, testable ideas. Each hypothesis must be a plausible solution to the problem statement, directly align with the high-level business objective, and be tailored to the target user persona if provided. Avoid general ideas. Each item should be a JSON object:
  {{
    "hypothesis": "1-line summary of what change you're testing and why it might improve the metric (10–20 words max)",
    "description": "Optional: add brief context or reasoning"
  }}
  Hypotheses must be clear, specific, and plausible. Avoid vague ideas like 'improve onboarding' — be explicit.
- variants: for each hypothesis, include a control and a variation. Example:
  [
    {{"hypothesis": "...", "control": "...", "variation": "..."}},
    ...
  ]
- metrics: list of 2–4 metric objects, each with:
  - name: e.g. "Activation Rate"
  - formula: e.g. "Activated Users / Signups"
- segments: relevant user groups to break down results by (e.g. "New Android Users", "Returning Gamers"). Must include the target user persona as one of the segments.
- success_criteria: include confidence_level, expected_lift, MDE, and estimated_test_duration in days
- effort: list of {{"hypothesis": "...", "effort": "Low/Medium/High"}}
- team_involved: list of functions needed (e.g. Design, Data, Backend)
- hypothesis_rationale: list of rationale objects. Each object must match the corresponding hypothesis. Structure:
  [
    {{"rationale": "Short, clear explanation (2–4 lines) of why this hypothesis is worth testing. Must include user insight or behavior pattern."}},
    ...
  ]
  DO NOT return just strings or single words. Each entry must be a dict with a full rationale explanation.
- risks_and_assumptions: list of 2–4 potential risks or assumptions that could impact test outcomes
- next_steps: list of clear action items to begin the experiment

⚠️ IMPORTANT FORMAT RULES:
- DO NOT include markdown or extra commentary — only clean JSON.
- DO NOT wrap numbers or % symbols in quotes.
- JSON must start with {{ and end with }}. Must be parsable.
    """

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are a structured, execution-focused product strategist."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
    )

    return response.choices[0].message.content.strip()
