# prompt_engine.py

import streamlit as st
from groq import Groq
import os
import re

def generate_experiment_plan(goal, context, api_key):
    """
    Generates a detailed, highly contextual A/B test plan in JSON format
    based on a product goal and full user-provided context.
    
    This function:
    - Leverages ALL provided context fields with equal importance
    - Instructs the LLM to produce clear, personalised, and polished outputs
    - Forces a strict single JSON object output for easy parsing
    - Cleans and returns the raw JSON string without markdown/code fences
    
    Args:
        goal (str): The high-level product goal that the experiment should address.
        context (dict): A dictionary containing detailed context for the experiment.
        api_key (str): The Groq API key for authentication.

    Returns:
        str: A JSON string containing the A/B test plan.
    """
    if not api_key:
        st.error("Groq API Key is missing. Please provide it in the sidebar.")
        return None

    client = Groq(api_key=api_key)

    # Prepare context variables
    unit = context.get("metric_unit", "")
    context["expected_lift_with_unit"] = f"{context['expected_lift']}{unit}" if context.get('expected_lift') is not None else "N/A"
    context["mde_with_unit"] = f"{context['minimum_detectable_effect']}%" if context.get('minimum_detectable_effect') is not None else "N/A"
    strategic_goal = context.get('strategic_goal', '')
    user_persona = context.get('user_persona', '')
    metric_type = context.get('metric_type', 'Conversion Rate')
    std_dev = context.get('std_dev', None)

    # ---------------------------------------------------------------------
    # Core Prompt — tuned for high-quality, personalised, contextual output
    # ---------------------------------------------------------------------
    prompt = f"""
You are an expert Senior Product Manager with deep experience in experimentation, metrics, and AI-driven product strategy.
You have been given **full product context** and your task is to create an exceptionally clear, detailed, and actionable A/B test plan.

💡 CRITICAL RULES:
1. Treat EVERY provided input field as important. If a field is empty, gracefully adapt while keeping the output professional.
2. All outputs must be personalised to the exact product type, goal, and target persona provided.
3. Every recommendation must clearly connect back to the high-level business objective and provided metrics.
4. The output must be structured as ONE single valid JSON object — nothing else.
5. NO markdown, comments, or code fences. No placeholders like "TBD".
6. Every section should sound as if it was written by a senior PM presenting to leadership — crisp, data-driven, and jargon-free.

📌 PRODUCT CONTEXT (all must be used thoughtfully):
- High-level business objective: {strategic_goal}
- Product type: {context['type']}
- Target user persona: {user_persona}
- Metric type: {metric_type}
- Standard Deviation: {std_dev if std_dev else "N/A"}
- User base size (DAU): {context['users']}
- Primary metric category: {context['metric']}
- Exact metric to improve: {context['exact_metric']}
- Current value: {context['current_value']}
- Target value: {context['target_value']}
- Expected lift: {context['expected_lift_with_unit']}
- Minimum detectable effect (MDE): {context['mde_with_unit']}
- Notes: {context['notes']}
- Product goal: "{goal}"

🎯 REQUIRED JSON KEYS & STRUCTURE:
{{
  "problem_statement": "2–3 sentences framing the barrier to the high-level business objective, naming the exact metric, current and target values, and risk if not addressed.",
  "hypotheses": [
    {{
      "hypothesis": "1-line summary of the change being tested and why",
      "description": "Brief reasoning or context"
    }},
    ...
  ],
  "variants": [
    {{
      "hypothesis": "...",
      "control": "Describe control state",
      "variation": "Describe test variation"
    }},
    ...
  ],
  "metrics": [
    {{
      "name": "Metric Name",
      "formula": "Formula"
    }},
    ...
  ],
  "segments": ["List of relevant user groups, must include target persona"],
  "success_criteria": {{
    "confidence_level": "e.g. 95%",
    "expected_lift": "...",
    "MDE": "...",
    "estimated_test_duration": "Days"
  }},
  "effort": [
    {{
      "hypothesis": "...",
      "effort": "Low/Medium/High"
    }},
    ...
  ],
  "team_involved": ["Functions needed (e.g., Design, Data, Backend)"],
  "hypothesis_rationale": [
    {{
      "rationale": "2–4 line explanation linking the hypothesis to user behavior insights"
    }},
    ...
  ],
  "risks_and_assumptions": [
    "Risk 1",
    "Risk 2",
    ...
  ],
  "next_steps": [
    "Action 1",
    "Action 2"
  ],
  "statistical_rationale": "2–3 sentences explaining MDE, sample size, and test duration, referencing metric type and DAU."
}}

🚫 STRICT FORMAT RULES:
- Output ONLY the JSON object above, with double quotes for keys and strings.
- All numbers must be JSON numbers (no quotes), except where the value is text.
- No comments, markdown, or additional text before or after the JSON.
"""

    # ---------------------------------------------------------------------
    # API call to Groq
    # ---------------------------------------------------------------------
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a structured, execution-focused senior product strategist and experimentation expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
        )
    except Exception as e:
        st.error(f"Error calling LLM: {e}")
        return None

    # ---------------------------------------------------------------------
    # Clean and return content
    # ---------------------------------------------------------------------
    content = response.choices[0].message.content.strip()

    # Remove common code fences if present
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)

    # Ensure we're returning only the JSON block if extra text slipped in
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        content = match.group(0).strip()

    return content
