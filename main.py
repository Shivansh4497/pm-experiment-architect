# main.py — Ultimate PRD Version with Dark Mode & 3D Visualizations
import streamlit as st
import json
import re
import os
import pandas as pd
from typing import Any, Dict, Optional, Tuple, List
from prompt_engine import generate_experiment_plan
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize
import numpy as np
import hashlib
from datetime import datetime
from io import BytesIO
import ast

# PDF Export Setup
REPORTLAB_AVAILABLE = False
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# NEW: Full-width layout config
st.set_page_config(
    page_title="A/B Test Architect", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# NEW: Inject full-width CSS
st.markdown("""
    <style>
        .appview-container .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 100%;
        }
        div[data-testid="stExpander"] div[role="button"] p {
            font-size: 1.2rem;
            font-weight: bold;
        }
    </style>
    <script>
        // Sync dark mode with Streamlit theme
        const syncTheme = () => {
            const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            document.documentElement.classList.toggle('dark-mode', isDark);
        };
        window.matchMedia('(prefers-color-scheme: dark)').addListener(syncTheme);
        syncTheme();
    </script>
""", unsafe_allow_html=True)

def create_header_with_help(header_text: str, help_text: str, icon: str = "🔗"):
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="font-size: 1.4rem;">{icon}</div>
                <div class="section-title" style="margin-bottom: 0;">{header_text}</div>
            </div>
            <span style="font-size: 0.95rem; color: #666; cursor: help; float: right;" title="{help_text}">❓</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

def sanitize_text(text: Any) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""
    text = text.replace("\r"," ").replace("\t", " ")
    text = re.sub(r"[ \f\v]+", " ", text)
    return text.strip()

def html_sanitize(text: Any) -> str:
    """Escapes special HTML characters from a string."""
    if text is None: return ""
    text = str(text)
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    text = text.replace("'", "&apos;")
    return text

def generate_problem_statement(plan: Dict, current: float, target: float, unit: str) -> str:
    """Auto-inserts target metric into problem statement"""
    base = plan.get("problem_statement", "")
    if not base.strip():
        return base
    
    metric_str = f" (current: {format_value_with_unit(current, unit)} → target: {format_value_with_unit(target, unit)})"
    
    # Insert after first sentence if not already present
    if metric_str not in base:
        sentences = base.split('.')
        if len(sentences) > 1:
            sentences[0] = sentences[0].strip() + metric_str + "."
            return '.'.join(sentences)
        return base + metric_str
    return base

def safe_display(text: Any, method=st.info):
    method(sanitize_text(text))

def _extract_json_first_braces(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    tag_match = re.search(r"<json>([\s\S]+?)</json>", text, re.IGNORECASE)
    if tag_match:
        return tag_match.group(1).strip()
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None

def _safe_single_to_double_quotes(s: str) -> str:
    s = re.sub(r"(?<=[:\{\[,]\s*)'([^']*?)'(?=\s*[,}\]])", r'"\1"', s)
    s = re.sub(r"'([A-Za-z0-9_ \-]+?)'\s*:", r'"\1":', s)
    return s

def extract_json(text: Any) -> Optional[Dict]:
    if text is None:
        st.error("No output returned from LLM.")
        return None

    if isinstance(text, dict):
        return text
    if isinstance(text, list):
        if all(isinstance(i, dict) for i in text):
            return {"items": text}
        st.error("LLM returned a JSON list when an object was expected.")
        return None

    try:
        raw = str(text)
    except Exception as e:
        st.error(f"Unexpected LLM output type: {e}")
        return None

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and all(isinstance(i, dict) for i in parsed):
            return {"items": parsed}
        st.error("Parsed JSON was not an object; expected an object.")
        return None
    except Exception:
        pass

    try:
        parsed_ast = ast.literal_eval(raw)
        if isinstance(parsed_ast, dict):
            return parsed_ast
        if isinstance(parsed_ast, list) and all(isinstance(i, dict) for i in parsed_ast):
            return {"items": parsed_ast}
    except Exception:
        pass

    candidate = _extract_json_first_braces(raw)
    if candidate:
        candidate_clean = candidate
        candidate_clean = re.sub(r"^```(?:json)?\s*", "", candidate_clean).strip()
        candidate_clean = re.sub(r"\s*```$", "", candidate_clean).strip()
        candidate_clean = re.sub(r',\s*,', ',', candidate_clean)
        candidate_clean = re.sub(r',\s*\}', '}', candidate_clean)
        candidate_clean = re.sub(r',\s*\]', ']', candidate_clean)
        try:
            parsed = json.loads(candidate_clean)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and all(isinstance(i, dict) for i in parsed):
                return {"items": parsed}
            st.error("Extracted JSON parsed but was not an object.")
            st.code(candidate_clean[:2000] + ("..." if len(candidate_clean) > 2000 else ""))
            return None
        except Exception:
            try:
                parsed_ast = ast.literal_eval(candidate_clean)
                if isinstance(parsed_ast, dict):
                    return parsed_ast
                if isinstance(parsed_ast, list) and all(isinstance(i, dict) for i in parsed_ast):
                    return {"items": parsed_ast}
                else:
                    raise ValueError("Extracted JSON parsed as a list but was expected to be an object.")
            except Exception:
                try:
                    converted = _safe_single_to_double_quotes(candidate_clean)
                    parsed = json.loads(converted)
                    if isinstance(parsed, dict):
                        return parsed
                    if isinstance(parsed, list) and all(isinstance(i, dict) for i in parsed):
                        return {"items": parsed}
                    else:
                        raise ValueError("Extracted JSON with single quotes parsed but was not an object.")
                except Exception:
                    st.error("Could not parse extracted JSON block. See snippet below.")
                    st.code(candidate_clean[:3000] + ("..." if len(candidate_clean) > 3000 else ""))
                    return None

    try:
        converted_full = _safe_single_to_double_quotes(raw)
        parsed = json.loads(converted_full)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and all(isinstance(i, dict) for i in parsed):
            return {"items": parsed}
    except Exception:
        pass

    st.error("LLM output could not be parsed as JSON. Please inspect or edit the raw output below.")
    try:
        st.code(raw[:3000] + ("..." if len(raw) > 3000 else ""))
    except Exception:
        st.write("LLM output could not be displayed.")
    return None
def post_process_llm_text(text: Any, unit: str) -> str:
    if text is None:
        return ""
    s = sanitize_text(text)
    if unit == "%":
        s = s.replace("%%", "%")
        s = re.sub(r"\s+%", "%", s)
    return s

def format_value_with_unit(value: Any, unit: str) -> str:
    try:
        if isinstance(value, (int, float)):
            if float(value).is_integer():
                v_str = str(int(value))
            else:
                v_str = str(round(float(value), 4)).rstrip("0").rstrip(".")
        else:
            v_str = str(value)
    except Exception:
        v_str = str(value)
    units_with_space = ["USD", "count", "minutes", "hours", "days", "INR"]
    if unit in units_with_space:
        return f"{v_str} {unit}"
    else:
        return f"{v_str}{unit}"
        
def _parse_value_from_text(text: str, default_unit: str = '%') -> Tuple[Optional[float], str]:
    """Extracts a numeric value and unit from a string with validation."""
    text = sanitize_text(text)
    match = re.match(r"([\d\.]+)\s*(\w+|%)?", text)
    if not match:
        try:
            return float(text), default_unit
        except ValueError:
            return None, default_unit
    
    value = float(match.group(1))
    unit = match.group(2) if match.group(2) else default_unit
    
    # Add validation for unit mismatch
    if default_unit != '%' and unit != default_unit:  # '%' is special case (common in LLM outputs)
        st.warning(f"Unit mismatch: Using '{unit}' from input instead of selected '{default_unit}'")
    
    return value, unit

def calculate_sample_size(baseline, mde, alpha, power, num_variants, metric_type, std_dev=None) -> Tuple[Optional[int], Optional[int]]:
    try:
        if baseline is None or mde is None:
            return None, None

        mde_relative = float(mde) / 100.0
        if metric_type == "Conversion Rate":
            try:
                baseline_prop = float(baseline) / 100.0
            except Exception:
                return None, None
            if baseline_prop <= 0:
                return None, None
            expected_prop = baseline_prop * (1 + mde_relative)
            expected_prop = min(expected_prop, 0.999)
            effect_size = proportion_effectsize(baseline_prop, expected_prop)
            if effect_size == 0:
                return None, None
            analysis = NormalIndPower()
            sample_size_per_variant = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative="two-sided")
        elif metric_type == "Numeric Value":
            if std_dev is None or float(std_dev) == 0:
                return None, None
            mde_absolute = float(baseline) * mde_relative
            effect_size = mde_absolute / float(std_dev)
            if effect_size == 0:
                return None, None
            analysis = TTestIndPower()
            sample_size_per_variant = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative="two-sided")
        else:
            return None, None

        if sample_size_per_variant is None or sample_size_per_variant <= 0 or not np.isfinite(sample_size_per_variant):
            return None, None

        total = sample_size_per_variant * num_variants
        return int(np.ceil(sample_size_per_variant)), int(np.ceil(total))
    except Exception:
        return None, None

def generate_pdf_bytes_from_prd_dict(prd: Dict, title: str = "Experiment PRD") -> Optional[bytes]:
    if not REPORTLAB_AVAILABLE:
        return None
    
    def pdf_sanitize(text: Any) -> str:
        if text is None: return ""
        text = str(text)
        # Escape XML special chars
        text = (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&apos;"))
        return text
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Sanitize all content
    sanitized_prd = {
        k: pdf_sanitize(v) if isinstance(v, str) else v
        for k, v in prd.items()
    }
    
    styles.add(ParagraphStyle(name="PRDTitle", fontSize=20, leading=24, spaceAfter=12, alignment=1))
    styles.add(ParagraphStyle(name="SectionHeading", fontSize=13, leading=16, spaceBefore=12, spaceAfter=6, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name="BodyTextCustom", fontSize=10.5, leading=14))
    styles.add(ParagraphStyle(name="BulletText", fontSize=10.5, leading=14, leftIndent=12, bulletIndent=6))
    story: List[Any] = []
    story.append(Paragraph(title, styles["PRDTitle"]))
    def add_paragraph(heading: str, content: Any):
        story.append(Paragraph(heading, styles["SectionHeading"]))
        if content is None:
            return
        if isinstance(content, str):
            story.append(Paragraph(pdf_sanitize(content).replace('\n', '<br/>'), styles["BodyTextCustom"]))
        elif isinstance(content, dict):
            for k, v in content.items():
                if v is None:
                    continue
                story.append(Paragraph(f"<b>{pdf_sanitize(str(k))}:</b> {pdf_sanitize(str(v))}", styles["BodyTextCustom"]))
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    story.append(Paragraph(json.dumps(item, ensure_ascii=False), styles["BulletText"], bulletText="•"))
                else:
                    story.append(Paragraph(pdf_sanitize(str(item)), styles["BulletText"], bulletText="•"))
        else:
            story.append(Paragraph(pdf_sanitize(str(content)), styles["BodyTextCustom"]))
        story.append(Spacer(1, 6))
    doc.build(story)
    buffer.seek(0)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# NEW: Ultimate PRD Generator with Dark Mode and 3D Visualizations
def generate_ultimate_prd(prd_dict: Dict, context: Dict) -> str:
    """Generate a stunning PRD with dark mode and 3D visualizations"""
    # Dynamic Color System
    COLORS = {
        'light': {
            'primary': "#3b82f6",
            'secondary': "#8b5cf6",
            'bg': "#ffffff",
            'text': "#1e293b",
            'card': "#f8fafc"
        },
        'dark': {
            'primary': "#60a5fa",
            'secondary': "#a78bfa",
            'bg': "#0f172a",
            'text': "#f1f5f9",
            'card': "#1e293b"
        }
    }

    # Dark Mode Toggle Component - using triple quotes for multi-line string
    dark_mode_toggle = """
    <div style="position: absolute; top: 20px; right: 20px; z-index: 1000;">
        <label class="switch">
            <input type="checkbox" id="darkModeToggle" onclick="toggleDarkMode()">
            <span class="slider round"></span>
        </label>
        <span style="margin-left: 8px; font-size: 0.8rem;">Dark Mode</span>
    </div>

    <style>
        .switch {
            position: relative;
            display: inline-block;
            width: 48px;
            height: 24px;
        }
        .slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #ccc;
            transition: .4s;
            border-radius: 24px;
        }
        .slider:before {
            position: absolute;
            content: "";
            height: 16px;
            width: 16px;
            left: 4px;
            bottom: 4px;
            background-color: white;
            transition: .4s;
            border-radius: 50%;
        }
        input:checked + .slider {
            background: linear-gradient(45deg, #3b82f6, #8b5cf6);
        }
        input:checked + .slider:before {
            transform: translateX(24px);
        }
    </style>

    <script>
        function toggleDarkMode() {
            const doc = document.documentElement;
            doc.classList.toggle('dark-mode');
            
            // Save preference
            const isDark = doc.classList.contains('dark-mode');
            localStorage.setItem('darkMode', isDark);
        }
        
        // Initialize
        if (localStorage.getItem('darkMode') === 'true') {
            document.documentElement.classList.add('dark-mode');
            document.getElementById('darkModeToggle').checked = true;
        }
    </script>
    """

    # 3D Metrics Visualization - using proper string escaping
    metrics_3d = ""
    if prd_dict.get('metrics'):
        metrics_data = []
        for m in prd_dict['metrics']:
            if isinstance(m, dict):
                try:
                    value_match = re.search(r'\d+', m.get('formula', '0'))
                    value = float(value_match.group()) if value_match else 0
                except ValueError:
                    value = 0
                importance = ['Low', 'Medium', 'High'].index(m.get('importance', 'Medium')) + 1
                metrics_data.append({
                    'name': m.get('name', ''),
                    'value': value,
                    'importance': importance
                })
        
        metrics_3d = f"""
        <div id="metrics-3d" style="height: 300px; margin: 2rem 0;">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script>
                const metricsData = {json.dumps(metrics_data)};
                
                const layout = {{
                    scene: {{
                        xaxis: {{ title: 'Metric' }},
                        yaxis: {{ title: 'Value' }},
                        zaxis: {{ title: 'Importance' }},
                        camera: {{ 
                            eye: {{ x: 1.5, y: 1.5, z: 0.8 }} 
                        }}
                    }},
                    margin: {{ t: 0, b: 0 }}
                }};
                
                Plotly.newPlot('metrics-3d', [{{
                    type: 'scatter3d',
                    mode: 'markers',
                    x: metricsData.map(m => m.name),
                    y: metricsData.map(m => m.value),
                    z: metricsData.map(m => m.importance),
                    marker: {{
                        size: 12,
                        color: metricsData.map(m => 
                            m.importance === 3 ? '#ef4444' : 
                            m.importance === 2 ? '#f59e0b' : '#10b981'),
                        line: {{ width: 0 }}
                    }},
                    hoverinfo: 'x+y+z+text',
                    hovertext: metricsData.map(m => 
                        `Target: ${m.value * 1.2} (${m.importance === 3 ? 'High' : 
                        m.importance === 2 ? 'Medium' : 'Low'} priority)`)
                }}], layout);
            </script>
        </div>
        """

    # Dynamic CSS with proper escaping
    dynamic_css = f"""
    <style>
        :root {{
            --primary: {COLORS['light']['primary']};
            --secondary: {COLORS['light']['secondary']};
            --bg: {COLORS['light']['bg']};
            --text: {COLORS['light']['text']};
            --card: {COLORS['light']['card']};
        }}
        
        .dark-mode {{
            --primary: {COLORS['dark']['primary']};
            --secondary: {COLORS['dark']['secondary']};
            --bg: {COLORS['dark']['bg']};
            --text: {COLORS['dark']['text']};
            --card: {COLORS['dark']['card']};
        }}
        
        body {{
            background: var(--bg);
            color: var(--text);
            transition: all 0.3s ease;
        }}
        
        .prd-card {{
            background: var(--card);
            color: var(--text);
            border-radius: 16px;
            padding: 2.5rem;
            margin: 2rem 0;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }}
        
        .prd-section-content {{
            background: var(--bg);
            padding: 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
        }}
        
        .hypothesis-title::before {{
            background: var(--secondary);
        }}
    </style>
    """

    # Progress bars with proper string handling
    progress_bars = ""
    if prd_dict.get('success_criteria'):
        progress_items = []
        for name, value in prd_dict['success_criteria'].items():
            progress_items.append(f"""
            <div style="margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span>{name}</span>
                    <span>{value}{'%' if name in ['Confidence', 'MDE'] else ''}</span>
                </div>
                <div style="height: 8px; background: #e2e8f0; border-radius: 4px; overflow: hidden;">
                    <div style="height: 100%; width: {value if name in ['Confidence', 'MDE'] else 100}%; 
                        background: {'var(--primary)' if name == 'Confidence' else '#10b981' if name == 'MDE' else 'var(--secondary)'}; 
                        border-radius: 4px; animation: grow 1s ease-out;">
                    </div>
                </div>
            </div>
            """)
        
        progress_bars = f"""
        <div class="progress-container" style="margin: 1.5rem 0;">
            <h3 style="margin-bottom: 1rem;">Success Criteria</h3>
            {''.join(progress_items)}
            <style>
                @keyframes grow {{
                    from {{ width: 0% }}
                    to {{ width: 100% }}
                }}
            </style>
        </div>
        """

    # Final assembly with proper string joining
    return f"""{dark_mode_toggle}
{dynamic_css}
<div class="prd-card">
    <div class="prd-header">
        <div class="logo-wrapper">A/B</div>
        <div class="header-text">
            <h1>Experiment PRD</h1>
            <p>{prd_dict.get('goal', '')}</p>
        </div>
    </div>

    <div class="prd-section">
        <div class="prd-section-title">
            <h2>Problem Statement</h2>
        </div>
        <div class="prd-section-content">
            {prd_dict.get('problem_statement', 'No problem statement provided.')}
        </div>
    </div>

    <div class="prd-section">
        <div class="prd-section-title">
            <h2>Hypothesis</h2>
        </div>
        <div class="prd-section-content">
            {prd_dict.get('hypotheses', [{}])[0].get('hypothesis', 'No hypothesis selected')}
        </div>
    </div>

    {metrics_3d}

    {progress_bars}
</div>
"""
    # Main Streamlit App UI
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800&display=swap" rel="stylesheet">
    <style>
        .appview-container .main .block-container {
            max-width: 100%;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        .stButton>button {
            width: 100%;
        }
        .stExpander {
            background: transparent;
        }
        .st-eb {
            background-color: transparent !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💡 A/B Test Architect — AI-assisted experiment PRD generator")
st.markdown("Create experiment PRDs, hypotheses, stats, and sample-size guidance — faster and with guardrails.")

# Initialize session state
if "output" not in st.session_state:
    st.session_state.output = None
if "ai_parsed" not in st.session_state:
    st.session_state.ai_parsed = None
if "calc_locked" not in st.session_state:
    st.session_state.calc_locked = False
if "locked_stats" not in st.session_state:
    st.session_state.locked_stats = {}
if "selected_index" not in st.session_state:
    st.session_state.selected_index = None
if "hypothesis_confirmed" not in st.session_state:
    st.session_state.hypothesis_confirmed = False
if "last_llm_hash" not in st.session_state:
    st.session_state.last_llm_hash = None
if "calculate_now" not in st.session_state:
    st.session_state.calculate_now = False
if "metrics_table" not in st.session_state:
    st.session_state.metrics_table = None
if "raw_llm_edit" not in st.session_state:
    st.session_state.raw_llm_edit = ""
if "context" not in st.session_state:
    st.session_state.context = {}

# Product Context Section
with st.expander("💡 Product Context (click to expand)", expanded=True):
    create_header_with_help(
        "Product Context",
        "Provide the product context and business goal so the AI can produce a focused experiment plan.",
        icon="💡",
    )
    col_a, col_b = st.columns([2, 1])
    with col_a:
        product_type = st.selectbox(
            "Product Type",
            ["SaaS", "Consumer App", "E-commerce", "Marketplace", "Gaming", "Other"],
            index=0,
            help="What kind of product are you testing?",
        )
        user_base_choice = st.selectbox(
            "User Base Size (DAU)",
            ["< 10K", "10K–100K", "100K–1M", "> 1M"],
            index=0,
            help="Average daily active users for the product.",
        )
        metric_focus = st.selectbox(
            "Primary Metric Focus",
            ["Activation", "Retention", "Monetization", "Engagement", "Virality"],
            index=0,
            help="The general category of metrics you're trying to move.",
        )
        product_notes = st.text_area(
            "Anything unique about your product or users? (optional)",
            placeholder="e.g. seasonality, power users, drop-off at pricing",
            help="Optional context to inform better suggestions.",
        )
    with col_b:
        strategic_goal = st.text_area(
            "High-Level Business Goal *",
            placeholder="e.g., Increase overall revenue from our premium tier",
            help="This is the broader business goal the experiment supports.",
        )
        user_persona = st.text_input(
            "Target User Persona (optional)",
            placeholder="e.g., First-time users from India, iOS users, power users",
            help="Focus the plan on a specific user segment.",
        )

# Metric Improvement Section
with st.expander("🎯 Metric Improvement Objective (click to expand)", expanded=True):
    create_header_with_help(
        "Metric Improvement Objective",
        "Provide the exact metric and current vs target values. Use the proper units.",
        icon="🎯",
    )
    col_m1, col_m2 = st.columns([2, 2])
    with col_m1:
        exact_metric = st.text_input(
            "Metric to Improve * (e.g. Activation Rate, ARPU, DAU/MAU)",
            help="Be specific — name the metric you want to shift.",
        )
    with col_m2:
        metric_type = st.radio("Metric Type", ["Conversion Rate", "Numeric Value"], horizontal=True)
    
    col_unit, col_values = st.columns([1, 2])
    with col_unit:
        metric_unit = st.selectbox(
            "Metric Unit", ["%", "USD", "INR", "minutes", "count", "other"], index=0, help="Choose the unit for clarity."
        )
    with col_values:
        current_value_raw = st.text_input("Current Metric Value *", placeholder="e.g., 55.0 or 55.0 INR")
        target_value_raw = st.text_input("Target Metric Value *", placeholder="e.g., 60.0 or 60.0 INR")

        current_value, current_unit = _parse_value_from_text(current_value_raw, metric_unit)
        target_value, target_unit = _parse_value_from_text(target_value_raw, metric_unit)

        if current_value is None and current_value_raw:
            st.error("Invalid format for Current Metric Value. Please enter a number.")
        if target_value is None and target_value_raw:
            st.error("Invalid format for Target Metric Value. Please enter a number.")

        std_dev = None
        if metric_type == "Numeric Value":
            std_dev_raw = st.text_input(
                "Standard Deviation of Metric * (required for numeric metrics)",
                placeholder="e.g., 10.5",
                help="The standard deviation is required for numeric metrics to compute sample sizes.",
            )
            std_dev, _ = _parse_value_from_text(std_dev_raw)
            if std_dev is None and std_dev_raw:
                st.error("Invalid format for Standard Deviation. Please enter a number.")

    metric_inputs_valid = True
    if current_value == target_value and current_value is not None:
        st.warning("The target metric must be different from the current metric to measure change. Please adjust one or the other.")
        metric_inputs_valid = False
    
    if metric_type == "Conversion Rate" and metric_unit != "%":
        st.warning("For 'Conversion Rate' metric type, the unit should be '%'.")
        metric_inputs_valid = False

# Generate Plan Section
with st.expander("🧠 Generate Experiment Plan", expanded=True):
    create_header_with_help("Generate Experiment Plan", "When ready, click Generate to call the LLM and create a plan.", icon="🧠")
    sanitized_metric_name = sanitize_text(exact_metric)
    
    try:
        if current_value is not None and current_value != 0:
            expected_lift_val = round(((target_value - current_value) / current_value) * 100, 2)
            mde_default = round(abs((target_value - current_value) / current_value) * 100, 2)
        else:
            expected_lift_val = 0.0
            mde_default = 5.0
    except Exception:
        expected_lift_val = 0.0
        mde_default = 5.0

    formatted_current = format_value_with_unit(current_value, metric_unit) if sanitized_metric_name and current_value is not None else ""
    formatted_target = format_value_with_unit(target_value, metric_unit) if sanitized_metric_name and target_value is not None else ""
    goal_with_units = f"I want to improve {sanitized_metric_name} from {formatted_current} to {formatted_target}." if sanitized_metric_name else ""

    required_ok = all(
        [
            product_type,
            user_base_choice,
            metric_focus,
            sanitized_metric_name,
            metric_inputs_valid,
            strategic_goal,
            current_value is not None,
            target_value is not None
        ]
    )
    generate_btn = st.button("Generate Plan", disabled=not required_ok)
    
    if generate_btn:
        st.session_state.selected_index = None
        st.session_state.hypothesis_confirmed = False
        st.session_state.calc_locked = False
        st.session_state.locked_stats = {}
        st.session_state.ai_parsed = None # Clear old plan

        context = {
            "type": product_type,
            "users": user_base_choice,
            "metric": metric_focus,
            "notes": product_notes,
            "exact_metric": sanitized_metric_name,
            "current_value": current_value,
            "target_value": target_value,
            "expected_lift": expected_lift_val,
            "minimum_detectable_effect": mde_default,
            "metric_unit": metric_unit,
            "strategic_goal": strategic_goal,
            "user_persona": user_persona,
            "metric_type": metric_type,
            "std_dev": std_dev,
        }

        with st.spinner("Generating your plan..."):
            try:
                raw_llm = generate_experiment_plan(goal_with_units, context)
                st.session_state.output = raw_llm if raw_llm is not None else ""
                st.session_state.raw_llm_edit = st.session_state.output
                parsed = extract_json(raw_llm)
                st.session_state.ai_parsed = parsed
                try:
                    h = hashlib.sha256((str(raw_llm) or "").encode("utf-8")).hexdigest()
                    st.session_state.last_llm_hash = h
                except Exception:
                    st.session_state.last_llm_hash = None
                if parsed:
                    st.success("Plan generated successfully — review and edit below.")
                else:
                    st.warning("Plan generated but parsing failed — edit the raw output to correct JSON or try regenerate.")
            except Exception as e:
                st.error(f"LLM generation failed: {e}")
                st.session_state.output = ""
                st.session_state.ai_parsed = None

# Results Display
if st.session_state.get("ai_parsed"):
    plan = st.session_state.ai_parsed
    unit = st.session_state.context.get("metric_unit", metric_unit)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Generate the ultimate PRD
    prd_dict = {
        "goal": goal_with_units,
        "problem_statement": st.session_state.get("editable_problem", generate_problem_statement(plan, current_value, target_value, unit)),
        "hypotheses": plan.get("hypotheses", []),
        "metrics": st.session_state.get("metrics_table", plan.get("metrics", [])),
        "segments": plan.get("segments", []),
        "success_criteria": st.session_state.get("locked_stats", plan.get("success_criteria", {})),
        "risks_and_assumptions": plan.get("risks_and_assumptions", []),
        "next_steps": plan.get("next_steps", [])
    }
    
    # Display the ultimate PRD
    st.markdown(generate_ultimate_prd(prd_dict, st.session_state.context), unsafe_allow_html=True)

    # Download buttons
    col_dl1, col_dl2, col_dl3, col_dl4 = st.columns([1,1,1,1])
    with col_dl1:
        st.download_button(
            "📄 Download PRD (.txt)",
            json.dumps(prd_dict, indent=2),
            file_name="experiment_prd.txt"
        )
    with col_dl2:
        st.download_button(
            "📥 Download Plan (.json)",
            json.dumps(prd_dict, indent=2),
            file_name="experiment_plan.json"
        )
    with col_dl3:
        html_blob = f"""
        <!doctype html>
        <html>
        <head><meta charset='utf-8'></head>
        <body>
            {generate_ultimate_prd(prd_dict, st.session_state.context)}
        </body>
        </html>
        """
        st.download_button(
            "🌐 Download PRD (.html)",
            html_blob,
            file_name="experiment_prd.html"
        )
    with col_dl4:
        if REPORTLAB_AVAILABLE:
            pdf_bytes = generate_pdf_bytes_from_prd_dict(prd_dict)
            if pdf_bytes:
                st.download_button(
                    "📁 Download PRD (.pdf)",
                    pdf_bytes,
                    file_name="experiment_prd.pdf",
                    mime="application/pdf"
                )
            else:
                st.warning("PDF generation failed")
        else:
            st.info("PDF export requires reportlab")

# Debug Section
st.markdown("<hr>", unsafe_allow_html=True)
with st.expander("⚙️ Debug & Trace"):
    st.write("Last LLM hash:", st.session_state.get("last_llm_hash"))
    st.write("AI parsed present:", bool(st.session_state.get("ai_parsed")))
    st.write("Raw output length:", len(st.session_state.get("output") or ""))
    if st.button("Clear session state"):
        keys_to_clear = [
            "output", "ai_parsed", "calculated_sample_size_per_variant",
            "calculated_total_sample_size", "calculated_duration_days",
            "locked_stats", "calc_locked", "selected_index",
            "hypothesis_confirmed", "last_llm_hash", "context",
            "metrics_table", "editable_problem", "editable_hypothesis",
            "editable_rationale", "editable_example", "editable_segments",
            "editable_risks", "editable_next_steps", "raw_llm_edit"
        ]
        for k in keys_to_clear:
            if k in st.session_state:
                del st.session_state[k]
        st.success("Session cleared.")
