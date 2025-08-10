# utils/helpers.py
import streamlit as st
from typing import Any, Dict

def initialize_session_state():
    """Initialize all required session state variables"""
    defaults = {
        "output": None,
        "ai_parsed": None,
        "calc_locked": False,
        "locked_stats": {},
        "selected_index": None,
        "hypothesis_confirmed": False,
        "last_llm_hash": None,
        "calculate_now": False,
        "metrics_table": []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def create_header_with_help(header_text: str, help_text: str, icon: str = "🔗"):
    """Create a styled header with help tooltip"""
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="font-size: 1.4rem;">{icon}</div>
                <div class="section-title" style="margin-bottom: 0;">{header_text}</div>
            </div>
            <span style="font-size: 0.95rem; color: #666; cursor: help; float: right;" 
                  title="{help_text}">❓</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

def sanitize_text(text: Any) -> str:
    """Clean and standardize text input"""
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""
    text = text.replace("\r", " ").replace("\t", " ")
    return re.sub(r"[ \f\v]+", " ", text).strip()
