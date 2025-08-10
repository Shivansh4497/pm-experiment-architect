# components/display.py
import streamlit as st
import pandas as pd
from utils.helpers import create_header_with_help, sanitize_text
from utils.calculations import calculate_sample_size

def show_results_section():
    """Display the generated results and calculator"""
    if st.session_state.get("ai_parsed") is None and st.session_state.get("output"):
        show_raw_llm_output()
    
    if st.session_state.get("ai_parsed"):
        show_calculator()
        show_parsed_results()

def show_raw_llm_output():
    """Display raw LLM output when parsing fails"""
    st.markdown("<div class='blue-section'>", unsafe_allow_html=True)
    create_header_with_help(
        "Raw LLM Output (fix JSON here)", 
        "When parsing fails you'll see the raw LLM output",
        icon="🛠️",
    )
    
    raw_edit = st.text_area(
        "Raw LLM output / edit here", 
        value=st.session_state.get("output", ""), 
        height=400, 
        key="raw_llm_edit"
    )
    
    if st.button("Parse JSON"):
        parsed_try = extract_json(st.session_state.get("raw_llm_edit", raw_edit))
        if parsed_try:
            st.session_state.ai_parsed = parsed_try
            st.success("Manual parse succeeded — plan is now usable.")
        else:
            st.error("Manual parse failed — edit the text and try again.")
    
    st.markdown("</div>", unsafe_allow_html=True)

def show_calculator():
    """Show the A/B test calculator"""
    st.markdown("<hr>", unsafe_allow_html=True)
    
    with st.expander("🔢 A/B Test Calculator: Fine-tune sample size", expanded=True):
        # Calculator implementation...
        pass
