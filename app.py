# app.py
import streamlit as st
from utils.helpers import initialize_session_state
from components.inputs import show_input_sections
from components.display import show_results_section
from components.export import show_export_section

def main():
    st.set_page_config(page_title="A/B Test Architect", layout="wide")
    initialize_session_state()
    
    # Load CSS and setup UI
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    st.title("💡 A/B Test Architect — AI-assisted experiment PRD generator")
    st.markdown("Create experiment PRDs, hypotheses, stats, and sample-size guidance — faster and with guardrails.")
    
    # App sections
    show_input_sections()
    show_results_section()
    show_export_section()

if __name__ == "__main__":
    main()
