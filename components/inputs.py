# components/inputs.py
import streamlit as st
from utils.helpers import create_header_with_help

def show_input_sections():
    """Show all input sections for the experiment"""
    with st.expander("💡 Product Context (click to expand)", expanded=True):
        show_product_context()
    
    with st.expander("🎯 Metric Improvement Objective (click to expand)", expanded=True):
        show_metric_inputs()
    
    with st.expander("🧠 Generate Experiment Plan", expanded=True):
        show_generate_section()

def show_product_context():
    """Product context inputs"""
    create_header_with_help(
        "Product Context",
        "Provide the product context and business goal",
        icon="💡",
    )
    
    col_a, col_b = st.columns([2, 1])
    with col_a:
        product_type = st.selectbox(
            "Product Type",
            ["SaaS", "Consumer App", "E-commerce", "Marketplace", "Gaming", "Other"],
            index=0,
        )
        user_base_choice = st.selectbox(
            "User Base Size (DAU)",
            ["< 10K", "10K–100K", "100K–1M", "> 1M"],
            index=0,
        )
        metric_focus = st.selectbox(
            "Primary Metric Focus",
            ["Activation", "Retention", "Monetization", "Engagement", "Virality"],
            index=0,
        )
        product_notes = st.text_area(
            "Anything unique about your product or users? (optional)",
            placeholder="e.g. seasonality, power users, drop-off at pricing",
        )
        
    with col_b:
        strategic_goal = st.text_area(
            "High-Level Business Goal *",
            placeholder="e.g., Increase overall revenue from our premium tier",
        )
        user_persona = st.text_input(
            "Target User Persona (optional)",
            placeholder="e.g., First-time users from India, iOS users, power users",
        )
    
    return {
        "product_type": product_type,
        "user_base_choice": user_base_choice,
        "metric_focus": metric_focus,
        "product_notes": product_notes,
        "strategic_goal": strategic_goal,
        "user_persona": user_persona
    }
