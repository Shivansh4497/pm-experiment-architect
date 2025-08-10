# components/export.py
import streamlit as st
import json
from datetime import datetime
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def show_export_section(prd_dict: dict):
    """Show export options for the PRD"""
    create_header_with_help("Export PRD", "Download your experiment plan", icon="📤")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.download_button(
            "📄 Download PRD (.txt)",
            json.dumps(prd_dict, indent=2),
            file_name="experiment_prd.txt"
        )
    
    with col2:
        st.download_button(
            "📥 Download Plan (.json)",
            json.dumps(prd_dict, indent=2),
            file_name="experiment_plan.json"
        )
    
    with col3:
        html_content = generate_html_prd(prd_dict)
        st.download_button(
            "🌐 Download PRD (.html)",
            html_content,
            file_name="experiment_prd.html"
        )
    
    with col4:
        if REPORTLAB_AVAILABLE:
            pdf_bytes = generate_pdf_prd(prd_dict)
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

def generate_pdf_prd(prd: dict) -> Optional[bytes]:
    """Generate PDF version of PRD"""
    if not REPORTLAB_AVAILABLE:
        return None
        
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    story = []
    story.append(Paragraph("Experiment PRD", styles["Title"]))
    
    # Add content to PDF story...
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
