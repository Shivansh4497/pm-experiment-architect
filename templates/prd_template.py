# templates/prd_template.py
def generate_prd_html(prd_dict: dict) -> str:
    """Generate HTML template for PRD"""
    return f"""
    <div class="prd-card">
      <div class="prd-header">
        <div class="prd-logo">A/B</div>
        <div>
          <div class="prd-title">Experiment PRD</div>
          <div class="prd-subtitle">{prd_dict.get('goal', '')}</div>
        </div>
      </div>
      
      <!-- Sections for problem, hypotheses, metrics, etc... -->
    </div>
    """

def get_prd_css() -> str:
    """Return the CSS styles for PRD template"""
    return """
    .prd-card {
      background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
      border-radius: 14px;
      padding: 28px;
      box-shadow: 0 12px 36px rgba(13,60,120,0.08);
      border: 1px solid rgba(13,60,120,0.06);
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    /* More CSS styles... */
    """
