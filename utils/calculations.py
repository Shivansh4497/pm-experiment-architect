# utils/calculations.py
import numpy as np
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize

def calculate_sample_size(baseline, mde, alpha, power, num_variants, metric_type, std_dev=None):
    """Calculate required sample size for A/B test"""
    try:
        if baseline is None or mde is None:
            return None, None

        mde_relative = float(mde) / 100.0
        
        if metric_type == "Conversion Rate":
            try:
                baseline_prop = float(baseline) / 100.0
                if baseline_prop <= 0:
                    return None, None
                    
                expected_prop = baseline_prop * (1 + mde_relative)
                expected_prop = min(expected_prop, 0.999)
                effect_size = proportion_effectsize(baseline_prop, expected_prop)
                
                if effect_size == 0:
                    return None, None
                    
                analysis = NormalIndPower()
                sample_size_per_variant = analysis.solve_power(
                    effect_size=effect_size,
                    alpha=alpha,
                    power=power,
                    alternative="two-sided"
                )
                
            except Exception:
                return None, None
                
        elif metric_type == "Numeric Value":
            if std_dev is None or float(std_dev) == 0:
                return None, None
                
            mde_absolute = float(baseline) * mde_relative
            effect_size = mde_absolute / float(std_dev)
            
            if effect_size == 0:
                return None, None
                
            analysis = TTestIndPower()
            sample_size_per_variant = analysis.solve_power(
                effect_size=effect_size,
                alpha=alpha,
                power=power,
                alternative="two-sided"
            )
            
        else:
            return None, None

        if sample_size_per_variant is None or sample_size_per_variant <= 0 or not np.isfinite(sample_size_per_variant):
            return None, None

        total = sample_size_per_variant * num_variants
        return int(np.ceil(sample_size_per_variant)), int(np.ceil(total))
        
    except Exception:
        return None, None
