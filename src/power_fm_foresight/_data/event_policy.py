"""Original event reporting roles."""
from fractions import Fraction

def prevalence_shift(calibration, test, threshold):
    a=Fraction(calibration["positive"],calibration["total"])
    b=Fraction(test["positive"],test["total"])
    if min(a,b)==0:
        return None, bool(max(a,b)>0)
    ratio=max(a,b)/min(a,b)
    return float(ratio),ratio>Fraction(str(threshold))



def reporting_policy(calibration, test, resolution, settings):
    ratio, shifted = prevalence_shift(calibration, test, settings["prevalence_shift_ratio"])
    low, high = settings.get("daily_calibration_prevalence_range", [0, 1])
    rate = Fraction(calibration["positive"], calibration["total"])
    eligible = resolution == "hourly" or Fraction(str(low)) <= rate <= Fraction(str(high))
    return {"ratio": ratio, "exceeds_1p5": shifted,
        "resolution_role": "primary" if resolution == "hourly" else "secondary",
        "panel_eligible_by_prevalence": eligible,
        "panel_exclusion_reason": None if eligible else "Daily calibration prevalence outside 20%-80%",
        "raw_frequency_role": "report_side_by_side",
        "posthoc_map_role": "report_side_by_side_with_shift_caption" if shifted else "report_side_by_side",
        "posthoc_fit_partition": "risk_calibration", "prevalence_table_required": True,
        "shift_caption_required": shifted, "training_blocker": False}
