# ================================================================
# src/scanner/scoring.py
# ----------------------------------------------------------------
# Unified conviction scoring engine.
#
# Combines three indicator modules into one score (0-100):
#   Volume indicators  (volume.py)  → max 35 pts
#   OBV indicators     (obv.py)     → max 40 pts
#   Price indicators   (price.py)   → max 25 pts
#
# Conviction Tiers:
#   HIGH    >= 70  → priority watchlist
#   MEDIUM  50-69  → secondary watchlist
#   LOW     30-49  → monitor only
#   SKIP    < 30   → ignore
#
# Also computes:
#   bonus_points   → extra pts for rare powerful combos
#   red_flags      → conditions that reduce conviction
#   final_score    → base + bonus - red_flags (capped 0-100)
# ================================================================

import pandas as pd
import numpy as np
from src.utils.logger import get_logger

log = get_logger(__name__)


# ================================================================
# SCORING WEIGHTS
# ================================================================

# Volume scoring (max 35 pts)
VOLUME_WEIGHTS = {
    "scenario_awakening"            : 12,
    "scenario_structural_rise"      : 8,
    "scenario_delivery_confirm"     : 8,
    "scenario_delivery_accel"       : 5,
    "scenario_delivery_progression" : 7,
    "scenario_volume_expansion"     : 5,
    "scenario_two_day_delivery_surge": 5,
    "scenario_contraction"          : 3,
    # Negative — climax means move already happened
    "scenario_climax"               : -10,
}
VOLUME_MAX = 35

# OBV scoring — uses obv.py internal score directly
# obv.py already scores 0-100 internally
# We scale it to max 40 pts
OBV_WEIGHT_SCALE = 0.40   # 40% of final score
OBV_MAX          = 40

# Price scoring (max 25 pts)
PRICE_WEIGHTS = {
    "scenario_consolidating"   : 8,
    "scenario_higher_lows"     : 7,
    "scenario_near_20ema"      : 6,
    "scenario_near_52w_high"   : 4,
    "scenario_multi_year_base" : 8,
    # Negative — below 200 EMA reduces confidence slightly
    # but not a hard block (OBV divergence can overcome this)
    "scenario_below_200ema"    : -2,
}
PRICE_MAX = 25

# Bonus combos (rare powerful combinations)
BONUS_COMBOS = [
    {
        "name"   : "absorption_plus_delivery",
        "desc"   : "Absorption event + high delivery",
        "pts"    : 8,
        "requires": [
            ("volume", "scenario_absorption"),
            ("volume", "scenario_delivery_confirm"),
        ]
    },
    {
        "name"   : "both_timeframe_obv_divergence",
        "desc"   : "Daily + weekly OBV both diverging",
        "pts"    : 6,
        "requires": [
            ("obv", "s12_both_timeframes"),
        ]
    },
    {
        "name"   : "coil_setup",
        "desc"   : "Tight range + OBV rising + delivery high",
        "pts"    : 7,
        "requires": [
            ("price",  "scenario_consolidating"),
            ("volume", "scenario_delivery_confirm"),
            ("obv",    "s6_price_flat_obv_rising"),
        ]
    },
    {
        "name"   : "shakeout_plus_divergence",
        "desc"   : "Shakeout detected + OBV diverging",
        "pts"    : 10,
        "requires": [
            ("obv", "s10_shakeout_detected"),
            ("obv", "s6_price_flat_obv_rising"),
        ]
    },
    {
        "name"   : "multi_year_base_obv",
        "desc"   : "Multi-year base + OBV awakening",
        "pts"    : 8,
        "requires": [
            ("price", "scenario_multi_year_base"),
            ("obv",   "s1_obv_new_high_price_not"),
        ]
    },
]

# Red flags — conditions that reduce final score
RED_FLAGS = [
    {
        "name"  : "climax_move",
        "desc"  : "Volume climax — move already happened",
        "pts"   : -15,
        "check" : lambda v, o, p: v.get("scenario_climax", False)
    },
    {
        "name"  : "price_extended",
        "desc"  : "Price >15% above 20 EMA — extended",
        "pts"   : -8,
        "check" : lambda v, o, p: (
            p.get("pct_from_ema20", 0) or 0
        ) > 15.0
    },
    {
        "name"  : "heavy_selling_week",
        "desc"  : "Price down >5% this week",
        "pts"   : -5,
        "check" : lambda v, o, p: (
            p.get("chg_1w", 0) or 0
        ) < -5.0
    },
]


# ================================================================
# MAIN SCORING FUNCTION
# ================================================================

def calculate_conviction_score(
    volume_indicators: dict,
    obv_indicators:    dict,
    price_indicators:  dict,
    symbol:            str = ""
) -> dict:
    """
    Calculates unified conviction score from all three indicators.

    Parameters
    ----------
    volume_indicators : dict
        Output from volume.calculate_volume_indicators()
    obv_indicators : dict
        Output from obv.calculate_obv_indicators()
    price_indicators : dict
        Output from price.calculate_price_indicators()
    symbol : str
        Stock symbol for logging

    Returns
    -------
    dict
        final_score      : int (0-100)
        conviction_tier  : str (HIGH/MEDIUM/LOW/SKIP)
        volume_score     : int (0-35)
        obv_score        : int (0-40)
        price_score      : int (0-25)
        bonus_pts        : int
        red_flag_pts     : int
        bonuses_fired    : list of bonus names
        red_flags_fired  : list of red flag names
        component_scores : dict per scenario
    """
    if not volume_indicators and not obv_indicators \
            and not price_indicators:
        return _empty_score()

    v = volume_indicators or {}
    o = obv_indicators    or {}
    p = price_indicators  or {}

    component_scores = {}

    # ================================================================
    # VOLUME SCORE (max 35 pts)
    # ================================================================

    volume_raw = 0
    for scenario, weight in VOLUME_WEIGHTS.items():
        fired = bool(v.get(scenario, False))
        pts = weight if fired else 0
        volume_raw += pts
        component_scores[f"vol_{scenario}"] = pts

    # ── Absorption tier bonus ──────────────────────────────────
    # Scored separately — tier S/A/B get different points
    # Tier S = super absorption (5x+ vol + 65%+ delivery) → 20 pts
    # Tier A = strong absorption (2x+ vol + 60%+ delivery) → 14 pts
    # Tier B = moderate absorption                         → 8 pts
    absorption_tier = v.get("absorption_tier")
    if absorption_tier == "S":
        absorption_pts = 20
    elif absorption_tier == "A":
        absorption_pts = 14
    elif absorption_tier == "B":
        absorption_pts = 8
    else:
        absorption_pts = 0

    volume_raw += absorption_pts
    component_scores["vol_absorption_tiered"] = absorption_pts

    # Cap at max
    volume_score = max(0, min(volume_raw, VOLUME_MAX))

    # ================================================================
    # OBV SCORE (max 40 pts)
    # ================================================================
    # obv.py already scored 0-100 internally
    # Scale down to 40 pts
    obv_internal = int(o.get("total_score", 0))
    obv_score    = min(
        int(obv_internal * OBV_WEIGHT_SCALE),
        OBV_MAX
    )
    component_scores["obv_internal_score"] = obv_internal
    component_scores["obv_scaled_score"]   = obv_score

    # ================================================================
    # PRICE SCORE (max 25 pts)
    # ================================================================
    price_raw = 0
    for scenario, weight in PRICE_WEIGHTS.items():
        fired = bool(p.get(scenario, False))
        pts   = weight if fired else 0
        price_raw += pts
        component_scores[f"price_{scenario}"] = pts

    price_score = max(0, min(price_raw, PRICE_MAX))

    # ================================================================
    # BASE SCORE
    # ================================================================
    base_score = volume_score + obv_score + price_score

    # ================================================================
    # BONUS COMBOS
    # ================================================================
    bonus_pts     = 0
    bonuses_fired = []

    for combo in BONUS_COMBOS:
        all_met = True
        for source, key in combo["requires"]:
            if source == "volume":
                val = v.get(key, False)
            elif source == "obv":
                val = o.get("flags", {}).get(key, False) \
                      if "flags" in o else o.get(key, False)
            elif source == "price":
                val = p.get(key, False)
            else:
                val = False

            if not bool(val):
                all_met = False
                break

        if all_met:
            bonus_pts += combo["pts"]
            bonuses_fired.append(combo["name"])
            component_scores[f"bonus_{combo['name']}"] = combo["pts"]
            log.debug(f"{symbol} bonus: {combo['name']} +{combo['pts']}")

    # ================================================================
    # RED FLAGS
    # ================================================================
    red_flag_pts  = 0
    red_flags_hit = []

    for flag in RED_FLAGS:
        try:
            triggered = flag["check"](v, o, p)
        except Exception:
            triggered = False

        if triggered:
            red_flag_pts  += flag["pts"]   # negative value
            red_flags_hit.append(flag["name"])
            component_scores[f"redflag_{flag['name']}"] = flag["pts"]
            log.debug(f"{symbol} red flag: {flag['name']} {flag['pts']}")

    # ================================================================
    # FINAL SCORE
    # ================================================================
    final_score = base_score + bonus_pts + red_flag_pts
    final_score = max(0, min(final_score, 100))

    # ================================================================
    # CONVICTION TIER
    # ================================================================
    if final_score >= 70:
        conviction_tier = "HIGH"
    elif final_score >= 50:
        conviction_tier = "MEDIUM"
    elif final_score >= 30:
        conviction_tier = "LOW"
    else:
        conviction_tier = "SKIP"

    log.debug(
        f"{symbol} score: {final_score} ({conviction_tier}) | "
        f"V:{volume_score} O:{obv_score} P:{price_score} "
        f"B:+{bonus_pts} R:{red_flag_pts}"
    )

    return {
        "final_score"      : final_score,
        "conviction_tier"  : conviction_tier,
        "volume_score"     : volume_score,
        "obv_score"        : obv_score,
        "price_score"      : price_score,
        "base_score"       : base_score,
        "bonus_pts"        : bonus_pts,
        "red_flag_pts"     : red_flag_pts,
        "bonuses_fired"    : bonuses_fired,
        "red_flags_fired"  : red_flags_hit,
        "component_scores" : component_scores,
        # OBV tier from obv.py
        "obv_tier"         : o.get("conviction_tier", "SKIP"),
        # Key signals for display
        "obv_divergence"   : bool(
            o.get("flags", {}).get("s6_price_flat_obv_rising", False)
            if "flags" in o
            else o.get("s6_price_flat_obv_rising", False)
        ),
        "weekly_divergence": bool(
            o.get("flags", {}).get("s11_weekly_divergence", False)
            if "flags" in o
            else o.get("s11_weekly_divergence", False)
        ),
        "absorption"       : bool(v.get("scenario_absorption", False)),
        "shakeout"         : bool(
            o.get("flags", {}).get("s10_shakeout_detected", False)
            if "flags" in o
            else o.get("s10_shakeout_detected", False)
        ),
    }


def _empty_score() -> dict:
    """Returns a zero score dict."""
    return {
        "final_score"     : 0,
        "conviction_tier" : "SKIP",
        "volume_score"    : 0,
        "obv_score"       : 0,
        "price_score"     : 0,
        "base_score"      : 0,
        "bonus_pts"       : 0,
        "red_flag_pts"    : 0,
        "bonuses_fired"   : [],
        "red_flags_fired" : [],
        "component_scores": {},
        "obv_tier"        : "SKIP",
        "obv_divergence"  : False,
        "weekly_divergence": False,
        "absorption"      : False,
        "shakeout"        : False,
    }