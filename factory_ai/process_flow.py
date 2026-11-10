"""Documented process-flow scenario -> from-to flow matrix.

Flow unit: unit loads (totes / pallets) moved per shift.  The numbers below are an
ASSUMED scenario (3 product families with different routings) and must be replaced by
real process-planning data if a case study provides it.
"""
from collections import defaultdict

PRODUCTS = {
    "A (high volume)": dict(volume=120, route=["Receiving", "Cutting", "Drilling", "Assembly",
                                               "Inspection", "Packaging", "Storage", "Dispatch"]),
    "B (no drilling)": dict(volume=60, route=["Receiving", "Cutting", "Assembly",
                                              "Inspection", "Packaging", "Storage", "Dispatch"]),
    "C (direct ship)": dict(volume=40, route=["Receiving", "Cutting", "Drilling", "Assembly",
                                              "Inspection", "Packaging", "Dispatch"]),
}
# Fraction of the total volume that fails inspection and returns to assembly (rework loop).
REWORK = {("Inspection", "Assembly"): 0.08}


def build_flows(products=None, rework=None, available=None):
    """Return {(from, to): unit loads/shift}.

    `available` (optional set of node names): routings are filtered to those nodes, so a
    missing/undetected machine is bridged rather than breaking the model.
    """
    products = PRODUCTS if products is None else products
    rework = REWORK if rework is None else rework
    flows = defaultdict(float)
    total = 0.0
    for p in products.values():
        route = [n for n in p["route"] if available is None or n in available]
        total += p["volume"]
        for a, b in zip(route[:-1], route[1:]):
            flows[(a, b)] += p["volume"]
    for (a, b), frac in rework.items():
        if available is None or (a in available and b in available):
            flows[(a, b)] += frac * total
    return {k: round(v, 2) for k, v in flows.items()}


def flow_table(flows):
    import pandas as pd
    return (pd.DataFrame([{"From": a, "To": b, "Unit_loads_per_shift": v}
                          for (a, b), v in sorted(flows.items(), key=lambda kv: -kv[1])]))
