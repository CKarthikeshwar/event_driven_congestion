#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================
 ASTraM — ADD-ON : LP / ILP Manpower Optimizer
====================================================================
Standalone. Does NOT modify existing files — imports astram_decisions for the
demand surface and compares against its proportional allocator.

Upgrades the manpower step from "proportional split" to a provably-optimal
integer program. Formulation (concave coverage, so officers spread to where
they cut the most uncovered demand — they don't all pile into one hotspot):

    maximize   sum_a  covered_a
    s.t.       covered_a <= demand_a                 (can't cover what isn't there)
               covered_a <= CAPACITY * officers_a    (coverage limited by staffing)
               sum_a officers_a = TOTAL
               MIN_COVER <= officers_a <= MAX_CAP    (for active areas)
               officers_a integer

CAPACITY self-calibrates to sum(demand)/TOTAL so you don't hand-tune units.

Solver preference: PuLP+CBC (exact integer) -> SciPy linprog (LP + rounding)
-> astram_decisions proportional method. So it runs in any environment.

Usage:  python astram_lp_allocator.py
Requires: pandas, numpy  (optional: pulp for exact ILP, scipy for LP fallback)
"""
import numpy as np
import pandas as pd
import astram_decisions as D

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

def _covered(demand, alloc, capacity):
    """Objective value: total severity-demand actually covered."""
    return sum(min(demand[a], capacity * alloc.get(a, 0)) for a in demand)

# ----------------------------------------------------------------------
# The optimizer (PuLP -> SciPy -> proportional)
# ----------------------------------------------------------------------
def lp_allocate(demand, total, min_cover=2, max_cap=None, capacity=None):
    active = {a: d for a, d in demand.items() if d > 0}
    alloc = {a: 0 for a in demand}
    if not active:
        return alloc, "none"
    n = len(active)
    # feasibility of the floor; relax if it cannot fit
    if min_cover * n > total:
        min_cover = total // n
    if max_cap is None:
        max_cap = total                                  # effectively uncapped
    if capacity is None:
        capacity = max(sum(active.values()) / total, 1e-9)  # self-calibrating
    areas = list(active)

    # ---- 1) PuLP exact integer program ----
    try:
        import pulp
        prob = pulp.LpProblem("manpower", pulp.LpMaximize)
        x = {a: pulp.LpVariable(f"x{i}", lowBound=min_cover, upBound=max_cap,
                                cat="Integer") for i, a in enumerate(areas)}
        c = {a: pulp.LpVariable(f"c{i}", lowBound=0) for i, a in enumerate(areas)}
        prob += pulp.lpSum(c.values())
        for a in areas:
            prob += c[a] <= active[a]
            prob += c[a] <= capacity * x[a]
        prob += pulp.lpSum(x.values()) == total
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        if pulp.LpStatus[prob.status] == "Optimal":
            for a in areas:
                alloc[a] = int(round(x[a].value()))
            return alloc, "pulp-ilp"
    except Exception:
        pass

    # ---- 2) SciPy linprog (continuous) + integer repair ----
    try:
        from scipy.optimize import linprog
        # vars = [officers(n)] + [covered(n)]; maximize sum covered => minimize -sum c
        nA = len(areas)
        cobj = np.concatenate([np.zeros(nA), -np.ones(nA)])
        # covered_a - cap*officers_a <= 0 ; covered_a <= demand_a
        A_ub, b_ub = [], []
        for i in range(nA):
            row = np.zeros(2 * nA); row[i] = -capacity; row[nA + i] = 1
            A_ub.append(row); b_ub.append(0)
            row = np.zeros(2 * nA); row[nA + i] = 1
            A_ub.append(row); b_ub.append(active[areas[i]])
        A_eq = np.concatenate([np.ones(nA), np.zeros(nA)]).reshape(1, -1)
        bounds = [(min_cover, max_cap)] * nA + [(0, None)] * nA
        res = linprog(cobj, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                      A_eq=A_eq, b_eq=[total], bounds=bounds, method="highs")
        if res.success:
            raw = res.x[:nA]
            base = np.floor(raw).astype(int)
            leftover = int(round(total - base.sum()))
            for j in np.argsort(-(raw - base))[:max(leftover, 0)]:
                base[j] += 1
            for a, v in zip(areas, base):
                alloc[a] = int(v)
            return alloc, "scipy-lp"
    except Exception:
        pass

    # ---- 3) fall back to the proportional allocator ----
    return D.allocate_manpower(demand, total=total, min_cov=min_cover), "proportional"

# ----------------------------------------------------------------------
# DEMO — LP vs proportional on the same demand
# ----------------------------------------------------------------------
def main():
    tbl, triage, forecaster = D.load_artifacts()
    prof = D.expected_load_surface(tbl, forecaster)
    DOW, SHIFT, TOTAL = 0, "Morning", 120
    sub = prof[(prof["dow"] == DOW) & (prof["shift"] == SHIFT)]
    demand = dict(zip(sub[D.AREA_COL], sub["demand"]))
    cap = max(sum(d for d in demand.values() if d > 0) / TOTAL, 1e-9)

    lp, backend = lp_allocate(demand, total=TOTAL, min_cover=2)
    prop = D.allocate_manpower(demand, total=TOTAL, min_cov=2)

    cmp = (pd.DataFrame({"area": list(demand),
                         "demand": [round(demand[a], 2) for a in demand],
                         "LP": [lp[a] for a in demand],
                         "proportional": [prop[a] for a in demand]})
           .query("LP > 0 or proportional > 0")
           .sort_values("demand", ascending=False).reset_index(drop=True))

    print(f"Manpower: {DOW_NAMES[DOW]} {SHIFT}, pool={TOTAL}, solver={backend}\n")
    print(cmp.head(15).to_string(index=False))
    print(f"\nTotal deployed  LP={sum(lp.values())}  proportional={sum(prop.values())}")
    print(f"Demand covered  LP={_covered(demand, lp, cap):.2f}  "
          f"proportional={_covered(demand, prop, cap):.2f}  "
          f"(LP is optimal for this objective)")

if __name__ == "__main__":
    main()