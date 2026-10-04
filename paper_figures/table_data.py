import pandas as pd

periods = ["Annual", "Winter", "Spring", "Summer", "Autumn"]
pcts = ["P75", "P90", "P95", "P99"]

# ---- Table 1 / S1: areal coverage (%) and mean scaling rate (%/degC) ----
cov = {
    # period: {category: [P75,P90,P95,P99]}
    "Annual": {"Super C-C": [45.2, 30.8, 21.3, 9.9],
               "C-C like": [28.7, 44.2, 54.9, 62.0],
               "Sub C-C":  [22.3, 20.0, 17.5, 18.0],
               "Negative": [3.8, 5.1, 6.3, 10.2]},
    "Winter": {"Super C-C": [4.1, 6.7, 6.4, 7.0],
               "C-C like": [14.4, 16.7, 17.2, 18.4],
               "Sub C-C":  [52.2, 49.9, 46.5, 41.9],
               "Negative": [29.3, 26.8, 29.9, 32.8]},
    "Spring": {"Super C-C": [15.8, 13.3, 12.1, 13.9],
               "C-C like": [44.4, 47.6, 47.6, 42.2],
               "Sub C-C":  [23.5, 22.1, 23.6, 27.2],
               "Negative": [16.3, 17.0, 16.8, 16.7]},
    "Summer": {"Super C-C": [4.1, 2.5, 2.3, 2.7],
               "C-C like": [16.9, 12.8, 10.0, 8.5],
               "Sub C-C":  [51.8, 53.5, 49.4, 45.6],
               "Negative": [27.3, 31.3, 38.2, 43.2]},
    "Autumn": {"Super C-C": [42.4, 25.3, 16.7, 10.9],
               "C-C like": [22.9, 37.5, 44.4, 39.3],
               "Sub C-C":  [21.1, 22.2, 23.4, 31.8],
               "Negative": [13.6, 15.0, 15.5, 18.0]},
}

mean_rate = {
    "Annual": {"Super C-C": [11.1, 10.7, 10.4, 10.8],
               "C-C like": [7.4, 7.6, 7.4, 6.8],
               "Sub C-C":  [2.7, 3.0, 3.1, 3.5],
               "Negative": [-2.4, -2.4, -3.0, -5.3]},
    "Winter": {"Super C-C": [11.6, 12.9, 13.0, 12.8],
               "C-C like": [6.5, 6.4, 6.5, 6.6],
               "Sub C-C":  [2.3, 2.5, 2.7, 2.7],
               "Negative": [-3.7, -3.9, -3.9, -4.3]},
    "Spring": {"Super C-C": [10.2, 10.8, 11.2, 11.7],
               "C-C like": [7.2, 7.1, 6.9, 6.8],
               "Sub C-C":  [2.7, 2.9, 2.9, 2.9],
               "Negative": [-3.3, -3.8, -4.0, -4.0]},
    "Summer": {"Super C-C": [12.6, 11.8, 10.9, 10.6],
               "C-C like": [6.6, 6.5, 6.5, 6.7],
               "Sub C-C":  [2.4, 2.4, 2.2, 2.1],
               "Negative": [-2.5, -2.5, -2.5, -3.4]},
    "Autumn": {"Super C-C": [11.3, 11.1, 11.0, 11.0],
               "C-C like": [7.4, 7.5, 7.2, 6.7],
               "Sub C-C":  [2.2, 2.5, 2.7, 3.3],
               "Negative": [-8.1, -8.9, -9.2, -9.3]},
}

rows = []
for p in periods:
    for c in ["Super C-C", "C-C like", "Sub C-C", "Negative"]:
        for i, q in enumerate(pcts):
            rows.append({"Period": p, "Category": c, "Percentile": q,
                         "Coverage_pct": cov[p][c][i],
                         "MeanRate_pct_per_C": mean_rate[p][c][i]})
t1 = pd.DataFrame(rows)

# ---- Table S2: monotonic / peak structure with sign decomposition, and T_peak ----
# values: [total, positive, negative] per percentile
s2 = {
    "Annual": {"Monotonic": {"P75": (47.2, 46.7, 0.5), "P90": (45.4, 45.0, 0.4),
                             "P95": (42.2, 42.1, 0.1), "P99": (37.0, 36.9, 0.1)},
               "Peak":      {"P75": (52.8, 49.4, 3.4), "P90": (54.6, 49.9, 4.7),
                             "P95": (57.8, 51.6, 6.2), "P99": (63.0, 53.0, 10.0)},
               "Tpeak": {"P75": 10.0, "P90": 9.4, "P95": 9.2, "P99": 9.1}},
    "Winter": {"Monotonic": {"P75": (40.9, 34.9, 6.0), "P90": (37.1, 30.8, 6.3),
                             "P95": (32.5, 23.5, 9.0), "P99": (29.5, 21.6, 7.9)},
               "Peak":      {"P75": (59.1, 35.8, 23.3), "P90": (62.9, 42.4, 20.5),
                             "P95": (67.5, 46.5, 21.0), "P99": (70.5, 45.6, 24.9)},
               "Tpeak": {"P75": -8.0, "P90": -7.8, "P95": -7.9, "P99": -8.1}},
    "Spring": {"Monotonic": {"P75": (57.8, 50.9, 6.9), "P90": (48.4, 44.6, 3.8),
                             "P95": (46.0, 42.6, 3.4), "P99": (36.5, 34.4, 2.1)},
               "Peak":      {"P75": (42.2, 32.6, 9.6), "P90": (51.6, 38.4, 13.2),
                             "P95": (54.0, 40.6, 13.4), "P99": (63.5, 48.8, 14.7)},
               "Tpeak": {"P75": 3.5, "P90": 2.6, "P95": 2.4, "P99": 2.9}},
    "Summer": {"Monotonic": {"P75": (30.3, 21.2, 9.1), "P90": (34.9, 23.8, 11.1),
                             "P95": (36.7, 21.5, 15.2), "P99": (33.4, 18.3, 15.1)},
               "Peak":      {"P75": (69.7, 51.4, 18.3), "P90": (65.1, 44.9, 20.2),
                             "P95": (63.4, 40.3, 23.1), "P99": (66.6, 38.5, 28.1)},
               "Tpeak": {"P75": 10.5, "P90": 10.4, "P95": 10.2, "P99": 9.9}},
    "Autumn": {"Monotonic": {"P75": (39.1, 37.3, 1.8), "P90": (40.4, 39.1, 1.3),
                             "P95": (44.0, 42.5, 1.5), "P99": (41.4, 39.5, 1.9)},
               "Peak":      {"P75": (60.9, 47.9, 13.0), "P90": (59.6, 45.9, 13.7),
                             "P95": (56.0, 42.0, 14.0), "P99": (58.6, 42.5, 16.1)},
               "Tpeak": {"P75": 7.7, "P90": 7.3, "P95": 7.1, "P99": 7.0}},
}

rows2 = []
for p in periods:
    for q in pcts:
        m = s2[p]["Monotonic"][q]
        k = s2[p]["Peak"][q]
        rows2.append({"Period": p, "Percentile": q,
                      "Monotonic_total": m[0], "Monotonic_pos": m[1], "Monotonic_neg": m[2],
                      "Peak_total": k[0], "Peak_pos": k[1], "Peak_neg": k[2],
                      "Tpeak_C": s2[p]["Tpeak"][q]})
t2 = pd.DataFrame(rows2)

with pd.ExcelWriter("scaling_tables.xlsx", engine="openpyxl") as xw:
    t1.to_excel(xw, sheet_name="Table1_S1_scaling", index=False)
    t2.to_excel(xw, sheet_name="TableS2_peak", index=False)

# sanity checks
chk1 = t1.groupby(["Period", "Percentile"])["Coverage_pct"].sum().round(2)
print("Coverage sums (should be ~100):")
print(chk1.to_string())
t2["sum"] = t2.Monotonic_total + t2.Peak_total
print("\nMono+Peak sums:", sorted(t2["sum"].round(2).unique()))
print("Decomp OK:", ((t2.Monotonic_pos + t2.Monotonic_neg - t2.Monotonic_total).abs() < 0.06).all(),
      ((t2.Peak_pos + t2.Peak_neg - t2.Peak_total).abs() < 0.06).all())
