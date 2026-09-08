import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data/features_final.csv", parse_dates=["Period"])

def croston_sba(ts, alpha=0.1, n_forecast=1):
    """Croston's Method dengan koreksi SBA (Syntetos-Boylan Approximation)."""
    ts = np.array(ts)
    n = len(ts)
    demand_sizes, intervals = [], []
    last_nonzero_idx = -1

    for i, val in enumerate(ts):
        if val > 0:
            demand_sizes.append(val)
            if last_nonzero_idx >= 0:
                intervals.append(i - last_nonzero_idx)
            last_nonzero_idx = i

    if len(demand_sizes) < 2:
        # terlalu sedikit demand nonzero untuk Croston -> fallback ke rata-rata sederhana
        return np.repeat(np.mean(ts) if len(ts) > 0 else 0, n_forecast)

    # exponential smoothing untuk ukuran demand & interval
    smoothed_size = demand_sizes[0]
    smoothed_interval = intervals[0] if intervals else 1
    for i in range(1, len(demand_sizes)):
        smoothed_size = alpha * demand_sizes[i] + (1 - alpha) * smoothed_size
        if i < len(intervals):
            smoothed_interval = alpha * intervals[i] + (1 - alpha) * smoothed_interval

    forecast_value = (smoothed_size / smoothed_interval) * (1 - alpha / 2)  # koreksi SBA
    return np.repeat(max(0, forecast_value), n_forecast)

def wape(actual, forecast):
    valid = pd.notna(actual) & pd.notna(forecast)
    a, f = np.array(actual)[valid], np.array(forecast)[valid]
    if len(a) == 0 or a.sum() == 0:
        return np.nan
    return np.abs(a - f).sum() / a.sum() * 100

# Pakai sampel combos yang SAMA seperti SARIMA/SARIMAX (biar perbandingan adil)
sample_combos = []
for rank in ["A", "B", "C", "D"]:
    for match in [0, 1]:
        subset = df[(df["Rank"] == rank) & (df["Sector_Match"] == match)]
        combos = subset[["Branch_ID", "Part_Number"]].drop_duplicates()
        n_sample = min(10, len(combos))
        sampled = combos.sample(n_sample, random_state=42)
        for _, row in sampled.iterrows():
            sample_combos.append((row["Branch_ID"], row["Part_Number"], rank, match))

results = []
for branch_id, part_number, rank, match in sample_combos:
    data = df[(df["Branch_ID"] == branch_id) & (df["Part_Number"] == part_number)].sort_values("Period").reset_index(drop=True)
    train = data[data["Period"] < "2024-01-01"]
    test = data[data["Period"] >= "2024-01-01"]

    if len(test) == 0 or len(train) < 24:
        continue

    y_train = train["Actual_Demand_Qty"].fillna(0).values
    y_test = test["Actual_Demand_Qty"]

    forecast = croston_sba(y_train, alpha=0.1, n_forecast=len(test))
    wape_croston = wape(y_test, forecast)

    results.append({"Branch_ID": branch_id, "Part_Number": part_number, "Rank": rank,
                     "Sector_Match": match, "WAPE_Croston": wape_croston})

results_df = pd.DataFrame(results)
summary = results_df.groupby("Rank")["WAPE_Croston"].median()
print("=== MEDIAN WAPE Croston's Method per Rank ===")
print(summary)

results_df.to_csv("data/croston_sample_results.csv", index=False)
print("\nTersimpan: data/croston_sample_results.csv")