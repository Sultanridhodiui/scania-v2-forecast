import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data/features_final.csv", parse_dates=["Period"])

sample_combos = []
for rank in ["A", "B", "C", "D"]:
    for match in [0, 1]:
        subset = df[(df["Rank"] == rank) & (df["Sector_Match"] == match)]
        combos = subset[["Branch_ID", "Part_Number"]].drop_duplicates()
        n_sample = min(10, len(combos))
        sampled = combos.sample(n_sample, random_state=42)
        for _, row in sampled.iterrows():
            sample_combos.append((row["Branch_ID"], row["Part_Number"], rank, match))

print(f"Total kombinasi yang diuji: {len(sample_combos)}")

def wape(actual, forecast):
    valid = pd.notna(actual) & pd.notna(forecast)
    a, f = np.array(actual)[valid], np.array(forecast)[valid]
    if len(a) == 0 or a.sum() == 0:
        return np.nan
    return np.abs(a - f).sum() / a.sum() * 100

results = []

for branch_id, part_number, rank, match in sample_combos:
    data = df[(df["Branch_ID"] == branch_id) & (df["Part_Number"] == part_number)].sort_values("Period").reset_index(drop=True)

    train = data[data["Period"] < "2024-01-01"]
    test = data[data["Period"] >= "2024-01-01"]

    if len(test) == 0 or len(train) < 24:
        continue

    y_train = train["Actual_Demand_Qty"].fillna(0)
    y_test = test["Actual_Demand_Qty"]

    try:
        sarima_model = SARIMAX(y_train, order=(1,1,1), seasonal_order=(1,1,1,12),
                                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        sarima_forecast = sarima_model.forecast(len(test))
        wape_sarima = wape(y_test, sarima_forecast.values)
    except Exception:
        wape_sarima = np.nan

    macro_train = train["Relevant_Macro_Value_Lag1"]
    if macro_train.notna().sum() == 0:
        wape_sarimax = np.nan
    else:
        try:
            fill_val = macro_train.mean()
            exog_train = train[["Relevant_Macro_Value_Lag1"]].fillna(fill_val)
            exog_test = test[["Relevant_Macro_Value_Lag1"]].fillna(fill_val)

            sarimax_model = SARIMAX(y_train, exog=exog_train, order=(1,1,1), seasonal_order=(1,1,1,12),
                                     enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            sarimax_forecast = sarimax_model.forecast(len(test), exog=exog_test)
            wape_sarimax = wape(y_test, sarimax_forecast.values)
        except Exception:
            wape_sarimax = np.nan

    results.append({
        "Branch_ID": branch_id, "Part_Number": part_number, "Rank": rank, "Sector_Match": match,
        "WAPE_SARIMA": wape_sarima, "WAPE_SARIMAX": wape_sarimax
    })

results_df = pd.DataFrame(results)
print(f"\nBerhasil diproses: {len(results_df)} kombinasi")

summary = results_df.groupby(["Rank", "Sector_Match"])[["WAPE_SARIMA", "WAPE_SARIMAX"]].median()
print("\n=== MEDIAN WAPE per Rank x Sector_Match (dengan Lag1 macro) ===")
print(summary)

results_df.to_csv("data/sarima_sarimax_sample_results.csv", index=False)
print("\nTersimpan: data/sarima_sarimax_sample_results.csv")