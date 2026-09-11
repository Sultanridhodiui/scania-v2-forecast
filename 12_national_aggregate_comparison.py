import pandas as pd
import numpy as np

def wape(actual, forecast):
    valid = pd.notna(actual) & pd.notna(forecast)
    a, f = np.array(actual)[valid], np.array(forecast)[valid]
    if len(a) == 0 or a.sum() == 0:
        return np.nan
    return np.abs(a - f).sum() / a.sum() * 100

# ============================================
# 1. LOAD BASELINE BISNIS (Existing UT & PSC) - sudah level nasional
# ============================================
demand = pd.read_csv("data/clean_04_historical_demand.csv")
forecast = pd.read_csv("data/clean_05_existing_forecast.csv")
psc = pd.read_csv("data/clean_06_psc_adjustment.csv")
part = pd.read_csv("data/clean_02_part_master.csv")

national_actual = demand.groupby(["Part_Number", "Year", "Month"], as_index=False)["Actual_Demand_Qty"].sum()

baseline = forecast.merge(national_actual, on=["Part_Number", "Year", "Month"])
baseline = baseline.merge(psc[["Part_Number", "Year", "Month", "Final_Order_Quantity"]],
                            on=["Part_Number", "Year", "Month"])
baseline = baseline.merge(part[["Part_Number", "Rank"]], on="Part_Number")

# filter periode test yang sama (2024) biar adil
baseline_2024 = baseline[baseline["Year"] == 2024]

# ============================================
# 2. AGREGASI PREDIKSI MODEL KITA: dari per-cabang ke NASIONAL
# ============================================
ml_pred = pd.read_csv("data/model_per_rank_predictions.csv")  # hasil model terpisah per Rank

# Jumlahkan prediksi dari SEMUA cabang jadi 1 angka nasional per Part x Bulan
ml_national = ml_pred.groupby(["Part_Number", "Year", "Month", "Rank"], as_index=False).agg(
    Actual_National=("Actual_Demand_Qty", "sum"),
    Pred_National=("Pred_Model_Per_Rank", "sum")
)

# ============================================
# 3. BANDINGKAN APPLE-TO-APPLE: semua di level nasional, periode sama
# ============================================
print("=== PERBANDINGAN ADIL: Level Nasional, Periode 2024 ===\n")

comparison = []
for rank in ["A", "B", "C", "D"]:
    b_sub = baseline_2024[baseline_2024["Rank"] == rank]
    m_sub = ml_national[ml_national["Rank"] == rank]

    wape_existing = wape(b_sub["Actual_Demand_Qty"], b_sub["Raw_Forecast_Demand"])
    wape_psc = wape(b_sub["Actual_Demand_Qty"], b_sub["Final_Order_Quantity"])
    wape_model_national = wape(m_sub["Actual_National"], m_sub["Pred_National"])

    comparison.append({
        "Rank": rank,
        "Existing_UT": wape_existing,
        "PSC_Final_Order": wape_psc,
        "Model_Kita_(agregat_nasional)": wape_model_national
    })

comparison_df = pd.DataFrame(comparison).set_index("Rank")
print(comparison_df.round(1))

comparison_df.to_csv("data/fair_comparison_national_level.csv")
print("\nTersimpan: data/fair_comparison_national_level.csv")