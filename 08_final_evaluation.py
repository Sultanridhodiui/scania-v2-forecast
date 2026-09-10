import pandas as pd
import numpy as np

def wape(actual, forecast):
    valid = pd.notna(actual) & pd.notna(forecast)
    a, f = np.array(actual)[valid], np.array(forecast)[valid]
    if len(a) == 0 or a.sum() == 0:
        return np.nan
    return np.abs(a - f).sum() / a.sum() * 100

# Load semua hasil prediksi yang sudah kita simpan
sarima = pd.read_csv("data/sarima_predictions_detail.csv", parse_dates=["Period"])
croston = pd.read_csv("data/croston_predictions_detail.csv", parse_dates=["Period"])
ml = pd.read_csv("data/ml_predictions.csv", parse_dates=["Period"])
ensemble = pd.read_csv("data/ensemble_final_results.csv", parse_dates=["Period"])

rank_lookup = pd.read_csv("data/croston_sample_results.csv")[["Branch_ID", "Part_Number", "Rank"]].drop_duplicates()

combined = croston.merge(sarima, on=["Branch_ID", "Part_Number", "Period"], how="inner")
combined = combined.merge(rank_lookup, on=["Branch_ID", "Part_Number"], how="left")
ensemble = ensemble.merge(rank_lookup, on=["Branch_ID", "Part_Number"], how="left")

print("=== TABEL RINGKASAN AKHIR: WAPE per Rank, SEMUA METODE ===\n")

summary_final = []
for rank in ["A", "B", "C", "D"]:
    row = {"Rank": rank}

    sub_combined = combined[combined["Rank"] == rank]
    row["SARIMA"] = wape(sub_combined["Actual"], sub_combined["Pred_SARIMA"])
    row["Croston"] = wape(sub_combined["Actual"], sub_combined["Pred_Croston"])

    sub_ml = ml[ml["Rank"] == rank]
    row["RandomForest"] = wape(sub_ml["Actual_Demand_Qty"], sub_ml["Pred_RandomForest"])
    row["XGBoost"] = wape(sub_ml["Actual_Demand_Qty"], sub_ml["Pred_XGBoost"])
    row["LightGBM"] = wape(sub_ml["Actual_Demand_Qty"], sub_ml["Pred_LightGBM"])

    sub_ens = ensemble[ensemble["Rank"] == rank]
    row["Ensemble_Ridge"] = wape(sub_ens["Actual_Demand_Qty"], sub_ens["Pred_Ensemble"])

    summary_final.append(row)

summary_df = pd.DataFrame(summary_final).set_index("Rank")
print(summary_df.round(1))

print("\n=== MODEL TERBAIK PER RANK ===")
for rank in summary_df.index:
    best_model = summary_df.loc[rank].idxmin()
    best_value = summary_df.loc[rank].min()
    print(f"Rank {rank}: {best_model} (WAPE {best_value:.1f}%)")

summary_df.to_csv("data/final_summary_all_methods.csv")
print("\nTersimpan: data/final_summary_all_methods.csv")

print("\n" + "="*60)
print("CATATAN METODOLOGIS PENTING:")
print("="*60)
print("""
- SARIMA & Croston's: dievaluasi dari 80 sampel kombinasi Branch+Part,
  periode penuh Jan-Des 2024 (12 bulan).
- RandomForest/XGBoost/LightGBM: dievaluasi dari SELURUH kombinasi
  (ribuan Branch+Part), periode Jan-Des 2024.
- Ensemble: dievaluasi HANYA dari 80 sampel yang sama, periode Jul-Des 2024
  saja (6 bulan, karena Jan-Jun dipakai untuk training meta-model).
- KARENA PERBEDAAN CAKUPAN SAMPEL & PERIODE INI, perbandingan antar
  kategori (statistik vs ML vs ensemble) sebaiknya dibaca sebagai
  indikasi arah/tren, bukan perbandingan apple-to-apple yang sempurna.
""")