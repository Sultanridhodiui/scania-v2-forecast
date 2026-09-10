import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor

df = pd.read_csv("data/features_final.csv", parse_dates=["Period"])

df["Criticality_Code"] = df["Criticality_Level"].map({"Critical": 4, "High": 3, "Medium": 2, "Low": 1})
df["Sector_Sensitivity_Code"] = df["Sector_Sensitivity"].astype("category").cat.codes
df["Sector_Focus_Code"] = df["Sector_Focus"].astype("category").cat.codes
df["Region_Code"] = df["Region"].astype("category").cat.codes

# catatan: Rank_Code TIDAK dipakai lagi sebagai fitur, karena sekarang model dipisah PER Rank
# (jadi tidak perlu diberi tahu "ini Rank apa" -- itu sudah pasti/given)
feature_cols = ["Criticality_Code", "Unit_Price_IDR", "Branch_Size_Factor",
                "Sector_Sensitivity_Code", "Sector_Focus_Code", "Region_Code", "Sector_Match",
                "Relevant_Macro_Value_Lag1", "Month",
                "Lag_1", "Lag_3", "MA_3", "MA_6", "Demand_Frequency_12M"]

model_df = df.dropna(subset=feature_cols + ["Actual_Demand_Qty"]).copy()

def wape(actual, forecast):
    valid_sum = actual.sum()
    if valid_sum == 0:
        return np.nan
    return np.abs(actual - forecast).sum() / valid_sum * 100

results = []
all_predictions = []

for rank in ["A", "B", "C", "D"]:
    print(f"\n=== Training model khusus Rank {rank} ===")
    rank_df = model_df[model_df["Rank"] == rank]

    train = rank_df[rank_df["Period"] < "2024-01-01"]
    test = rank_df[rank_df["Period"] >= "2024-01-01"]

    X_train, y_train = train[feature_cols], train["Actual_Demand_Qty"]
    X_test, y_test = test[feature_cols], test["Actual_Demand_Qty"]

    print(f"Training: {len(X_train)} baris, Testing: {len(X_test)} baris")

    model = LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                           num_leaves=31, min_child_samples=20, random_state=42, verbose=-1, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = np.maximum(0, model.predict(X_test))

    wape_score = wape(y_test, pred)
    print(f"Rank {rank} - WAPE (model terpisah): {wape_score:.2f}%")
    results.append({"Rank": rank, "WAPE_Model_Per_Rank": wape_score})

    test_copy = test.copy()
    test_copy["Pred_Model_Per_Rank"] = pred
    all_predictions.append(test_copy)

results_df = pd.DataFrame(results)
print("\n=== RINGKASAN: Model Terpisah per Rank ===")
print(results_df)

# ============================================
# BANDINGKAN ke model gabungan (LightGBM 1-model-untuk-semua yang sudah dievaluasi sebelumnya)
# ============================================
ml_combined = pd.read_csv("data/ml_predictions.csv")
print("\n=== PERBANDINGAN: 1 Model Gabungan vs Model Terpisah per Rank ===")
for rank in ["A", "B", "C", "D"]:
    combined_wape = wape(ml_combined[ml_combined["Rank"] == rank]["Actual_Demand_Qty"],
                          ml_combined[ml_combined["Rank"] == rank]["Pred_LightGBM"])
    separate_wape = results_df[results_df["Rank"] == rank]["WAPE_Model_Per_Rank"].values[0]
    improvement = combined_wape - separate_wape
    print(f"Rank {rank}: Gabungan={combined_wape:.1f}% | Terpisah={separate_wape:.1f}% | Perbaikan={improvement:+.1f}pp")

all_pred_df = pd.concat(all_predictions, ignore_index=True)
all_pred_df.to_csv("data/model_per_rank_predictions.csv", index=False)
print("\nTersimpan: data/model_per_rank_predictions.csv")