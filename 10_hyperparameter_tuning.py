import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV

df = pd.read_csv("data/features_final.csv", parse_dates=["Period"])

df["Rank_Code"] = df["Rank"].map({"A": 1, "B": 2, "C": 3, "D": 4})
df["Criticality_Code"] = df["Criticality_Level"].map({"Critical": 4, "High": 3, "Medium": 2, "Low": 1})
df["Sector_Sensitivity_Code"] = df["Sector_Sensitivity"].astype("category").cat.codes
df["Sector_Focus_Code"] = df["Sector_Focus"].astype("category").cat.codes
df["Region_Code"] = df["Region"].astype("category").cat.codes

feature_cols = ["Rank_Code", "Criticality_Code", "Unit_Price_IDR", "Branch_Size_Factor",
                "Sector_Sensitivity_Code", "Sector_Focus_Code", "Region_Code", "Sector_Match",
                "Relevant_Macro_Value_Lag1", "Month",
                "Lag_1", "Lag_3", "MA_3", "MA_6", "Demand_Frequency_12M"]

model_df = df.dropna(subset=feature_cols + ["Actual_Demand_Qty"]).copy()
train = model_df[model_df["Period"] < "2024-01-01"]
test = model_df[model_df["Period"] >= "2024-01-01"]

X_train, y_train = train[feature_cols], train["Actual_Demand_Qty"]
X_test, y_test = test[feature_cols], test["Actual_Demand_Qty"]

def wape(actual, forecast):
    return np.abs(actual - forecast).sum() / actual.sum() * 100

# ============================================
# BASELINE: LightGBM default (untuk pembanding "sebelum tuning")
# ============================================
baseline = LGBMRegressor(n_estimators=150, max_depth=6, random_state=42, verbose=-1, n_jobs=-1)
baseline.fit(X_train, y_train)
baseline_pred = np.maximum(0, baseline.predict(X_test))
print(f"BASELINE (sebelum tuning) - WAPE: {wape(y_test, baseline_pred):.2f}%")

# ============================================
# HYPERPARAMETER TUNING
# ============================================
param_dist = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [4, 6, 8, 10, -1],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "num_leaves": [15, 31, 63, 127],
    "min_child_samples": [10, 20, 50],
}

print("\nMemulai RandomizedSearchCV (20 kombinasi x 3-fold CV = 60 model dilatih, mungkin makan waktu beberapa menit)...")
search = RandomizedSearchCV(
    LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1),
    param_dist, n_iter=20, cv=3, scoring="neg_mean_absolute_error",
    random_state=42, n_jobs=1, verbose=1  # n_jobs=1 di sini karena LGBM sudah pakai n_jobs=-1 sendiri
)
search.fit(X_train, y_train)

print("\n=== Best Parameters ===")
print(search.best_params_)

best_model = search.best_estimator_
best_pred = np.maximum(0, best_model.predict(X_test))
print(f"\nSETELAH TUNING - WAPE: {wape(y_test, best_pred):.2f}%")

print(f"\nPerbaikan: {wape(y_test, baseline_pred) - wape(y_test, best_pred):.2f} poin persentase")

# breakdown per Rank
test_copy = test.copy()
test_copy["Pred_Tuned"] = best_pred
print("\n=== WAPE per Rank (setelah tuning) ===")
for rank in ["A", "B", "C", "D"]:
    subset = test_copy[test_copy["Rank"] == rank]
    print(f"Rank {rank}: {wape(subset['Actual_Demand_Qty'], subset['Pred_Tuned']):.2f}%")

test_copy.to_csv("data/lightgbm_tuned_predictions.csv", index=False)
print("\nTersimpan: data/lightgbm_tuned_predictions.csv")