import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

df = pd.read_csv("data/features_final.csv", parse_dates=["Period"])

# ============================================
# ENCODING kolom kategorikal
# ============================================
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
print(f"Data setelah dropna: {model_df.shape} (dari total {df.shape})")

train = model_df[model_df["Period"] < "2024-01-01"]
test = model_df[model_df["Period"] >= "2024-01-01"]

X_train, y_train = train[feature_cols], train["Actual_Demand_Qty"]
X_test, y_test = test[feature_cols], test["Actual_Demand_Qty"]

print(f"Training: {len(X_train)} baris, Testing: {len(X_test)} baris")

def wape(actual, forecast):
    return np.abs(actual - forecast).sum() / actual.sum() * 100

models = {
    "RandomForest": RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=150, max_depth=6, random_state=42, n_jobs=-1),
    "LightGBM": LGBMRegressor(n_estimators=150, max_depth=6, random_state=42, n_jobs=-1, verbose=-1),
}

test = test.copy()
predictions = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pred = np.maximum(0, pred)  # demand tidak boleh negatif
    test[f"Pred_{name}"] = pred
    predictions[name] = pred
    print(f"{name} - WAPE keseluruhan: {wape(y_test, pred):.2f}%")

# ============================================
# BREAKDOWN WAPE PER RANK (untuk perbandingan adil ke Croston/SARIMA)
# ============================================
print("\n=== WAPE per Rank, per Model ===")
summary_rows = []
for rank in ["A", "B", "C", "D"]:
    row = {"Rank": rank}
    subset = test[test["Rank"] == rank]
    for name in models.keys():
        row[name] = wape(subset["Actual_Demand_Qty"], subset[f"Pred_{name}"])
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows).set_index("Rank")
print(summary_df)

# Feature importance (XGBoost sebagai referensi)
importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": models["XGBoost"].feature_importances_
}).sort_values("importance", ascending=False)
print("\n=== Feature Importance (XGBoost) ===")
print(importance)

test.to_csv("data/ml_predictions.csv", index=False)
print("\nTersimpan: data/ml_predictions.csv")