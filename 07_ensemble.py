import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

# ============================================
# 1. LOAD & GABUNGKAN PREDIKSI DARI 3 METODE
# ============================================
sarima = pd.read_csv("data/sarima_predictions_detail.csv", parse_dates=["Period"])
croston = pd.read_csv("data/croston_predictions_detail.csv", parse_dates=["Period"])
ml = pd.read_csv("data/ml_predictions.csv", parse_dates=["Period"])

# ambil kolom yang relevan saja dari ML, gunakan LightGBM (juara di antara model ML)
ml_slim = ml[["Branch_ID", "Part_Number", "Period", "Actual_Demand_Qty", "Pred_LightGBM"]]

combined = croston.merge(sarima, on=["Branch_ID", "Part_Number", "Period"], how="inner")
combined = combined.merge(ml_slim, on=["Branch_ID", "Part_Number", "Period"], how="inner")

print("Shape gabungan (harus match di 80 kombinasi x periode test):", combined.shape)
print(combined.head())

# ============================================
# 2. SPLIT META-TRAIN vs META-TEST (mencegah leakage di level ensemble)
#    Meta-train: Jan-Jun 2024 (untuk melatih bobot ensemble)
#    Meta-test: Jul-Des 2024 (untuk evaluasi FINAL, belum pernah dilihat meta-model)
# ============================================
meta_train = combined[combined["Period"] < "2024-07-01"]
meta_test = combined[combined["Period"] >= "2024-07-01"]

print(f"\nMeta-train: {len(meta_train)} baris, Meta-test: {len(meta_test)} baris")

feature_models = ["Pred_Croston", "Pred_SARIMA", "Pred_LightGBM"]

X_meta_train = meta_train[feature_models].fillna(0)
y_meta_train = meta_train["Actual_Demand_Qty"]
X_meta_test = meta_test[feature_models].fillna(0)
y_meta_test = meta_test["Actual_Demand_Qty"]

# ============================================
# 3. TRAINING META-MODEL (Ridge Regression)
# ============================================
meta_model = Ridge(alpha=1.0, positive=True)  # positive=True agar bobot tidak negatif (lebih masuk akal bisnis)
meta_model.fit(X_meta_train, y_meta_train)

print("\n=== Bobot yang dipelajari meta-model ===")
for name, coef in zip(feature_models, meta_model.coef_):
    print(f"{name}: {coef:.4f}")
print(f"Intercept: {meta_model.intercept_:.4f}")

# ============================================
# 4. EVALUASI DI META-TEST (data yang benar-benar belum pernah dilihat)
# ============================================
def wape(actual, forecast):
    valid = pd.notna(actual) & pd.notna(forecast)
    a, f = np.array(actual)[valid], np.array(forecast)[valid]
    if len(a) == 0 or a.sum() == 0:
        return np.nan
    return np.abs(a - f).sum() / a.sum() * 100

ensemble_pred = np.maximum(0, meta_model.predict(X_meta_test))

print("\n=== WAPE di Meta-Test (perbandingan adil, periode sama) ===")
print(f"Croston's saja  : {wape(y_meta_test, meta_test['Pred_Croston']):.2f}%")
print(f"SARIMA saja     : {wape(y_meta_test, meta_test['Pred_SARIMA']):.2f}%")
print(f"LightGBM saja   : {wape(y_meta_test, meta_test['Pred_LightGBM']):.2f}%")
print(f"ENSEMBLE (Ridge): {wape(y_meta_test, ensemble_pred):.2f}%")

# Simple averaging sebagai pembanding tambahan (ensemble paling sederhana)
simple_avg = meta_test[feature_models].mean(axis=1)
print(f"Simple Average  : {wape(y_meta_test, simple_avg):.2f}%")

meta_test = meta_test.copy()
meta_test["Pred_Ensemble"] = ensemble_pred
meta_test.to_csv("data/ensemble_final_results.csv", index=False)
print("\nTersimpan: data/ensemble_final_results.csv")