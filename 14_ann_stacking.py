import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

tf.random.set_seed(42)
np.random.seed(42)

def wape(actual, forecast):
    valid = pd.notna(actual) & pd.notna(forecast)
    a, f = np.array(actual)[valid], np.array(forecast)[valid]
    if len(a) == 0 or a.sum() == 0:
        return np.nan
    return np.abs(a - f).sum() / a.sum() * 100

# ============================================
# 1. LOAD PREDIKSI BASE MODEL (RF, XGBoost, LightGBM) - dari SELURUH data, bukan sampel kecil
# ============================================
ml = pd.read_csv("data/ml_predictions.csv", parse_dates=["Period"])
ml["Rank_Code"] = ml["Rank"].map({"A": 1, "B": 2, "C": 3, "D": 4})

# ============================================
# 2. SPLIT META-TRAIN vs META-TEST (time-based, cegah leakage)
# ============================================
meta_train = ml[ml["Period"] < "2024-07-01"]
meta_test = ml[ml["Period"] >= "2024-07-01"]
print(f"Meta-train: {len(meta_train)} baris, Meta-test: {len(meta_test)} baris")

base_preds = ["Pred_RandomForest", "Pred_XGBoost", "Pred_LightGBM"]
feature_cols = base_preds + ["Rank_Code"]

X_meta_train = meta_train[feature_cols].fillna(0)
y_meta_train = meta_train["Actual_Demand_Qty"]
X_meta_test = meta_test[feature_cols].fillna(0)
y_meta_test = meta_test["Actual_Demand_Qty"]

# ============================================
# 3. NORMALISASI (WAJIB untuk ANN, beda dari tree-based yang tidak butuh scaling)
# ============================================
scaler = StandardScaler()
X_meta_train_scaled = scaler.fit_transform(X_meta_train)
X_meta_test_scaled = scaler.transform(X_meta_test)

# ============================================
# 4. BANGUN ANN META-MODEL
# ============================================
model = keras.Sequential([
    keras.layers.Input(shape=(X_meta_train_scaled.shape[1],)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="linear")
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mae")

early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

print("\nTraining ANN meta-model...")
history = model.fit(
    X_meta_train_scaled, y_meta_train,
    validation_split=0.2, epochs=100, batch_size=64,
    callbacks=[early_stop], verbose=1
)

ann_pred = np.maximum(0, model.predict(X_meta_test_scaled).flatten())

# ============================================
# 5. PEMBANDING: Ridge stacking (sama seperti sebelumnya, tapi sampel penuh sekarang)
# ============================================
ridge = Ridge(alpha=1.0, positive=True)
ridge.fit(X_meta_train[base_preds], y_meta_train)
ridge_pred = np.maximum(0, ridge.predict(X_meta_test[base_preds]))

# ============================================
# 6. EVALUASI FINAL
# ============================================
print("\n=== WAPE di Meta-Test (Jul-Des 2024, sampel penuh) ===")
print(f"Random Forest saja : {wape(y_meta_test, meta_test['Pred_RandomForest']):.2f}%")
print(f"XGBoost saja       : {wape(y_meta_test, meta_test['Pred_XGBoost']):.2f}%")
print(f"LightGBM saja      : {wape(y_meta_test, meta_test['Pred_LightGBM']):.2f}%")
print(f"Ridge Stacking     : {wape(y_meta_test, ridge_pred):.2f}%")
print(f"ANN Stacking       : {wape(y_meta_test, ann_pred):.2f}%")

print("\n=== Breakdown per Rank ===")
meta_test = meta_test.copy()
meta_test["Pred_Ridge"] = ridge_pred
meta_test["Pred_ANN"] = ann_pred

for rank in ["A", "B", "C", "D"]:
    sub = meta_test[meta_test["Rank"] == rank]
    print(f"\nRank {rank}:")
    print(f"  RF={wape(sub['Actual_Demand_Qty'], sub['Pred_RandomForest']):.1f}% | "
          f"XGB={wape(sub['Actual_Demand_Qty'], sub['Pred_XGBoost']):.1f}% | "
          f"LGBM={wape(sub['Actual_Demand_Qty'], sub['Pred_LightGBM']):.1f}% | "
          f"Ridge={wape(sub['Actual_Demand_Qty'], sub['Pred_Ridge']):.1f}% | "
          f"ANN={wape(sub['Actual_Demand_Qty'], sub['Pred_ANN']):.1f}%")

meta_test.to_csv("data/ann_stacking_results.csv", index=False)
print("\nTersimpan: data/ann_stacking_results.csv")