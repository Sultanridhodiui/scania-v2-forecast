import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

summary_df = pd.read_csv("data/final_summary_all_methods.csv", index_col="Rank")

# ============================================
# GRAFIK 1: Bar chart perbandingan WAPE semua metode per Rank
# ============================================
fig, ax = plt.subplots(figsize=(12, 6))
summary_df.plot(kind="bar", ax=ax)
ax.set_ylabel("WAPE (%)")
ax.set_xlabel("Rank")
ax.set_title("Perbandingan WAPE Semua Metode Forecasting per Rank\n(Scania Spare Parts v2)")
ax.legend(title="Metode", bbox_to_anchor=(1.02, 1), loc="upper left")
ax.set_xticklabels(summary_df.index, rotation=0)
plt.tight_layout()
plt.savefig("data/chart_wape_comparison_all_methods.png", dpi=150)
print("Tersimpan: data/chart_wape_comparison_all_methods.png")
plt.close()

# ============================================
# GRAFIK 2: Bar chart perbandingan WAPE (skala log, karena Rank D sangat tinggi
#           dan membuat perbedaan di Rank A/B sulit terlihat di skala linear)
# ============================================
fig, ax = plt.subplots(figsize=(12, 6))
summary_df.plot(kind="bar", ax=ax, logy=True)
ax.set_ylabel("WAPE (%) - skala log")
ax.set_xlabel("Rank")
ax.set_title("Perbandingan WAPE Semua Metode per Rank (Skala Logaritmik)")
ax.legend(title="Metode", bbox_to_anchor=(1.02, 1), loc="upper left")
ax.set_xticklabels(summary_df.index, rotation=0)
plt.tight_layout()
plt.savefig("data/chart_wape_comparison_logscale.png", dpi=150)
print("Tersimpan: data/chart_wape_comparison_logscale.png")
plt.close()

# ============================================
# GRAFIK 3: Heatmap - model terbaik per Rank (highlight)
# ============================================
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(summary_df.values, cmap="RdYlGn_r", aspect="auto")
ax.set_xticks(range(len(summary_df.columns)))
ax.set_xticklabels(summary_df.columns, rotation=45, ha="right")
ax.set_yticks(range(len(summary_df.index)))
ax.set_yticklabels(summary_df.index)
for i in range(len(summary_df.index)):
    for j in range(len(summary_df.columns)):
        val = summary_df.values[i, j]
        ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                color="white" if val > summary_df.values.mean() else "black", fontsize=9)
ax.set_title("Heatmap WAPE per Rank x Metode (hijau=baik, merah=buruk)")
plt.colorbar(im, ax=ax, label="WAPE (%)")
plt.tight_layout()
plt.savefig("data/chart_wape_heatmap.png", dpi=150)
print("Tersimpan: data/chart_wape_heatmap.png")
plt.close()

# ============================================
# GRAFIK 4: Line chart contoh time series - prediksi vs aktual (1 kombinasi Rank C)
# ============================================
ml = pd.read_csv("data/ml_predictions.csv", parse_dates=["Period"])
sample = ml[(ml["Rank"] == "C")].groupby(["Branch_ID", "Part_Number"]).size().idxmax()
branch_sel, part_sel = sample
line_data = ml[(ml["Branch_ID"] == branch_sel) & (ml["Part_Number"] == part_sel)].sort_values("Period")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(line_data["Period"], line_data["Actual_Demand_Qty"], label="Aktual", marker="o", markersize=4)
ax.plot(line_data["Period"], line_data["Pred_LightGBM"], label="Prediksi LightGBM", marker="x", markersize=4)
ax.set_title(f"Prediksi vs Aktual (Rank C) - {branch_sel}, {part_sel}")
ax.set_ylabel("Demand Quantity")
ax.legend()
plt.tight_layout()
plt.savefig("data/chart_prediction_vs_actual_rankC.png", dpi=150)
print("Tersimpan: data/chart_prediction_vs_actual_rankC.png")
plt.close()

print("\nSemua grafik tersimpan di folder data/")