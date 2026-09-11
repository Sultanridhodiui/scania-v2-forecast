import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

demand = pd.read_csv("data/clean_04_historical_demand.csv")
part = pd.read_csv("data/clean_02_part_master.csv")
branch = pd.read_csv("data/clean_01_branch_master.csv")
macro = pd.read_csv("data/clean_03_macroeconomic_variables.csv")

df = demand.merge(part[["Part_Number", "Rank", "Sector_Sensitivity"]], on="Part_Number")
df = df.merge(branch[["Branch_ID", "Sector_Focus", "Region"]], on="Branch_ID")

# ============================================
# 1. Statistik dasar per Rank (validasi desain data)
# ============================================
print("=== % Zero Demand per Rank ===")
print(df.groupby("Rank")["Actual_Demand_Qty"].apply(lambda x: (x == 0).mean() * 100).round(1))

print("\n=== Rata-rata demand per Rank ===")
print(df.groupby("Rank")["Actual_Demand_Qty"].mean().round(2))

# ============================================
# 2. Demand per Region
# ============================================
print("\n=== Total demand per Region ===")
print(df.groupby("Region")["Actual_Demand_Qty"].sum().sort_values(ascending=False))

# ============================================
# 3. Sector_Match effect
# ============================================
df["Sector_Match"] = (df["Sector_Focus"] == df["Sector_Sensitivity"]).astype(int)
print("\n=== Rata-rata demand: Sector_Match vs tidak ===")
print(df.groupby("Sector_Match")["Actual_Demand_Qty"].mean())

# ============================================
# 4. Visual: Demand nasional bulanan vs Coal Price Index
# ============================================
national_monthly = df.groupby(["Year", "Month"])["Actual_Demand_Qty"].sum().reset_index()
national_monthly = national_monthly.merge(macro[["Year", "Month", "Coal_Price_Index"]], on=["Year", "Month"])
national_monthly["Period"] = pd.to_datetime(national_monthly["Year"].astype(str) + "-" + national_monthly["Month"].astype(str) + "-01")
national_monthly = national_monthly.sort_values("Period")

fig, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(national_monthly["Period"], national_monthly["Actual_Demand_Qty"], color="tab:blue")
ax1.set_ylabel("Total Demand Nasional", color="tab:blue")
ax2 = ax1.twinx()
ax2.plot(national_monthly["Period"], national_monthly["Coal_Price_Index"], color="tab:red")
ax2.set_ylabel("Coal Price Index", color="tab:red")
plt.title("Total Demand Nasional vs Coal Price Index (2020-2024)")
plt.tight_layout()
plt.savefig("data/eda_demand_vs_coal_price.png", dpi=150)
print("\nTersimpan: data/eda_demand_vs_coal_price.png")
plt.close()

# ============================================
# 5. Visual: Boxplot demand per Rank
# ============================================
fig, ax = plt.subplots(figsize=(10, 6))
data_by_rank = [df[df["Rank"] == r]["Actual_Demand_Qty"].dropna() for r in ["A", "B", "C", "D"]]
ax.boxplot(data_by_rank, tick_labels=["A", "B", "C", "D"], showfliers=False)
ax.set_ylabel("Demand Quantity")
ax.set_title("Distribusi Demand per Rank (tanpa outlier ekstrem)")
plt.tight_layout()
plt.savefig("data/eda_boxplot_demand_per_rank.png", dpi=150)
print("Tersimpan: data/eda_boxplot_demand_per_rank.png")
plt.close()

# ============================================
# 6. MATRIKS KORELASI antar variabel numerik
# ============================================
numeric_df = df.merge(part[["Part_Number", "Unit_Price_IDR", "Standard_Leadtime_SeaFreight_Days"]],
                       on="Part_Number", how="left", suffixes=("", "_dup"))
numeric_df = numeric_df.merge(branch[["Branch_ID", "Branch_Size_Factor", "Distance_to_Central_Warehouse_KM"]],
                                on="Branch_ID", how="left")

corr_cols = ["Actual_Demand_Qty", "Unit_Price_IDR", "Standard_Leadtime_SeaFreight_Days",
             "Branch_Size_Factor", "Distance_to_Central_Warehouse_KM", "Sector_Match"]
corr_matrix = numeric_df[corr_cols].corr()

print("\n=== MATRIKS KORELASI ANTAR VARIABEL NUMERIK ===")
print(corr_matrix.round(3))

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_cols)))
ax.set_xticklabels(corr_cols, rotation=45, ha="right")
ax.set_yticks(range(len(corr_cols)))
ax.set_yticklabels(corr_cols)
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        ax.text(j, i, f"{corr_matrix.values[i,j]:.2f}", ha="center", va="center", fontsize=9)
plt.colorbar(im, ax=ax, label="Koefisien Korelasi")
ax.set_title("Matriks Korelasi Antar Variabel Numerik")
plt.tight_layout()
plt.savefig("data/eda_correlation_matrix.png", dpi=150)
print("\nTersimpan: data/eda_correlation_matrix.png")
plt.close()

# ============================================
# 7. Korelasi antar variabel makroekonomi (cek multikolinearitas)
# ============================================
macro_corr = macro[["Coal_Price_Index", "Nickel_Price_Index", "Construction_PMI",
                     "Fuel_Price_Index", "USD_IDR_Rate", "GDP_Growth_YoY_Pct"]].corr()
print("\n=== Korelasi antar Variabel Makroekonomi (cek multikolinearitas) ===")
print(macro_corr.round(3))

print("\n=== Korelasi demand nasional vs Coal Price Index ===")
print(national_monthly["Actual_Demand_Qty"].corr(national_monthly["Coal_Price_Index"]))