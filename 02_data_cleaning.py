import pandas as pd
import numpy as np

branch = pd.read_csv("data/01_branch_master.csv")
part = pd.read_csv("data/02_part_master.csv")
macro = pd.read_csv("data/03_macroeconomic_variables.csv")
demand = pd.read_csv("data/04_historical_demand.csv")
forecast = pd.read_csv("data/05_existing_forecast.csv")
psc = pd.read_csv("data/06_psc_adjustment.csv")

# ============================================
# 1. BRANCH MASTER - normalisasi Sector_Focus
# ============================================
sector_map = {
    "mining-coal": "Mining-Coal", "mining-nickel": "Mining-Nickel",
    "construction": "Construction", "forestry": "Forestry", "agriculture": "Agriculture",
}
branch["Sector_Focus"] = branch["Sector_Focus"].str.strip().str.lower().map(sector_map)
print("Branch Sector_Focus setelah cleaning:", branch["Sector_Focus"].unique())

# ============================================
# 2. PART MASTER - normalisasi Rank & Criticality_Level
# ============================================
part["Rank"] = part["Rank"].str.strip().str.upper()
part["Criticality_Level"] = part["Criticality_Level"].apply(
    lambda x: str(x).strip().title() if pd.notna(x) else np.nan
)
print("Part Rank setelah cleaning:", part["Rank"].unique())
print("Part Criticality setelah cleaning:", part["Criticality_Level"].unique())
print("Missing Criticality_Level:", part["Criticality_Level"].isna().sum())
print("Missing Unit_Price_IDR:", part["Unit_Price_IDR"].isna().sum())

# ============================================
# 3. HISTORICAL DEMAND - inti cleaning
# ============================================
print(f"\nDemand sebelum cleaning: {demand.shape}")

# 3a. negatif -> NaN (data entry error, sama seperti keputusan project sebelumnya)
demand.loc[demand["Actual_Demand_Qty"] < 0, "Actual_Demand_Qty"] = np.nan

# 3b. hapus duplikat penuh
demand = demand.drop_duplicates()
print(f"Demand setelah hapus duplikat: {demand.shape}")

# 3c. missing value -> BIARKAN NaN (keputusan konsisten dari project sebelumnya)
print("Missing Actual_Demand_Qty (dibiarkan):", demand["Actual_Demand_Qty"].isna().sum())

# ============================================
# 4. PSC ADJUSTMENT - isi missing reason
# ============================================
psc["Adjustment_Reason"] = psc["Adjustment_Reason"].fillna("Reason not recorded")

# ============================================
# SIMPAN HASIL CLEANING
# ============================================
branch.to_csv("data/clean_01_branch_master.csv", index=False)
part.to_csv("data/clean_02_part_master.csv", index=False)
demand.to_csv("data/clean_04_historical_demand.csv", index=False)
psc.to_csv("data/clean_06_psc_adjustment.csv", index=False)
macro.to_csv("data/clean_03_macroeconomic_variables.csv", index=False)
forecast.to_csv("data/clean_05_existing_forecast.csv", index=False)

print("\nSemua file cleaning tersimpan.")