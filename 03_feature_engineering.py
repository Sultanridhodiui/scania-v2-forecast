import pandas as pd
import numpy as np

demand = pd.read_csv("data/clean_04_historical_demand.csv")
part = pd.read_csv("data/clean_02_part_master.csv")
branch = pd.read_csv("data/clean_01_branch_master.csv")
macro = pd.read_csv("data/clean_03_macroeconomic_variables.csv")

# ============================================
# 1. GABUNGKAN dengan master data
# ============================================
df = demand.merge(part[["Part_Number", "Rank", "Criticality_Level", "Unit_Price_IDR", "Sector_Sensitivity"]],
                   on="Part_Number", how="left")
df = df.merge(branch[["Branch_ID", "Sector_Focus", "Branch_Size_Factor", "Region"]],
              on="Branch_ID", how="left")
df = df.merge(macro, on=["Year", "Month"], how="left")

print("Shape setelah merge:", df.shape)

# ============================================
# 2. FITUR SECTOR_MATCH (kunci untuk insight sektoral)
# ============================================
df["Sector_Match"] = (df["Sector_Focus"] == df["Sector_Sensitivity"]).astype(int)
print("\nDistribusi Sector_Match:", df["Sector_Match"].value_counts().to_dict())

# ============================================
# 3. FITUR MAKRO YANG RELEVAN (dipilih sesuai Sector_Sensitivity part)
# ============================================
sector_to_macro_col = {
    "Mining-Coal": "Coal_Price_Index",
    "Mining-Nickel": "Nickel_Price_Index",
    "Construction": "Construction_PMI",
    "Forestry": "Fuel_Price_Index",
    "Agriculture": "GDP_Growth_YoY_Pct",
    "General": None,
}

def get_relevant_macro(row):
    col = sector_to_macro_col.get(row["Sector_Sensitivity"])
    return row[col] if col is not None else np.nan

df["Relevant_Macro_Value"] = df.apply(get_relevant_macro, axis=1)
print("\nMissing Relevant_Macro_Value (wajar untuk Sector_Sensitivity='General'):",
      df["Relevant_Macro_Value"].isna().sum())

# ============================================
# 4. LAG & MOVING AVERAGE per Branch+Part
# ============================================
df["Period"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01")
df = df.sort_values(["Branch_ID", "Part_Number", "Period"]).reset_index(drop=True)

def add_ts_features(group):
    group = group.sort_values("Period").copy()
    group["Lag_1"] = group["Actual_Demand_Qty"].shift(1)
    group["Lag_3"] = group["Actual_Demand_Qty"].shift(3)
    group["MA_3"] = group["Actual_Demand_Qty"].shift(1).rolling(3, min_periods=1).mean()
    group["MA_6"] = group["Actual_Demand_Qty"].shift(1).rolling(6, min_periods=1).mean()
    group["Demand_Frequency_12M"] = group["Actual_Demand_Qty"].shift(1).gt(0).rolling(12, min_periods=1).sum()
    return group

print("\nMemproses fitur time series per Branch+Part (mungkin perlu waktu beberapa menit)...")
df = pd.concat([add_ts_features(g) for _, g in df.groupby(["Branch_ID", "Part_Number"])], ignore_index=True)

print("\nShape akhir:", df.shape)
print(df[["Branch_ID", "Part_Number", "Period", "Actual_Demand_Qty", "Sector_Match",
          "Relevant_Macro_Value", "Lag_1", "MA_3"]].head(10))

df.to_csv("data/features_final.csv", index=False)
print("\nTersimpan: data/features_final.csv")
