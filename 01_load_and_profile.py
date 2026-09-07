import pandas as pd

branch = pd.read_csv("data/01_branch_master.csv")
part = pd.read_csv("data/02_part_master.csv")
macro = pd.read_csv("data/03_macroeconomic_variables.csv")
demand = pd.read_csv("data/04_historical_demand.csv")
forecast = pd.read_csv("data/05_existing_forecast.csv")
psc = pd.read_csv("data/06_psc_adjustment.csv")

print("=== SHAPE SEMUA TABEL ===")
for name, df in [("branch", branch), ("part", part), ("macro", macro),
                  ("demand", demand), ("forecast", forecast), ("psc", psc)]:
    print(f"{name}: {df.shape}")

print("\n=== MISSING VALUE ===")
for name, df in [("branch", branch), ("part", part), ("macro", macro),
                  ("demand", demand), ("forecast", forecast), ("psc", psc)]:
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(f"\n{name}:")
        print(missing)

print("\n=== DUPLIKAT ===")
print("demand duplicated (semua kolom):", demand.duplicated().sum())

print("\n=== NILAI ANEH (negatif) di demand ===")
print("Jumlah baris negatif:", (demand["Actual_Demand_Qty"] < 0).sum())

print("\n=== VARIASI TEKS KATEGORIKAL (cek konsistensi) ===")
print("Rank unique:", part["Rank"].unique())
print("Criticality_Level unique:", part["Criticality_Level"].unique())
print("Sector_Focus unique:", branch["Sector_Focus"].unique())

print("\n=== SAMPLE DATA ===")
print(demand.head())