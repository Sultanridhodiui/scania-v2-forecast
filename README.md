# Dataset Scania Spare Parts Forecasting v2 (Skala Besar)

Dataset sintetis untuk simulasi forecasting demand spare parts Scania di PT United Tractors Tbk,
skala industri (ratusan ribu baris), dengan dimensi cabang dan variabel makroekonomi sektoral.

Seed random: 123.

## Daftar File

| File | Baris | Isi |
|---|---|---|
| `01_branch_master.csv` | 36 | Cabang UT (kota, provinsi, region, sektor fokus, ukuran cabang) |
| `02_part_master.csv` | 150 | Part number Scania (kategori, rank, kritikalitas, harga, lead time, sensitivitas sektor) |
| `03_macroeconomic_variables.csv` | 60 | Variabel makro bulanan nasional (2020-2024) |
| `04_historical_demand.csv` | **327.240** | Demand aktual per Cabang x Part x Bulan |
| `05_existing_forecast.csv` | 9.000 | Forecast model existing UT, level nasional (Part x Bulan) |
| `06_psc_adjustment.csv` | 9.000 | Adjustment PSC, level nasional (Part x Bulan) |

## Poin Desain Penting

**1. Grain berbeda antar tabel** — `historical_demand` itu granular (per cabang), sementara
`existing_forecast` dan `psc_adjustment` itu **agregat nasional** (dijumlahkan semua cabang).
Ini meniru proses bisnis nyata: keputusan order ke principal Belgia/Singapore dibuat di level
nasional, tapi realisasi demand terjadi per cabang. Anda perlu **agregasi**
(`groupby(["Part_Number","Year","Month"]).sum()`) untuk membandingkan `historical_demand` dengan
2 tabel lainnya.

**2. Setiap part punya `Sector_Sensitivity`** (Mining-Coal, Mining-Nickel, Construction, Forestry,
Agriculture, atau General) — demand part itu akan lebih tinggi & lebih terpengaruh pergerakan
variabel makro terkait, TERUTAMA di cabang yang `Sector_Focus`-nya sama. Ini dasar untuk
menjelaskan kenapa fitur makroekonomi tertentu relevan untuk part/cabang tertentu, tapi tidak
untuk yang lain — insight yang bisa divalidasi lewat SARIMAX (exogenous variable) atau feature
importance ML.

**3. Ada shock pandemi** di 6 bulan pertama 2020 — GDP, harga batu bara, dan PMI konstruksi
sama-sama anjlok bertahap lalu pulih. Ini menambah kompleksitas tren yang realistis (bukan cuma
naik linear atau musiman sederhana).

**4. Data quality issues (sengaja, untuk latihan cleaning):**
- `historical_demand`: ~3% missing, ~0.8% nilai negatif (error input), ~0.3% outlier (typo
  tambahan nol), ~1% duplikat baris
- `part_master`: `Rank` dan `Criticality_Level` ada variasi kapitalisasi/spasi, beberapa
  `Criticality_Level` dan `Unit_Price_IDR` kosong
- `branch_master`: `Sector_Focus` ada variasi kapitalisasi/spasi

## Alur yang Disarankan (sesuai rencana Anda)

1. **Data Cleaning** — normalisasi teks kategorikal, tangani missing/negatif/outlier/duplikat di
   `historical_demand`, putuskan strategi untuk missing value di `part_master`.
2. **Feature Engineering** — lag & moving average per Branch+Part, fitur dari `Sector_Focus`
   (cabang) & `Sector_Sensitivity` (part) — misal buat kolom `Sector_Match` (1 jika sektor cabang
   = sektor sensitivitas part, 0 jika tidak), gabungkan variabel makro sesuai relevansi.
3. **Forecasting Statistik** (2 metode terbaik dari eksperimen sebelumnya — ARIMA untuk pola
   stabil, WMA/metode intermittent untuk pola jarang) + 1 metode sederhana sebagai baseline.
4. **Machine Learning** — beberapa algoritma relevan (Random Forest, XGBoost, LightGBM), dengan
   fitur lengkap termasuk makroekonomi & sektoral.
5. **Ensemble Learning** — gabungkan model-model di atas (weighted average / stacking / model
   routing berdasarkan Rank atau Sector_Match).
6. **Evaluasi Akhir** — bandingkan WAPE/MAE/RMSE seluruh metode (termasuk existing UT & PSC
   sebagai baseline pembanding bisnis), breakdown per Rank dan per Sector.
