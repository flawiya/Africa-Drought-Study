#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# 1. CONFIGURATION AND BASE PATHS
# =============================================================================
BASE_PATH = r"outputs\Zambia\Files"
OUTPUT_PATH = os.path.join(BASE_PATH, "Analysis_Outputs")
os.makedirs(OUTPUT_PATH, exist_ok=True)

TARGET_DISTRICTS = [
    'Chirundu', 'Choma', 'Gwembe', 'Kalomo', 'Kazungula', 
    'Mazabuka', 'Monze', 'Namwala', 'Pemba', 'Siavonga', 
    'Sinazongwe', 'Zimba'
]

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def get_crop_year(row):
    """Aligns Nov-Dec data points to the following calendar harvest year."""
    return row['date'].year + 1 if row['date'].month in [11, 12] else row['date'].year

def calculate_sheffield_baseline(df, col, window=31):
    """
    Calculates a stable historical baseline using a 31-day moving window 
    across all years for each Julian day (Sheffield et al., 2014).
    """
    # 1. Calculate raw daily mean/std across all years
    daily_stats = df.groupby(['district', 'day_of_year'])[col].agg(['mean', 'std']).reset_index()
    
    results = []
    # 2. Iterate through districts to apply circular rolling logic
    for district, group in daily_stats.groupby('district'):
        group = group.sort_values('day_of_year')
        
        # Triple data to handle circular wrap-around (Dec 31 -> Jan 1)
        triple = pd.concat([group] * 3).reset_index(drop=True)
        
        # Calculate moving averages
        triple['mu_stable'] = triple['mean'].rolling(window=window, center=True).mean()
        triple['std_stable'] = triple['std'].rolling(window=window, center=True).mean()
        
        # Take the middle section (the original 366 days)
        middle = triple.iloc[len(group):2*len(group)].copy()
        results.append(middle)

    # 3. Combine results
    stable_baseline = pd.concat(results, ignore_index=True)
    
    # Return only necessary columns for the merge
    return stable_baseline[['district', 'day_of_year', 'mu_stable', 'std_stable']]

def calculate_neg_integral(series, threshold=-0.8):
    """Calculates the cumulative 'Drought Energy' (Area Under the Curve)."""
    stress = series[series < threshold]
    if stress.empty:
        return 0
    return (stress - threshold).sum()

# =============================================================================
# 3. DATA LOADING AND INITIAL MERGE
# =============================================================================

print("📂 Loading data sources...")
df_soil = pd.read_csv(os.path.join(BASE_PATH, "master_southern_province_soil-moisture-layer2.csv"))
df_ndvi = pd.read_csv(os.path.join(BASE_PATH, "master_southern_province_ndvi.csv"))
df_lst = pd.read_csv(os.path.join(BASE_PATH, "master_southern_province_lst.csv"))
df_climate = pd.read_csv(os.path.join(BASE_PATH, "climate_merged.csv"))
df_spei_data = pd.read_csv(os.path.join(BASE_PATH, "master_southern_province_data.csv"))

# Standardize Dates
for df in [df_soil, df_ndvi, df_lst, df_climate, df_spei_data]:
    df['date'] = pd.to_datetime(df['date'])

print("🔗 Merging datasets...")
# Base reanalysis merge (Soil + Precip/PET)
master = pd.merge(df_soil[['date', 'district', 'soil_moisture_layer2']], 
                 df_spei_data[['date', 'district', 'precip_mm', 'pet_mm']], 
                 on=['date', 'district'], how='inner')

# Clean and merge LST (Land Surface Temp)
df_lst['lst_celsius'] = df_lst['lst_celsius'].replace(-999.0, np.nan)
master = pd.merge(master, df_lst[['date', 'district', 'lst_celsius']], on=['date', 'district'], how='left')

# Merge NDVI and Global Climate state (NINO3.4)
master = pd.merge(master, df_ndvi[['date', 'district', 'ndvi']], on=['date', 'district'], how='left')
master = pd.merge(master, df_climate[['date', 'NINO34']], on='date', how='left')

# Setup helper columns for grouping
master['day_of_year'] = master['date'].dt.dayofyear
master['crop_year'] = master.apply(get_crop_year, axis=1)

# =============================================================================
# 4. INDEX CALCULATION (SSI, SPEI3, VHI)
# =============================================================================

print("📉 Calculating Standardized Indices (Sheffield 2014 Methodology)...")

# --- SSI (Standardized Soil Moisture Index) ---
baseline_stats_ssi = calculate_sheffield_baseline(master, 'soil_moisture_layer2')
master = master.merge(baseline_stats_ssi, on=['district', 'day_of_year'], how='left')
master['SSI'] = (master['soil_moisture_layer2'] - master['mu_stable']) / (master['std_stable'] + 1e-6)

# --- SPEI-3 (90-Day Water Balance Index) ---
master['precip_mm'] = master['precip_mm'].clip(lower=0)
master['D'] = master['precip_mm'] - master['pet_mm'].abs()
# 90-day rolling sum (Meteorological supply/demand balance)
master['D_90'] = master.groupby('district')['D'].transform(lambda x: x.rolling(window=90, min_periods=28).sum())

baseline_stats_spei = calculate_sheffield_baseline(master, 'D_90')
baseline_stats_spei = baseline_stats_spei.rename(columns={'mu_stable': 'mu_d90', 'std_stable': 'std_d90'})
master = master.merge(baseline_stats_spei, on=['district', 'day_of_year'], how='left')
master['SPEI3'] = (master['D_90'] - master['mu_d90']) / (master['std_d90'] + 1e-6)

# --- VHI (Vegetation Health Index) ---
# Data Smoothing
master['ndvi'] = master.groupby('district')['ndvi'].transform(lambda x: x.interpolate().ffill().bfill())
master['lst_celsius'] = master.groupby('district')['lst_celsius'].transform(lambda x: x.interpolate().ffill().bfill())

# Calculate Z-scores for NDVI (VCI) and LST (TCI) relative to each day of year
master['VCI_z'] = master.groupby(['district', 'day_of_year'])['ndvi'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))
master['TCI_z'] = master.groupby(['district', 'day_of_year'])['lst_celsius'].transform(lambda x: (x.mean() - x) / (x.std() + 1e-6))
master['VHI'] = (0.5 * master['VCI_z']) + (0.5 * master['TCI_z'])

# Smooth results (7-day window to eliminate sensor jitter)
for col in ['SSI', 'SPEI3', 'VHI']:
    master[col] = master.groupby('district')[col].transform(lambda x: x.rolling(window=7, center=True, min_periods=1).mean())

# =============================================================================
# 5. SEASONAL AGGREGATION & EXPORT
# =============================================================================

# Filter for strictly agricultural months (November to April)
master_crop = master[master['date'].dt.month.isin([11, 12, 1, 2, 3, 4])].copy()

print("📋 Generating Seasonal Aggregation Matrix for 2024...")
seasonal_matrix = master_crop[master_crop['crop_year'] == 2024].groupby('district').agg(
    SSI_Mean=('SSI', 'mean'),
    SSI_Min=('SSI', 'min'),
    SSI_Neg_Integral=('SSI', calculate_neg_integral),
    VHI_Mean=('VHI', 'mean'),
    NDVI_Max=('ndvi', 'max'),
    SPEI3_Min=('SPEI3', 'min'),
    NINO34_Avg=('NINO34', 'mean')
).reset_index().sort_values(by='SSI_Neg_Integral')

# Save Matrix
matrix_path = os.path.join(OUTPUT_PATH, "Seasonal_Drought_Matrix_2024.csv")
seasonal_matrix.to_csv(matrix_path, index=False)

# =============================================================================
# 6. VISUALIZATION 1: THE DROUGHT CASCADE (2024)
# =============================================================================

print("📊 Rendering Spatiotemporal Drought Cascade (2024)...")
df_2024 = master_crop[master_crop['crop_year'] == 2024].copy()
df_prov_avg = df_2024.groupby('date').mean(numeric_only=True).reset_index()

fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                    subplot_titles=("1. Meteorological: Daily Rainfall vs. Evaporation (PET)", 
                                    "2. Hydrological: Daily SSI (Julian-Day Baseline)", 
                                    "3. Cumulative: SPEI-3 (90-Day Anomaly)", 
                                    "4. Biological: Vegetation Health Index (VHI)"))

# Row 1: Rain/PET
fig.add_trace(go.Bar(x=df_prov_avg['date'], y=df_prov_avg['precip_mm'], name="Avg Rain", marker_color='dodgerblue'), row=1, col=1)
fig.add_trace(go.Scatter(x=df_prov_avg['date'], y=df_prov_avg['pet_mm'].abs(), name="Avg PET", line=dict(color='crimson', width=2)), row=1, col=1)

# Row 2 & 3: Shaded anomalies
for r, col in [(2, 'SSI'), (3, 'SPEI3')]:
    for d in TARGET_DISTRICTS:
        if d in df_2024['district'].values:
            df_d = df_2024[df_2024['district'] == d]
            fig.add_trace(go.Scatter(x=df_d['date'], y=df_d[col], line=dict(color='rgba(150,150,150,0.12)', width=1), showlegend=False), row=r, col=1)
    
    fig.add_trace(go.Scatter(x=df_prov_avg['date'], y=df_prov_avg[col].clip(lower=0), fill='tozeroy', fillcolor='rgba(0, 0, 255, 0.2)', line_color='blue', name=f"Prov {col} (Wet)"), row=r, col=1)
    fig.add_trace(go.Scatter(x=df_prov_avg['date'], y=df_prov_avg[col].clip(upper=0), fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.2)', line_color='red', name=f"Prov {col} (Dry)"), row=r, col=1)

# Row 4: Biological VHI
fig.add_trace(go.Scatter(x=df_prov_avg['date'], y=df_prov_avg['VHI'], line=dict(color='darkgreen', width=3), name="Avg VHI"), row=4, col=1)

# Add thresholds
for r in [2, 3]:
    fig.add_hline(y=-1.2, line_dash="dash", line_color="darkred", annotation_text="Extreme Drought", row=r, col=1)

fig.update_layout(height=1200, title_text="Southern Province: Spatiotemporal Drought Analysis (2024)", template="plotly_white", hovermode="x unified")
fig.write_html(os.path.join(OUTPUT_PATH, "Drought_Cascade_2024.html"))

# =============================================================================
# 7. VISUALIZATION 2: HISTORICAL DROUGHT MATRIX
# =============================================================================

print("🌡️ Generating Historical SSI Matrix (2001-2025)...")
heatmap_df = master_crop[master_crop['district'].isin(TARGET_DISTRICTS)].copy()
matrix_data = heatmap_df.groupby(['crop_year', 'district'])['SSI'].mean().reset_index()
heatmap_pivot = matrix_data.pivot(index='crop_year', columns='district', values='SSI')
heatmap_pivot = heatmap_pivot[heatmap_pivot.index >= 2001]

plt.figure(figsize=(16, 10), dpi=300)
sns.heatmap(heatmap_pivot, cmap='RdBu', center=0, vmin=-3, vmax=3, linewidths=0.5, linecolor='white', cbar_kws={'label': 'Seasonal Mean SSI'})
plt.title("Southern Province: Historical Drought Matrix (Nov-Apr)", fontsize=16, fontweight='bold')
plt.savefig(os.path.join(OUTPUT_PATH, "Historical_SSI_Matrix.png"), bbox_inches='tight')

print(f"\n✨ DONE! Results saved in: {OUTPUT_PATH}")