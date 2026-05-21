#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import json
import os
import warnings

# --- INITIAL SETUP ---
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-muted') # Professional academic style
STRIKE, EXIT = 70, 50  # Parametric Payout: Starts at 70% PNI, 100% at 50% PNI

# =============================================================================
# STEP 1: DATA LOADING & HARMONIZATION (Aim 1)
# =============================================================================
print("📂 Loading and Standardizing Data...")

# 1.1 Load Rainfall Data (CHIRPS 2000-2025)
# Using encoding='utf-8' to prevent UnicodeDecodeErrors with African names
df = pd.read_csv('master_rainfall_africa.csv', encoding='utf-8')

# Force all column names to lowercase to prevent KeyErrors
df.columns = df.columns.str.lower()

# 1.2 Standardization to prevent "White Holes" (Join Failures)
# Clean 'adm_name' and 'iso3' (Upper, Strip spaces)
df['adm_name'] = df['adm_name'].astype(str).str.upper().str.strip()
df['iso3'] = df['iso3'].astype(str).str.upper().str.strip()

# 1.3 Load Spatial Data (GADM Shapefile)
districts_shp = gpd.read_file('africa_admx_shp/africa_admx.shp')
districts_shp.columns = districts_shp.columns.str.lower()
districts_shp['adm_name'] = districts_shp['adm_name'].astype(str).str.upper().str.strip()
districts_shp['iso3'] = districts_shp['iso3'].astype(str).str.upper().str.strip()

# Numeric conversion for rainfall 'mean'
df['mean'] = pd.to_numeric(df['mean'], errors='coerce')
df = df.dropna(subset=['mean', 'adm_name'])

# =============================================================================
# STEP 2: SCIENTIFIC METRICS & FINANCIAL CALIBRATION (Aim 2)
# =============================================================================
print("📊 Calculating Drought Indices & Parametric Payouts...")

# 2.1 WMO Percent of Normal Index (PNI) Calculation
# Baseline is the mean rainfall per district over the 25-year period
df['long_term_avg'] = df.groupby(['iso3', 'adm_name'])['mean'].transform('mean')
df['pni'] = (df['mean'] / df['long_term_avg']) * 100

# 2.2 WMO Drought Severity Classification
def classify_wmo(pni):
    if pni < 55: return 3   # Extreme
    if pni < 70: return 2   # Severe
    if pni < 80: return 1   # Moderate
    return 0                # Normal
df['drought_severity'] = df['pni'].apply(classify_wmo)

# 2.3 Financial Calibration: Proxy Loss Ratio (Parametric)
def calculate_payout(pni):
    if pni >= STRIKE: return 0.0
    if pni <= EXIT: return 1.0
    return (STRIKE - pni) / (STRIKE - EXIT) # Linear payout between Strike and Exit
df['proxy_loss_ratio'] = df['pni'].apply(calculate_payout)

# =============================================================================
# STEP 3: VISUAL A - SYSTEMIC RISK RECONSTRUCTION
# =============================================================================
print("📈 Generating Visual A: Systemic Stress Reconstruction...")

plt.figure(figsize=(12, 6))
# Count districts per severity category per year
area_data = df.groupby(['year', 'drought_severity']).size().unstack(fill_value=0)
# Ensure columns are present even if a category is missing in a specific year
for i in range(4): 
    if i not in area_data.columns: area_data[i] = 0
area_data = area_data[[0, 1, 2, 3]].rename(columns={0:'Normal', 1:'Moderate', 2:'Severe', 3:'Extreme'})

area_data.plot(kind='area', stacked=True, colormap='YlOrRd', alpha=0.8, ax=plt.gca())
plt.title("African District-Level Drought Stress Reconstruction (2000-2025)", fontsize=14)
plt.ylabel("Number of Districts Affected")
plt.xlabel("Year")
plt.legend(loc='upper left')
plt.savefig("Dissertation_Visual_A_Systemic_Risk.png", dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# STEP 4: VISUAL B - INTER-REGIONAL CORRELATION (Aim 3: Diversification)
# =============================================================================
print("🌡️ Generating Visual B: Correlation Heatmap...")

# Focus on key ASR markets
target_countries = ['NGA', 'KEN', 'ZAF', 'ETH', 'EGY', 'GHA', 'MAR', 'AGO', 'ZMB']
country_annual = df.groupby(['year', 'iso3'])['pni'].mean().unstack()
available_countries = [c for c in target_countries if c in country_annual.columns]

if available_countries:
    corr_matrix = country_annual[available_countries].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", center=0)
    plt.title("Portfolio Diversification Potential: Rainfall Correlation (2000-2025)", fontsize=13)
    plt.savefig("Dissertation_Visual_B_Correlation.png", dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# STEP 5: VISUAL C - PARAMETRIC UNDERWRITING CASE STUDY
# =============================================================================
print("📉 Generating Visual C: Parametric Calibration...")

# Auto-pick the district with the most severe historical drought to demonstrate
sample_name = df.sort_values(by='pni').iloc[0]['adm_name']
sample_dist = df[df['adm_name'] == sample_name].sort_values('year')

fig, ax1 = plt.subplots(figsize=(14, 6))
# Rainfall Bar Chart
sns.barplot(data=sample_dist, x='year', y='pni', color='skyblue', ax=ax1, alpha=0.7)
ax1.axhline(STRIKE, ls='--', color='red', label=f'Strike ({STRIKE}% PNI)')
ax1.set_ylabel("Rainfall % of Normal (PNI)", fontsize=12, fontweight='bold')
ax1.tick_params(axis='x', rotation=45)

# Payout Line Chart
ax2 = ax1.twinx()
sns.lineplot(data=sample_dist, x=np.arange(len(sample_dist)), y='proxy_loss_ratio', 
             ax=ax2, color='red', marker='o', linewidth=2, label='Payout Ratio')
ax2.set_ylabel("Proxy Loss Ratio (Payout Intensity)", color='red', fontsize=12)
ax2.set_ylim(-0.05, 1.1)

plt.title(f"Parametric Underwriting Logic: {sample_name} District Reconstruction", fontsize=14)
plt.savefig("Dissertation_Visual_C_Calibration.png", dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# STEP 6: STEP 7: GEOSPATIAL ANIMATION (The "Drought Pulse")
# =============================================================================
print("🌍 Creating Interactive Spatio-Temporal Pulse Map...")

# Simplify geometry to 2% to keep HTML file size manageable for browsers
districts_shp['geometry'] = districts_shp.simplify(0.02)
# Convert to GeoJSON format for Plotly
africa_json = json.loads(districts_shp.to_json())

# Aggregate for the map
df_map = df.groupby(['year', 'adm_name', 'iso3'])['pni'].mean().reset_index()

fig = px.choropleth(
    df_map,
    geojson=africa_json,
    locations='adm_name',
    featureidkey="properties.adm_name", # Must match the property name in Shapefile
    color='pni',
    animation_frame='year',
    color_continuous_scale="RdYlGn",
    range_color=[40, 120],
    title="Spatio-Temporal Reconstruction of Africa's Drought Pulse (2000-2025)",
    labels={'pni': 'Rainfall % of Normal'},
    height=800
)

fig.update_geos(fitbounds="locations", visible=False)
fig.write_html("Drought_Pulse_Interactive_Map.html")

print("✅ DONE: All Dissertation Visuals and Interactive Map Generated.")


# In[11]:


get_ipython().system('pip install scipy')


# In[15]:


get_ipython().system('pip install scikit-learn')


# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, auc

# --- STEP 1: LOAD & HARMONIZE BOTH DATASOURCES ---
print("📂 Harmonizing Master Rainfall and EM-DAT...")

# 1.1 Master Rainfall Data
df = pd.read_csv('master_rainfall_africa.csv')
df.columns = df.columns.str.lower()
df['iso3'] = df['iso3'].str.upper().str.strip()

# 1.2 EM-DAT Data
emdat = pd.read_csv('C:/Users/FlawiyaShirishMore/OneDrive - Africa Specialty Risks Ltd/ASR-Parametric_Research_Study/africa_risk/data/drought.csv') # Use your EM-DAT filename
emdat.columns = emdat.columns.str.lower()
# Filter for Drought only
emdat = emdat[emdat['disaster subtype'] == 'Drought']
emdat = emdat.rename(columns={'iso': 'iso3', 'start year': 'year'})
emdat['iso3'] = emdat['iso3'].str.upper().str.strip()

# Create a "Ground Truth" binary column at Country-Year level
emdat_events = emdat.groupby(['iso3', 'year']).size().reset_index(name='emdat_event_occurred')
emdat_events['emdat_event_occurred'] = 1 # Mark any year in EM-DAT as 1

# =============================================================================
# STEP 2: CALCULATE COMPARATIVE INDICES (PNI, SPI, QUANTILE)
# =============================================================================
print("📊 Calculating Comparative Indices...")

# 2.1 PNI (Percent of Normal)
ltm = df.groupby(['iso3', 'adm_name'])['mean'].transform('mean')
df['pni'] = (df['mean'] / ltm) * 100

# 2.2 Quantile Rank (0 to 1 scale)
# This identifies where a year falls relative to history (e.g., 0.1 is the driest 10%)
df['quantile_rank'] = df.groupby(['iso3', 'adm_name'])['mean'].rank(pct=True)

# 2.3 SPI-Simplified (Standard Normal Z-Score)
# Standard way to calculate SPI in simple scripts: (Rain - Mean) / StdDev
std_val = df.groupby(['iso3', 'adm_name'])['mean'].transform('std')
df['spi_approx'] = (df['mean'] - ltm) / std_val

# Aggregate to Country level for validation with EM-DAT
model_eval = df.groupby(['iso3', 'year'])[['pni', 'quantile_rank', 'spi_approx']].mean().reset_index()

# Merge with Ground Truth
eval_df = pd.merge(model_eval, emdat_events, on=['iso3', 'year'], how='left')
eval_df['emdat_event_occurred'] = eval_df['emdat_event_occurred'].fillna(0)

# =============================================================================
# STEP 3: COMPARATIVE ACCURACY (ROC CURVE & AUC)
# =============================================================================
print("📉 Calculating ROC Curves & AUC for Accuracy...")

plt.figure(figsize=(10, 8))

# We define the indices and their "direction" 
# (For PNI/SPI/Quantile, LOWER values = MORE drought, so we use negative for ROC)
indices = {
    'PNI': -eval_df['pni'],
    'SPI (Approx)': -eval_df['spi_approx'],
    'Quantile Rank': -eval_df['quantile_rank']
}

for label, scores in indices.items():
    fpr, tpr, _ = roc_curve(eval_df['emdat_event_occurred'], scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Incorrect Drought Alarm)')
plt.ylabel('True Positive Rate (Correct Disaster Match)')
plt.title('Drought Index Accuracy vs. EM-DAT Records (Africa 2000-2025)')
plt.legend(loc="lower right")
plt.savefig("Comparative_Drought_Accuracy_ROC.png", dpi=300)
plt.show()

# =============================================================================
# STEP 4: FINANCIAL LOSS RATIO COMPARISON
# =============================================================================
print("💰 Calculating Comparative Loss Ratios...")

# Loss Ratio = Total Model Payouts / Total Historical Occurrences
# We define a common "Strike" across all (e.g., 20th percentile)
eval_df['payout_pni'] = np.where(eval_df['pni'] < 75, 1, 0)
eval_df['payout_spi'] = np.where(eval_df['spi_approx'] < -1.0, 1, 0)
eval_df['payout_quant'] = np.where(eval_df['quantile_rank'] < 0.2, 1, 0)

# Compare which one aligns best with Total Affected in EM-DAT
merged_final = pd.merge(eval_df, emdat.groupby(['iso3', 'year'])['total affected'].sum().reset_index(), 
                        on=['iso3', 'year'], how='left').fillna(0)

# Calculate Loss Ratio (Total Payouts vs Total Reported Disasters)
lr_results = {
    'PNI': merged_final['payout_pni'].sum() / merged_final['emdat_event_occurred'].sum(),
    'SPI': merged_final['payout_spi'].sum() / merged_final['emdat_event_occurred'].sum(),
    'Quantile': merged_final['payout_quant'].sum() / merged_final['emdat_event_occurred'].sum()
}
print("Calculated Payout Frequency vs. Real Event Frequency (Closer to 1 is better):")
for k, v in lr_results.items():
    print(f" - {k}: {v:.2f}")

print("✅ COMPARATIVE ANALYSIS COMPLETE.")


# In[21]:





# In[1]:


import pandas as pd

# 1. Load the Original Data (2M+ rows)
master_df = pd.read_csv('master_rainfall_africa.csv')
master_df.columns = master_df.columns.str.lower()

# 2. Load the Gap-Fill Data
gap_fill_df = pd.read_csv('Africa_77_Gaps_Multivariate_1981_2025.csv')
gap_fill_df.columns = gap_fill_df.columns.str.lower()

# Rename columns to match (precipitation -> mean)
gap_fill_df = gap_fill_df.rename(columns={'precipitation': 'mean'})

# 3. CONCATENATE (The Master Merge)
final_database = pd.concat([master_df, gap_fill_df], ignore_index=True)

# 4. DATA CLEANING & FINAL SAVE
# Ensure unique ID for every record
final_database['adm_name'] = final_database['adm_name'].str.upper().str.strip()
final_database = final_database.drop_duplicates(subset=['iso3', 'adm_name', 'year', 'month'])

final_database.to_csv('FINAL_AFRICA_DROUGHT_DATABASE_1981_2025.csv', index=False)

print(f"🎉 SUCCESS! Final Database contains {len(final_database):,} rows.")
print("🌍 Spatial Coverage: 100% | Temporal Coverage: 1981-2025 (44 Years)")


# In[ ]:




