#!/usr/bin/env python
# coding: utf-8

# # DISSERTATION: A Spatio-Temporal Reconstruction of Extreme Weather Events
# ##### PHASE: Data Processing, Hazard Identification, and Financial Calibration

# In[21]:


import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import json
import os


# In[22]:


# STEP 1: DATA LOADING & CLEANING (Aim 1: Building the Pipeline)
# -------------------------------------------------------------------------
print("📂 Loading Master Rainfall Database...")
# Load the 40-year historical extraction from GEE
df = pd.read_csv('C:/Users/FlawiyaShirishMore/OneDrive - Africa Specialty Risks Ltd/ASR-Parametric_Research_Study/africa_risk/master_rainfall_africa.csv')
districts_shp = gpd.read_file('C:/Users/FlawiyaShirishMore/OneDrive - Africa Specialty Risks Ltd/ASR-Parametric_Research_Study/africa_risk/data/africa_admx.shp')
# Standardize names to uppercase and strip whitespace to prevent join failures
df['ADM_NAME'] = df['ADM_NAME'].astype(str).str.upper().str.strip()
districts_shp['ADM_NAME'] = districts_shp['ADM_NAME'].astype(str).str.upper().str.strip()

# Handle missing rainfall values
df['mean'] = pd.to_numeric(df['mean'], errors='coerce')
df = df.dropna(subset=['mean'])


# In[23]:


# STEP 2: SCIENTIFIC METRICS (Aim 2: Proxy Metric Development)
# -------------------------------------------------------------------------
print("📉 Calculating WMO Percent of Normal Index (PNI)...")

# Calculate localized 40-year baseline
df['long_term_avg'] = df.groupby(['ISO3', 'ADM_NAME'])['mean'].transform('mean')
df['pni'] = (df['mean'] / df['long_term_avg']) * 100

# WMO Drought Severity Classification
def classify_wmo(pni):
    if pni < 55: return 3   # Extreme
    if pni < 70: return 2   # Severe
    if pni < 80: return 1   # Moderate
    return 0                # Normal

df['drought_severity'] = df['pni'].apply(classify_wmo)


# In[24]:


# STEP 3: PARAMETRIC CALIBRATION (Weather to Money)
# -------------------------------------------------------------------------
# Strike (Payout starts) at 70% PNI | Exit (100% Payout) at 50% PNI
STRIKE, EXIT = 70, 50

def calculate_payout(pni):
    if pni >= STRIKE: return 0.0
    if pni <= EXIT: return 1.0
    return (STRIKE - pni) / (STRIKE - EXIT)

df['proxy_loss_ratio'] = df['pni'].apply(calculate_payout)


# In[25]:


# STEP 4: PORTFOLIO CORRELATION (Aim 3: Diversification)
# -------------------------------------------------------------------------
print("🔗 Running Correlation Analysis...")
country_annual = df.groupby(['year', 'ISO3'])['pni'].mean().reset_index()
corr_matrix = country_annual.pivot(index='year', columns='ISO3', values='pni').corr()


# In[26]:


import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import json
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------------------------
# STEP 1: DATA LOADING & HARMONIZATION (Aim 1)
# -------------------------------------------------------------------------
print("📂 Loading and Harmonizing Data...")
df = pd.read_csv('master_rainfall_africa.csv')
districts_shp = gpd.read_file('africa_admx_shp/africa_admx.shp')

# Standardize to fix join failures ("White Holes")
df['ADM_NAME'] = df['ADM_NAME'].astype(str).str.upper().str.strip()
df['ISO3'] = df['ISO3'].astype(str).str.upper().str.strip()
districts_shp['ADM_NAME'] = districts_shp['ADM_NAME'].astype(str).str.upper().str.strip()
districts_shp['ISO3'] = districts_shp['ISO3'].astype(str).str.upper().str.strip()

# Numeric conversion
df['mean'] = pd.to_numeric(df['mean'], errors='coerce')
df = df.dropna(subset=['mean', 'ADM_NAME'])

# -------------------------------------------------------------------------
# STEP 2: CALCULATE SCIENTIFIC METRICS (Aim 2)
# -------------------------------------------------------------------------
print("📉 Calculating Drought Indices (PNI)...")
# Calculate localized 40-year baseline using transform
df['long_term_avg'] = df.groupby(['ISO3', 'ADM_NAME'])['mean'].transform('mean')
df['pni'] = (df['mean'] / df['long_term_avg']) * 100

# Classification (WMO Reference)
def classify_wmo(pni):
    if pni < 55: return 3   # Extreme
    if pni < 70: return 2   # Severe
    if pni < 80: return 1   # Moderate
    return 0                # Normal
df['drought_severity'] = df['pni'].apply(classify_wmo)

# -------------------------------------------------------------------------
# STEP 3: FINANCIAL CALIBRATION (Aim 2: Proxy Loss Ratio)
# -------------------------------------------------------------------------
# Strike at 70% PNI, Exit at 50% PNI
def calculate_payout(pni):
    if pni >= 70: return 0.0
    if pni <= 50: return 1.0
    return (70 - pni) / (70 - 50) # Linear interpolation

df['proxy_loss_ratio'] = df['pni'].apply(calculate_payout)

# -------------------------------------------------------------------------
# STEP 4: VISUAL C (FIXED) - The Parametric Underwriting Story
# -------------------------------------------------------------------------
print("📊 Generating Visual C: Parametric Calibration...")

# AUTO-FIND A DISTRICT: Instead of hardcoding 'KAJIADO', 
# find the district that has the highest drought severity to show a good example.
sample_district_name = df.sort_values(by='pni').iloc[0]['ADM_NAME']
sample_dist = df[df['ADM_NAME'] == sample_district_name].sort_values('year')

fig, ax1 = plt.subplots(figsize=(14, 7))

# Primary Axis: Rainfall
sns.barplot(data=sample_dist, x='year', y='pni', palette="Blues_d", ax=ax1, alpha=0.7)
ax1.set_ylabel('Rainfall % of Normal (PNI)', fontsize=12, fontweight='bold')
ax1.axhline(70, ls='--', color='red', label='Strike (70% PNI)')

# Secondary Axis: Payout
ax2 = ax1.twinx()
# Plotting using integer positions of the bars to ensure alignment
x_coords = np.arange(len(sample_dist))
ax2.plot(x_coords, sample_dist['proxy_loss_ratio'], color='red', marker='o', linewidth=3, label='Payout Ratio')

ax2.set_ylabel('Proxy Loss Ratio (Payout Intensity)', color='red', fontsize=12, fontweight='bold')
ax2.set_ylim(-0.05, 1.1)

plt.title(f"Parametric Underwriting Logic: {sample_district_name} District", fontsize=16)
ax1.tick_params(axis='x', rotation=45)
plt.savefig("Visual_C_Calibration.png", dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------------------------------------------------
# STEP 5: VISUAL A - Systemic Risk Reconstruction
# -------------------------------------------------------------------------
print("📊 Generating Visual A: Systemic Stress...")
# Aggregate by year
area_data = df.groupby(['year', 'drought_severity']).size().unstack(fill_value=0)
# Map numbers to names for the legend
area_data = area_data.rename(columns={0:'Normal', 1:'Moderate', 2:'Severe', 3:'Extreme'})

plt.figure(figsize=(12, 6))
area_data.plot(kind='area', stacked=True, colormap='YlOrRd', alpha=0.8, ax=plt.gca())
plt.title("40-Year African Drought Stress Reconstruction (1981-2024)", fontsize=14)
plt.ylabel("Number of Districts Affected")
plt.savefig("Visual_A_Drought_Stack.png", dpi=300)
plt.show()

# -------------------------------------------------------------------------
# STEP 6: VISUAL B - Correlation (Diversification)
# -------------------------------------------------------------------------
print("🔗 Generating Visual B: Correlation Heatmap...")
# Country level annual PNI
country_annual = df.groupby(['year', 'ISO3'])['pni'].mean().reset_index()
pivot_corr = country_annual.pivot(index='year', columns='ISO3', values='pni')

# Select key African economies/ASR targets
top_countries = ['NGA', 'KEN', 'ZAF', 'ETH', 'EGY', 'GHA', 'MAR', 'AGO', 'ZMB']
corr_matrix = pivot_corr[top_countries].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Inter-Country Drought Correlation (Portfolio Risk)", fontsize=14)
plt.savefig("Visual_B_Correlation.png", dpi=300)
plt.show()

# -------------------------------------------------------------------------
# STEP 7: INTERACTIVE PULSE (The "Showstopper")
# -------------------------------------------------------------------------
print("🗺️ Creating Interactive HTML Pulse Map...")
# Simplify geometry to make the file size smaller for the browser
districts_shp['geometry'] = districts_shp.simplify(0.02)
africa_json = json.loads(districts_shp.to_json())

# Prepare annual data
df_yearly = df.groupby(['year', 'ADM_NAME', 'ISO3'])['pni'].mean().reset_index()

fig = px.choropleth(
    df_yearly,
    geojson=africa_json,
    locations='ADM_NAME',
    featureidkey="properties.ADM_NAME",
    color='pni',
    animation_frame='year',
    color_continuous_scale="RdYlGn",
    range_color=[40, 120],
    title="Spatio-Temporal Reconstruction of Africa's Drought Pulse (1981-2024)",
    labels={'pni': 'Rainfall % of Normal'},
    height=800
)
fig.update_geos(fitbounds="locations", visible=False)
fig.write_html("Drought_Pulse_Reconstruction.html")

print("✅ DONE. All files generated successfully.")


# In[27]:


# STEP 6: PRESENTATION VISUALS (For ASR Stakeholders)
# -------------------------------------------------------------------------

# VISUAL A: The Drought Stack (Systemic Risk Reconstruction)
plt.figure(figsize=(12, 6))
yearly_counts = df.groupby(['year', 'drought_severity']).size().unstack(fill_value=0)
yearly_counts.columns = ['Normal', 'Moderate', 'Severe', 'Extreme']
yearly_counts.plot(kind='area', stacked=True, colormap='YlOrRd', alpha=0.8)
plt.title("Spatio-Temporal Reconstruction: African Districts under Drought Stress (1981-2024)")
plt.ylabel("Number of Administrative Districts")
plt.savefig("Presentation_Drought_Stack.png")


# In[28]:


# VISUAL B: The Diversification Heatmap
plt.figure(figsize=(10, 8))
# Selecting a balanced sample of African regions
sample_countries = ['EGY', 'ETH', 'KEN', 'NGA', 'ZAF', 'ZMB', 'SEN', 'MAR']
sns.heatmap(corr_matrix.loc[sample_countries, sample_countries], annot=True, cmap='coolwarm')
plt.title("Aim 3: Inter-Regional Correlation (Diversification Potential)")
plt.savefig("Presentation_Correlation_Heatmap.png")


# In[ ]:


# -------------------------------------------------------------------------
# VISUAL C: Rainfall vs Payout (The Parametric Mechanism)
# -------------------------------------------------------------------------
plt.figure(figsize=(10, 6))

# 1. Filter and sort data (Ensure Kajiado is uppercase if you ran the normalization step)
# If you didn't normalize, use 'Kajiado'. If you did, use 'KAJIADO'.
target_name = 'KAJIADO' if 'KAJIADO' in df['ADM_NAME'].values else 'Kajiado'
sample_dist = df[df['ADM_NAME'] == target_name].sort_values('year')

# 2. Create the base plot (Rainfall)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=sample_dist, x='year', y='pni', color='skyblue', ax=ax)
ax.set_ylabel("Rainfall PNI (%)", fontsize=12, color='blue')
ax.tick_params(axis='y', labelcolor='blue')

# 3. Create the twin axis for the Payout Ratio
ax2 = ax.twinx()

# FIX: Use the column name 'year' so Seaborn knows to look inside 'sample_dist'
# We use markers=True to ensure the dots appear on the line
sns.lineplot(data=sample_dist, x='year', y='proxy_loss_ratio', ax=ax2, 
             color='red', marker='o', linewidth=2.5)

ax2.set_ylabel("Proxy Loss Ratio (Payout Intensity)", fontsize=12, color='red')
ax2.set_ylim(-0.05, 1.05) # Standardize scale for payout (0 to 1)
ax2.tick_params(axis='y', labelcolor='red')

plt.title(f"Parametric Mechanism: Rainfall Deficit vs. Proxy Loss ({target_name})")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Presentation_Calibration.png", dpi=300)
plt.show()


# In[ ]:


# STEP 7: GEOSPATIAL ANIMATION (The "Drought Pulse")
# -------------------------------------------------------------------------
print("🗺️ Preparing Animated Pulse...")

# Load and simplify geometry
districts_shp = gpd.read_file('africa_admx_shp/africa_admx.shp')
districts_shp['geometry'] = districts_shp.simplify(0.05)
africa_json = json.loads(districts_shp.to_json())

# Animate PNI by year
fig = px.choropleth(
    df.groupby(['ISO3', 'ADM_NAME', 'year'])['pni'].mean().reset_index(),
    geojson=africa_json,
    locations='ADM_NAME',
    featureidkey="properties.ADM_NAME",
    color='pni',
    animation_frame='year',
    color_continuous_scale="RdYlGn",
    range_color=[40, 120],
    title="Spatio-Temporal Reconstruction of Africa's Drought Pulse",
    labels={'pni': 'Rainfall % of Normal'},
    height=800
)
fig.update_geos(fitbounds="locations", visible=False)
fig.write_html("Africa_Drought_Pulse_Reconstruction.html")

print("✅ ALL GOALS ACHIEVED.")
print("Files generated: Stack Plot, Heatmap, Calibration Plot, and Interactive HTML Pulse.")


# FAB EDA

# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --- 1. LOAD AND NORMALIZE ---
df = pd.read_csv('master_rainfall_africa.csv')
df.columns = df.columns.str.lower()
df['adm_name'] = df['adm_name'].astype(str).str.upper().str.strip()
df['iso3'] = df['iso3'].astype(str).str.upper().str.strip()

# --- 2. DATA QUALITY SUMMARY (Show this in your presentation!) ---
print("📋 DATA EXPLORATION SUMMARY")
stats_table = df.groupby('iso3')['mean'].describe()
print(stats_table.head(10)) # Top 10 countries overview

# =============================================================================
# PLOT 1: THE CONTINENTAL HEARTBEAT (25-Year Time Series)
# =============================================================================
plt.figure(figsize=(14, 6))
yearly_total = df.groupby('year')['mean'].mean()
rolling_avg = yearly_total.rolling(window=3).mean()

plt.plot(yearly_total.index, yearly_total.values, marker='o', color='navy', label='Annual Mean Rainfall')
plt.plot(rolling_avg.index, rolling_avg.values, color='red', linestyle='--', label='3-Year Moving Average')

# Annotate key historical events
events = {2011: 'East Africa Drought', 2015: 'El Niño Spike', 2024: 'Southern Africa Crisis'}
for yr, txt in events.items():
    if yr in yearly_total.index:
        plt.annotate(txt, xy=(yr, yearly_total[yr]), xytext=(yr, yearly_total[yr]+20),
                     arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)

plt.title("Africa's Rainfall Heartbeat: 2000-2025 Trend", fontsize=16)
plt.ylabel("Mean Rainfall (mm)")
plt.legend()
plt.savefig("EDA_1_Continental_Trend.png", dpi=300)
plt.show()

# =============================================================================
# PLOT 2: RAINFALL VARIABILITY (COEFFICIENT OF VARIATION)
# =============================================================================
# Formula: (Std Dev / Mean) * 100. High CV = High Unpredictability.
cv_data = df.groupby('iso3')['mean'].agg(['std', 'mean'])
cv_data['cv'] = (cv_data['std'] / cv_data['mean']) * 100
cv_data = cv_data.sort_values('cv', ascending=False).head(15)

plt.figure(figsize=(12, 6))
sns.barplot(x=cv_data.index, y=cv_data['cv'], palette='magma')
plt.title("Top 15 Countries with Highest Rainfall Unpredictability (CV%)", fontsize=14)
plt.ylabel("Coefficient of Variation (%)")
plt.xlabel("Country ISO Code")
plt.savefig("EDA_2_Variability_Risk.png", dpi=300)
plt.show()

# =============================================================================
# PLOT 3: MONTHLY CLIMATOLOGY (The "Rainy Season" Pulse)
# =============================================================================
plt.figure(figsize=(12, 6))
# Let's compare two different regions: e.g., Ethiopia (ETH) vs South Africa (ZAF)
regions = ['ETH', 'ZAF', 'NGA', 'KEN']
for r in regions:
    subset = df[df['iso3'] == r].groupby('month')['mean'].mean()
    plt.plot(subset.index, subset.values, marker='s', label=f'Climatology: {r}')

plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.title("Regional Seasonality Patterns: Identifying the 'Hungry Season'", fontsize=14)
plt.ylabel("Mean Monthly Rainfall (mm)")
plt.legend()
plt.savefig("EDA_3_Seasonality.png", dpi=300)
plt.show()

# =============================================================================
# PLOT 4: DISTRIBUTION SKEW (Extreme Events)
# =============================================================================
plt.figure(figsize=(12, 6))
sns.violinplot(data=df[df['iso3'].isin(regions)], x='iso3', y='mean', palette='Set3')
plt.title("Rainfall Distribution & Extreme Outliers by Representative Countries", fontsize=14)
plt.ylabel("Rainfall Amount (mm)")
plt.savefig("EDA_4_Distributions.png", dpi=300)
plt.show()


# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# --- 1. LOAD & HARMONIZE ---
print("🔗 Harmonizing Datasets...")
df = pd.read_csv('master_rainfall_africa.csv')
df.columns = df.columns.str.lower()
emdat = pd.read_csv('C:/Users/FlawiyaShirishMore/OneDrive - Africa Specialty Risks Ltd/ASR-Parametric_Research_Study/africa_risk/data/drought.csv') # Ensure your file name is correct
emdat.columns = emdat.columns.str.lower()

# Filter EM-DAT for Droughts and create a binary 'Disaster' flag
emdat_droughts = emdat[emdat['disaster subtype'] == 'Drought'].copy()
emdat_droughts['is_disaster'] = 1
# Aggregate EM-DAT to Country-Year level
ground_truth = emdat_droughts.groupby(['iso', 'start year'])['is_disaster'].first().reset_index()
ground_truth = ground_truth.rename(columns={'iso': 'iso3', 'start year': 'year'})

# =============================================================================
# PHASE 2.1: CALCULATING THE INDICES
# =============================================================================
print("🧮 Calculating PNI, SPI, and Quantiles...")

# We aggregate to country level to match EM-DAT reporting scale
country_annual = df.groupby(['iso3', 'year'])['mean'].mean().reset_index()

# 1. PNI: (Rain / Mean) * 100
ltm = country_annual.groupby('iso3')['mean'].transform('mean')
country_annual['pni'] = (country_annual['mean'] / ltm) * 100

# 2. SPI (Simplified Z-Score Approximation)
# SPI = (Rain - Mean) / StdDev
std_dev = country_annual.groupby('iso3')['mean'].transform('std')
country_annual['spi'] = (country_annual['mean'] - ltm) / std_dev

# 3. Quantile Rank (0 to 1)
# 0.05 means this year is in the driest 5% of history
country_annual['quantile'] = country_annual.groupby('iso3')['mean'].rank(pct=True)

# Merge with Ground Truth (EM-DAT)
comparison_df = pd.merge(country_annual, ground_truth, on=['iso3', 'year'], how='left')
comparison_df['is_disaster'] = comparison_df['is_disaster'].fillna(0)

# =============================================================================
# PHASE 2.2: ROC CURVE ANALYSIS (The Accuracy Test)
# =============================================================================
print("📉 Generating ROC Accuracy Curve...")

plt.figure(figsize=(10, 7))

# For ROC, we use the "Drought Strength" (Inverse of the values)
# Because lower PNI/SPI/Quantile means Higher Drought
methods = {
    'PNI (Percent of Normal)': -comparison_df['pni'],
    'SPI (Standardized Index)': -comparison_df['spi'],
    'Quantile (Rank-based)': -comparison_df['quantile']
}

auc_scores = {}

for name, score in methods.items():
    fpr, tpr, _ = roc_curve(comparison_df['is_disaster'], score)
    roc_auc = auc(fpr, tpr)
    auc_scores[name] = roc_auc
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title("Comparative Accuracy: Which Index Best Predicts EM-DAT Disasters?", fontsize=14)
plt.xlabel("False Positive Rate (False Alarms)")
plt.ylabel("True Positive Rate (Correct Detections)")
plt.legend(loc="lower right")
plt.grid(alpha=0.2)
plt.savefig("Validation_ROC_Curve.png", dpi=300)
plt.show()

# =============================================================================
# PHASE 2.3: BASIS RISK & LOSS RATIO (Financial Suitability)
# =============================================================================
print("💰 Calculating Suitability & Basis Risk...")

# Define "Drought Events" for each index based on standard thresholds
comparison_df['trigger_pni'] = (comparison_df['pni'] < 75).astype(int)      # WMO Moderate
comparison_df['trigger_spi'] = (comparison_df['spi'] < -1.0).astype(int)   # Standard SPI
comparison_df['trigger_quant'] = (comparison_df['quantile'] < 0.2).astype(int) # 1-in-5 year event

# Calculate Basis Risk (When model triggers but no disaster, or vice versa)
summary = []
for m in ['pni', 'spi', 'quant']:
    trigger_col = f'trigger_{m}'
    # False Negative (Missed Disaster): Disaster happened (1) but Model said No (0)
    missed = comparison_df[(comparison_df['is_disaster'] == 1) & (comparison_df[trigger_col] == 0)].shape[0]
    # False Positive (False Alarm): Model said Yes (1) but no Disaster (0)
    false_alarm = comparison_df[(comparison_df['is_disaster'] == 0) & (comparison_df[trigger_col] == 1)].shape[0]
    
    summary.append({'Method': m.upper(), 'Missed Disasters': missed, 'False Alarms': false_alarm, 'AUC': auc_scores.get(m, 0)})

summary_df = pd.DataFrame(summary)
print("\n--- FINAL COMPARISON TABLE ---")
print(summary_df)


# In[36]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# --- STEP 1: LOAD & HARMONIZE ---
df = pd.read_csv('master_rainfall_africa.csv')
df.columns = df.columns.str.lower()

# Correct Path for EM-DAT
emdat_path = 'C:/Users/FlawiyaShirishMore/OneDrive - Africa Specialty Risks Ltd/ASR-Parametric_Research_Study/africa_risk/data/drought.csv'
emdat = pd.read_csv(emdat_path)
emdat.columns = emdat.columns.str.lower()

# Prepare EM-DAT (Ground Truth) - CONVERTING TO BINARY
emdat_droughts = emdat[emdat['disaster subtype'] == 'Drought'].copy()
# We use .size() then clip(max=1) to ensure it is 0 or 1 (Binary)
emdat_events = emdat_droughts.groupby(['iso', 'start year']).size().reset_index(name='disaster')
emdat_events['disaster'] = emdat_events['disaster'].clip(upper=1) # THE FIX: 1 or 0 only
emdat_events = emdat_events.rename(columns={'iso': 'iso3', 'start year': 'year'})

# =============================================================================
# STEP 2: CALCULATE "INSURANCE-GRADE" INDICATORS (SPI-3 & SEASONAL)
# =============================================================================
# Reference: WMO (2016) Handbook of Drought Indicators
# Reference: African Risk Capacity (ARC) - Seasonal Monitoring Logic
print("🌾 Calculating Agricultural Drought Indices (SPI-3)...")

# 2.1 3-Month Rolling Rainfall
df = df.sort_values(['iso3', 'adm_name', 'year', 'month'])
df['rain_3m'] = df.groupby(['iso3', 'adm_name'])['mean'].transform(lambda x: x.rolling(window=3).sum())

# 2.2 Calculate SPI-3 (Z-Score of the 3-month window)
def calc_spi(group):
    mu = group.mean()
    sigma = group.std()
    return (group - mu) / (sigma if sigma > 0 else 1)

df['spi_3'] = df.groupby(['iso3', 'adm_name', 'month'])['rain_3m'].transform(calc_spi)

# 2.3 The "Insurance Trigger": Pick the WORST SPI-3 of the year
annual_model = df.groupby(['iso3', 'year'])['spi_3'].min().reset_index()

# 2.4 Seasonal PNI (Worst month's PNI relative to its own climatology)
df['pni_seasonal'] = (df['mean'] / df.groupby(['iso3', 'adm_name', 'month'])['mean'].transform('mean')) * 100
annual_model['pni_min'] = df.groupby(['iso3', 'year'])['pni_seasonal'].min().reset_index()['pni_seasonal']

# Merge with EM-DAT
eval_df = pd.merge(annual_model, emdat_events, on=['iso3', 'year'], how='left').fillna(0)

# =============================================================================
# STEP 3: THE IMPROVED ROC EVALUATION
# =============================================================================
plt.figure(figsize=(10, 8))

# Methodology Reference: Dutra et al. (2013) "Drought forecasting performance in Africa"
indices = {
    'Agricultural Standard: Min SPI-3': -eval_df['spi_3'],
    'Seasonal PNI (Min Month)': -eval_df['pni_min'],
}

for label, scores in indices.items():
    # Clean scores
    scores = scores.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Ensure y_true is strictly binary int
    y_true = eval_df['disaster'].astype(int)
    
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=3, label=f'{label} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random Chance (AUC = 0.50)')
plt.title("Comparative Accuracy: SPI-3 vs. PNI vs. EM-DAT Records", fontsize=14)
plt.xlabel("False Positive Rate (Incorrect Alarms)")
plt.ylabel("True Positive Rate (Detected Disasters)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig("Final_ROC_Validation.png", dpi=300)
plt.show()

# =============================================================================
# STEP 4: PARAMETRIC LOSS RATIO & CLASSIFICATION
# =============================================================================
# Based on African Risk Capacity (ARC) and ASR Professional Design
def classify_asr(spi):
    if spi <= -2.0: return "Extreme (Payout 100%)"
    if spi <= -1.5: return "Severe (Payout 70%)"
    if spi <= -1.0: return "Moderate (Payout 20%)"
    return "Normal (No Payout)"

eval_df['payout_status'] = eval_df['spi_3'].apply(classify_asr)

# Display results for presentation
print("\n--- FINAL CLASSIFICATION RESULTS (ASR Standard) ---")
print(eval_df[['iso3', 'year', 'spi_3', 'payout_status']].sort_values('spi_3').head(15))


# machine learning model:

# In[38]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, classification_report

# --- STEP 1: FEATURE ENGINEERING ---
print("⚙️ Engineering ML Features (Persistence, SPI-6, Variability)...")

# Calculate SPI-3 and SPI-6
df = df.sort_values(['iso3', 'adm_name', 'year', 'month'])
df['rain_3m'] = df.groupby(['iso3', 'adm_name'])['mean'].transform(lambda x: x.rolling(3).sum())
df['rain_6m'] = df.groupby(['iso3', 'adm_name'])['mean'].transform(lambda x: x.rolling(6).sum())

def z_score(x):
    return (x - x.mean()) / (x.std() if x.std() > 0 else 1)

df['spi3'] = df.groupby(['iso3', 'adm_name', 'month'])['rain_3m'].transform(z_score)
df['spi6'] = df.groupby(['iso3', 'adm_name', 'month'])['rain_6m'].transform(z_score)

# Feature: Magnitude of Deficit (Cumulative % loss)
df['pni'] = (df['mean'] / df.groupby(['iso3', 'adm_name', 'month'])['mean'].transform('mean')) * 100
df['deficit'] = np.where(df['pni'] < 100, 100 - df['pni'], 0)
df['cum_deficit_6m'] = df.groupby(['iso3', 'adm_name'])['deficit'].transform(lambda x: x.rolling(6).sum())

# Feature: Persistence (Consecutive Dry Months)
df['is_dry'] = (df['pni'] < 80).astype(int)
df['dry_streak'] = df.groupby(['iso3', 'adm_name'])['is_dry'].transform(lambda x: x.rolling(6).sum())

# Aggregate to Annual Country-Level for EM-DAT
ml_data = df.groupby(['iso3', 'year']).agg({
    'spi3': 'min',
    'spi6': 'min',
    'pni': 'mean',
    'cum_deficit_6m': 'max',
    'dry_streak': 'max'
}).reset_index()

# Merge with Binary EM-DAT Disaster Flag
eval_df = pd.merge(ml_data, emdat_events, on=['iso3', 'year'], how='left').fillna(0)

# =============================================================================
# STEP 2: RANDOM FOREST MACHINE LEARNING
# =============================================================================
print("🤖 Training Random Forest Classifier...")

# Prepare Features (X) and Target (y)
features = ['spi3', 'spi6', 'pni', 'cum_deficit_6m', 'dry_streak']
X = eval_df[features].fillna(0)
y = eval_df['disaster'].astype(int)

# Split data (Train on 80%, Test on 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# Predict Probabilities
y_probs = rf.predict_proba(X_test)[:, 1]

# =============================================================================
# STEP 3: RESULTS & ROC
# =============================================================================
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'Random Forest ML (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title("Machine Learning Validation: Multi-Factor vs. EM-DAT Records", fontsize=14)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig("Machine Learning Validation: Multi-Factor vs. EM-DAT Records", dpi=300)
plt.show()

# Feature Importance (Tell the professor which factor mattered most!)
importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("\n--- Feature Importance (What drives the disaster?) ---")
print(importance)


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

# --- STEP 1: LOAD THE NEW 1981-2025 MULTIVARIATE DATA ---
# (Assume you have merged Rainfall, Temp, and PET into one CSV)
df = pd.read_csv('africa_multivariate_1981_2025.csv')
df.columns = df.columns.str.lower()

# --- STEP 2: CALCULATE SPEI (Water Balance) ---
# SPEI = Standardized (Precipitation - Potential Evapotranspiration)
df['water_balance'] = df['precip'] - df['pet']

# Function to calculate Standardized Index (Z-score) over the 45-year baseline
def standardize(x):
    return (x - x.mean()) / (x.std() if x.std() > 0 else 1)

# Group by District and Month to account for seasonality
df['spei_3'] = df.groupby(['adm_name', 'month'])['water_balance'].transform(lambda x: standardize(x.rolling(3).mean()))
df['temp_anomaly'] = df.groupby(['adm_name', 'month'])['temp'].transform(standardize)

# --- STEP 3: ML TRAINING (The Random Forest) ---
# We add the "Heat" and "Balance" factors
features = ['spei_3', 'temp_anomaly', 'precip_pni', 'dry_streak']
X = df[features].fillna(0)
y = df['emdat_disaster'] # Your binary target from EM-DAT

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = RandomForestClassifier(n_estimators=200, max_depth=15)
model.fit(X_train, y_train)

# --- STEP 4: THE FINAL VALIDATION ---
y_probs = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probs)
print(f"New Multivariate AUC: {auc(fpr, tpr):.2f}")

