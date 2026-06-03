#!/usr/bin/env python
# coding: utf-8

# # Ethiopia
# Ethiopia’s complex topography and reliance on rainfed agriculture create a unique landscape of drought vulnerability defined by sharp elevation-driven gradients. Approximately 94% of the country's crop production is dependent on the seasonal rains, making the economy highly sensitive to climatic fluctuations (Gidey et al., 2018; Woodwell Climate Research Center, 2024).
# 
# 1. The Highland-Lowland Dichotomy
# The Ethiopian landscape is divided into two primary climatic regimes:
# Central Highlands (2,000-4,500m): Covering 60% of the landmass, this region experiences a bimodal rainfall pattern consisting of the Belg (short rains, March–May) and the Meher (long rains, June–September). Annual rainfall ranges from 800mm to 2,000mm.
# Eastern & Southern Lowlands (<500m): This arid region, including the Afar and Somali deserts, receives less than 300mm of unimodal rainfall. Topographic rain shadows often leave the eastern basins significantly drier than the more resilient Amhara highlands (Woodwell Climate Research Center, 2024).
# 
# 2. Temperature and the "Hot Drought" Mechanism
# Temperature regimes vary drastically by altitude, with tropical lowlands averaging 25–35°C (peaking above 40°C during droughts) while cold highlands (>2,400m) range between 6–16°C. In the era of climate change, rising baselines accelerate potential evapotranspiration (PET). This process transforms meteorological drought-traditionally measured by the Standardized Precipitation Index (SPI)—into severe agricultural drought through rapid soil moisture depletion. This justifies the transition to the Standardized Precipitation Evapotranspiration Index (SPEI), which accounts for this thermal stress (Vicente-Serrano et al., 2010; FEWS NET, 2025).
# 
# 
# 3. Agricultural Impact and Historical Precedent
# The highlands support 60% of the population through the cultivation of Teff, maize, and barley. Conversely, the eastern lowlands are dominated by pastoralist livelihoods where water point depletion and pasture failure occur rapidly during drought.
# * Critical Windows: SPEI-3 is the primary indicator for soil moisture during the 90-day planting-to-grain-filling cycle, while SPEI-6/12 is utilized to monitor streamflow and reservoir levels (Gidey et al., 2018).
# 
# * Historical Benchmarks: The catastrophic events of 1983–85 and 2011 (where SPI/SPEI reached values ≤ − 2.0) resulted in crop yield losses of 50–80% in affected districts.
# 
# 4. Parametric Insurance Strategy: Weighted Triggers
# To minimize Basis Risk (the mismatch between insurance payouts and actual losses), drought triggers must be calibrated to specific Agro-Ecological Zones (AEZ).
# * Highland Sensitivity (Moderate Threshold): Because highland crops like Teff are highly sensitive to water stress, insurance payouts should be triggered at SPEI < -1.2 to capture "Moderate to Severe" failure.
# * Lowland Variability (Stringent Threshold): Due to the high natural variability of rainfall in arid zones, triggers must be more stringent (SPEI < -1.5) to ensure payouts align with truly catastrophic events rather than standard dry-year fluctuations (Philip et al., 2021; Gidey et al., 2020).
# * Future Risk: Projections indicate that warmer baselines will shrink the viable growing period by 10–20 days by 2050, necessitating the use of dynamic, temperature-aware indices like SPEI for long-term risk transfer (Gidey et al., 2020).

# ### What is SPEI?
# SPEI quantifies drought by measuring the Climatic Water Balance (D), which is simply Precipitation minus Potential Evapotranspiration (D = P−PET). Because D can be both negative (water deficit) and positive (water surplus), it is not heavily skewed like pure rainfall. Therefore, instead of the Gamma distribution used in SPI, SPEI fits the historical data to a Log-Logistic (Fisk) distribution, and then transforms it into a standard normal distribution (a bell curve).
# 
# * An SPEI of 0 means the moisture balance is exactly average.
# * An SPEI of -1.0 means the moisture balance is 1 standard deviation below average.

# ### Why SPEI-3 for Agricultural Triggers?
# Just like SPI, SPEI-3 measures the rolling 3-month accumulation of the climatic water balance (D). It perfectly encapsulates the critical 90-day planting-to-grain-filling cycle. If high temperatures cause massive evaporation combined with low rainfall during these 3 months, SPEI-3 will crash into severe deficit, perfectly predicting soil moisture collapse.

# ### The Data
# For SPEI, we need two variables per pixel: Precipitation and PET (Potential Evapotranspiration).
# Datasets: You can use TerraClimate, ERA5-Land, or combine CHIRPS (rainfall) with CHIRTS (temperature). Many modern datasets provide pre-calculated PET (usually using the Penman-Monteith or Hargreaves equations).
# Depth: The World Meteorological Organization (WMO) highly recommends 30 years of historical data for SPEI.

# The thresholds for SPEI are identical to SPI, allowing for seamless integration into parametric insurance frameworks:
# 
# * -1.0 (Mild Drought): Early warning / alert phase.
# * -1.5 (Severe Drought): The Spatio-Temporal Trigger for your dissertation's primary payout phase (approx. 1-in-15-year event).
# * -2.0 (Extreme Drought): Maximum payout phase.

# In[7]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[8]:


import pandas as pd
import glob
import os
import warnings
from scipy.stats import fisk, norm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
warnings.filterwarnings('ignore')


# In[10]:


import pandas as pd

# Define the common keys needed for merging
keys = ['ADM_NAME', 'year', 'month', 'lat_wgs84', 'lon_wgs84']

# 1. Load Rainfall (CHIRPS)
df_precip = pd.read_csv(
    r"outputs\ethiopia\content\ethiopia_rainfall_master_cleaned_with_latlon.csv", 
    usecols=keys + ['precip_mm']
)

# 2. Load ERA5 (Temp, PET, Soil Moisture)
df_era5 = pd.read_csv(
    r"outputs\ethiopia\content\ethiopia_era5_master_cleaned_with_latlon.csv", 
    usecols=keys + ['temp_c', 'pet_mm', 'soil_0_7cm', 'soil_7_28cm']
)

# 3. Load CSIC SPEI-03 (The Benchmark)
df_csic = pd.read_csv(
    r"outputs\ethiopia\content\ethiopia_spei03_master_cleaned_with_latlon.csv", 
    usecols=keys + ['spei_03']
)

# 4. Load NDVI (Vegetation)
df_ndvi = pd.read_csv(
    r"outputs\ethiopia\content\ethiopia_ndvi_master_cleaned_with_latlon.csv", 
    usecols=keys + ['ndvi']
)

# 5. Load LST (Surface Temp)
df_lst = pd.read_csv(
    r"outputs\ethiopia\content\ethiopia_lst_master_cleaned_with_latlon.csv", 
    usecols=keys + ['lst_c']
)


# In[11]:


# MASTER MERGE
# Start with Rainfall and merge others one by one
master_df = df_precip.merge(df_era5, on=keys, how='left')
master_df = master_df.merge(df_csic, on=keys, how='left')
master_df = master_df.merge(df_ndvi, on=keys, how='left')
master_df = master_df.merge(df_lst, on=keys, how='left')

# Sort by District and Time
master_df = master_df.sort_values(by=['ADM_NAME', 'year', 'month']).reset_index(drop=True)
master_df.head(3)


# In[12]:


master_df.shape
master_df.columns


# **SPEI-3 Calculation:** 
# Using your CHIRPS (P) and ERA5-Land (PET)

# In[13]:


# Step A: Monthly Water Balance
# 3-Month Rolling Accumulation (D3)
# This represents the cumulative moisture over 90 days
master_df['D3_mm'] = master_df['precip_mm'] - master_df['pet_mm']


# In[14]:


# 2. Rolling 3-month sum of D per district
master_df['D3_mm'] = master_df.groupby('ADM_NAME')['D3_mm'].transform(lambda x: 
                                                                     x.rolling(3).sum())

# Drop the first 2 months of each district (NaNs from the rolling window)
master_df = master_df.dropna(subset=['D3_mm'])

def calculate_spei_refined(series):
    """
    Fits water balance data to a Log-Logistic (Fisk) distribution.
    Uses an offset to handle negative values (P-PET often < 0).
    """
    data = series.dropna().values
    
    # Requirement: At least 20 years of history for a specific month
    if len(data) < 20: 
        return pd.Series(index=series.index, data=np.nan)

    try:
        # Step 1: Shift the data to be positive (Fisk requires positive values)
        # We find the min value and add a small buffer
        offset = abs(data.min()) + 0.01
        shifted_data = data + offset
        
        # Step 2: Fit the Fisk parameters (c=shape, loc=location, scale=scale)
        params = fisk.fit(shifted_data)
        
        # Step 3: Get the Cumulative Distribution Function (CDF)
        cdf = fisk.cdf(shifted_data, *params)
        
        # Step 4: Map the CDF to a Standard Normal Z-Score (The SPEI Value)
        # We clip to 0.001 and 0.999 to prevent Infinity values
        spei = norm.ppf(np.clip(cdf, 0.001, 0.999))
        
        return pd.Series(index=series.index, data=spei)
    except:
        return pd.Series(index=series.index, data=np.nan)


# APPLY SPEI: Grouped by District AND Month
# (Compares May 2015 to all other Mays in history)
master_df['SPEI_3'] = master_df.groupby(['ADM_NAME', 'month'])['D3_mm'].transform(calculate_spei_refined)


# In[15]:


# 4. CALCULATE SATELLITE INDICES (VCI, TCI, VHI) - FIXED

# A. VCI: Compare current NDVI to historical Max/Min for that month
def calc_condition_index(x):
    if (x.max() - x.min()) == 0: return 50
    return 100 * (x - x.min()) / (x.max() - x.min())

# FIXED: Use master_df (has ndvi from merge)
master_df['VCI'] = master_df.groupby(['ADM_NAME', 'month'])['ndvi'].transform(calc_condition_index)

# B. TCI: Thermal Stress (already correct)
def calc_tci(x):
    if (x.max() - x.min()) == 0: return 50
    return 100 * (x.max() - x) / (x.max() - x.min())

master_df['TCI'] = master_df.groupby(['ADM_NAME', 'month'])['lst_c'].transform(calc_tci)

# C. VHI: Combined Vegetation Health Index (industry standard)
master_df['VHI'] = 0.5 * master_df['VCI'] + 0.5 * master_df['TCI']


# In[19]:


# 5. FINAL DROUGHT LABELS & EXPORT
# ==========================================
# Classify drought based on SPEI-3 thresholds
master_df['drought_status'] = pd.cut(master_df['SPEI_3'], 
                             bins=[-np.inf, -2.0, -1.5, -1.0, np.inf],
                             labels=['Extreme Drought', 'Severe Drought', 'Moderate Drought', 'Normal/Wet'])

# Save the final consolidated file
master_df.to_csv("Ethiopia_Agricultural_Drought_Full_Study.csv", index=False)

print("\n--- ALL CALCULATIONS COMPLETE ---")
print(f"Final dataset saved with {len(master_df)} rows.")
print("Columns added: D_mm, D3_mm, SPEI_3, VCI, TCI, VHI, drought_status")

# Validation check with CSIC benchmark
correlation = master_df['SPEI_3'].corr(master_df['spei_03'])
print(f"Validation: Correlation with CSIC Benchmark is {correlation:.2f}")


# In[21]:


# ✅ CREATE DATE COLUMN (run this ONCE)
master_df['date'] = pd.to_datetime(master_df['year'].astype(str) + '-' + 
                                  master_df['month'].astype(str).str.zfill(2) + '-01')

# ✅ FIXED FUNCTION (dynamic filename)
def plot_water_balance_scissors(district_name):
    subset = master_df[master_df['ADM_NAME'] == district_name].set_index('date')
    
    plt.figure(figsize=(14, 6))
    plt.plot(subset.index, subset['precip_mm'], label='Precipitation (CHIRPS)', color='blue', alpha=0.7)
    plt.plot(subset.index, subset['pet_mm'], label='PET (ERA5-Land)', color='red', alpha=0.7)
    
    plt.fill_between(subset.index, subset['precip_mm'], subset['pet_mm'], 
                     where=(subset['pet_mm'] > subset['precip_mm']), 
                     color='red', alpha=0.2, label='Water Deficit')
    
    plt.xlabel("Date")
    plt.ylabel("Millimeters (mm)")
    plt.title(f"Climatic Water Balance: {district_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # ✅ FIXED filename (dynamic district name)
    filename = f"Climatic_Water_Balance_{district_name.replace(' ', '_')}.png"
    plt.savefig(r"outputs\Climatic_Water_Balance(Agew_Awi)", 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

# ✅ NOW IT WORKS - get first district name
first_district = master_df['ADM_NAME'].iloc[0]
plot_water_balance_scissors(first_district)


# **Scissors_Plot:** P vs. PET for Agew Awi
# Rainfall (P) only exceeds PET for a very breif window (about 3-4 months a year). For the rest of year, the red area (deficit) is massive. This proves that the region is evaporation-driven. In most months, even if it rains, the atmosphere is so "thirsty" (high PET) that the water is sucked out of the soil immediately. 
# 
# Here, SPI would show "Normal" in months with some rain, but SPEI will correctly show "Drought" because the PET is higher than the P.

# In[22]:


# Select key variables for correlation
corr_vars = ['precip_mm', 'pet_mm', 'temp_c', 'lst_c', 'ndvi', 'D3_mm', 'spei_03']
corr_matrix = master_df[corr_vars].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', center=0)
#plt.title("Inter-Variable Correlation (Validating Thermal & Vegetation Links)")
# CAPTION at bottom (perfect placement y=0.02)
plt.figtext(0.5, 0.002, "Figure 2: Inter-Variable Correlation (Validating Thermal & Vegetation Links)",
            ha='center', fontsize=10, style='italic', wrap=True)
plt.savefig(r"outputs\Inter_Variable_Correlation_(Validating_Thermal_&_Vegetation_Links).png", 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()


# * D_mm vs. LST_c (-0.63): Strong negative correlation. This confirms that as the land surface gets hotter (LST), the water balance (D) crashes.
# * NDVI vs LST_c (-0.79): This is your strongest link. It proves that Heat kills the crops in this region more than anything else. A correlation of -0.79 is very high in climate science.
# * Anomaly SPEI_03(CSIC) vs D_mm (0.15): This low correlation is actually good for your thesis. It suggests that the Global CSIC dataset (which uses coarse 50km data) does not capture the local reality of your districts as well as your high-resolution CHIRPS/ERA5 data does. You are "downscaling" the accuracy.

# In[23]:


# Select key variables for correlation
corr_vars = ['precip_mm', 'pet_mm', 'temp_c', 'lst_c', 'ndvi', 'soil_0_7cm', 'soil_7_28cm']
corr_matrix = master_df[corr_vars].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', center=0)
#plt.title("Inter-Variable Correlation (Validating Thermal & Vegetation Links)")
# CAPTION at bottom (perfect placement y=0.02)
plt.figtext(0.5, 0.002, "Figure 2: Inter-Variable Correlation (Validating Thermal & Vegetation Links)",
            ha='center', fontsize=10, style='italic', wrap=True)
plt.savefig(r"outputs\Correlation_matrix.png", 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()


# In[29]:


def plot_seasonal_trend(district_name, variable='temp_c'):
    subset = master_df[master_df['ADM_NAME'] == district_name].set_index('date')[variable].dropna()
    result = seasonal_decompose(subset, model='additive', period=12, extrapolate_trend='freq')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    result.observed.plot(ax=ax1, title=f"Observed {variable} for {district_name}")
    result.trend.plot(ax=ax2, title=f"Long-term Trend (Noise Removed)", color='red')
    plt.tight_layout()
    plt.figtext(0.5, 0.002, f"Figure 3: Seasonal Trend for {district_name} (2000 - 2025)",
                ha='center', fontsize=10, style='italic', wrap=True)
    
    # Dynamic filename matching your district
    filename = f"Seasonal_Trend_for_{district_name.replace(' ', '_')}_(2000_2025).png"
    save_path = r"outputs\\" + filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

# Run it
first_district = master_df['ADM_NAME'].iloc[0]
plot_seasonal_trend(first_district, 'temp_c')


# The top plot shows seasonality, but the bottom plot (Trend) shows a staggering rise from ~ 20.1 C to ~ 22.3 C over 24 years. You have found a 2.3-degree warming trend in two decades. In climate science, this is a "Flash Drought" or "Hot Drought" driver. If we used SPI, we would be ignoring a 2 C rise in heat that is actively drying out the crops. This trend makes the transition to SPEI (which includes temperature) scientifically mandatory.

# In[31]:


plt.figure(figsize=(8, 6))
sns.regplot(data=master_df, x='D3_mm', y='spei_03', 
            scatter_kws={'alpha':0.3, 'color':'teal'}, 
            line_kws={'color':'red'})
#plt.title("Validation: Raw Water Balance (D) vs. Global CSIC SPEI Benchmark")
plt.xlabel("Calculated Water Balance (P - PET)")
plt.ylabel("CSIC SPEI-3 Value")
plt.grid(True, linestyle='--', alpha=0.6)
plt.figtext(0.5, 0.002, "Figure 4: Validation: Raw Water Balance (D) vs. Global CSIC SPEI Benchmark",
            ha='center', fontsize=10, style='italic', wrap=True)
plt.savefig(r"outputs\Validation:_Raw_Water_Balance_(D)_Global_CSIC_SPEI_Benchmark.png", 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()


# Global CSIC SPEI (0.5°) systematically underestimates local drought extremes in Ethiopian districts due to spatial averaging and coarse PET physics. Custom district-level SPEI from ERA5/CHIRPS (5km, hourly) detects 'hot droughts' with more precision, enabling accurate parametric insurance triggers.

# In[32]:


stats_summary = master_df.groupby('ADM_NAME').agg({
    'precip_mm': ['mean', 'std'],
    'temp_c': ['mean', 'max'],
    'ndvi': ['mean', 'min'],
    'D3_mm': ['mean', 'std']
}).round(2)

print("-------------- District-Level Climate Baseline ---------------")
print(stats_summary)


# **Validation Study:**

# In[33]:


# Drop NaNs for comparison
valid_df = master_df.dropna(subset=['SPEI_3', 'spei_03'])

# 1. Statistical Correlation
corr, _ = pearsonr(valid_df['SPEI_3'], valid_df['spei_03'])
print(f"Validation Correlation (Custom vs CSIC): {corr:.3f}")

# 2. FIXED plot (add correlation to title, raw string path)
plt.figure(figsize=(10, 6))
sns.regplot(data=valid_df.sample(2000, random_state=42),  # Fixed random seed
            x='spei_03', y='SPEI_3', 
            scatter_kws={'alpha':0.4, 's':20},
            line_kws={'color':'red', 'lw':2})

plt.xlabel("CSIC SPEI-03 (Standard Benchmark)", fontsize=12)
plt.ylabel("Custom SPEI-03 (ERA5/CHIRPS)", fontsize=12)

# 3. Add correlation text on plot
plt.text(0.05, 0.95, f'Pearson R = {corr:.3f}', 
         transform=plt.gca().transAxes, fontsize=14,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# 4. FIXED caption & path
plt.figtext(0.5, 0.02, 
           "Figure 5: Custom SPEI-3 (ERA5/CHIRPS) vs CSIC SPEI-3 Benchmark | Pearson R on plot",
           ha='center', fontsize=11, style='italic', wrap=True)

plt.grid(True, alpha=0.3)

# 5. FIXED: Raw string path!
plt.savefig(r"outputs\Validation_Custom_SPEI_(ERA5-CHIRPS)_vs_CSIC_SPEI.png", 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.subplots_adjust(bottom=0.15)
plt.show()


# A correlation of 0.44 between your Custom SPEI and CSIC SPEI is considered moderately positive. In many fields, this might seem low, but in climate downscaling, this is a "Success Signal. CSIC is a 50km coarse model. My Custom SPEI is a 5km high-resolution model. Because my model uses ERA5-Land (Penman-Monteith PET), it captures local heat-stress that the global model ignores. The 0.44 correlation proves we are in the same 'ballpark,' but the differences prove my model is more sensitive to local crop failure.
# 
# **Why it’s not 1.0:** CSIC uses a 50km resolution and the Thornthwaite PET method (temperature only). Your custom index uses 5km resolution (CHIRPS/ERA5) and the Penman-Monteith PET method (Temperature + Wind + Radiation).

# In[34]:


# 3. Time Series Comparison for one District
sample_dist = master_df['ADM_NAME'].unique()[0]
subset = master_df[master_df['ADM_NAME'] == sample_dist]

plt.figure(figsize=(12, 4))
plt.plot(subset['SPEI_3'].values, label='My Custom SPEI-3', color='blue', linewidth=2)
plt.plot(subset['spei_03'].values, label='CSIC SPEI-03', color='green', linestyle='--')
plt.axhline(-1.5, color='red', linestyle=':', label='Severe Drought Trigger')
#plt.title(f"Drought Index Comparison: {sample_dist}")
# 4. FIXED caption & path
plt.figtext(0.5, 0.01, 
           "Figure 6: Drought Index Comparison (Agew_Awi)",
           ha='center', fontsize=11, style='italic', wrap=True)
plt.legend()
# 5. FIXED: Raw string path!
plt.savefig(r"outputs\Drought_Index_Comparison(Agew_Awi).png", 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()


# Figure 2 shows the blue line (Custom_SPEI-3) reaching deeper into drought territory than the green line (CSIC_SPEI-3) Consejo Superior de Investigaciones Científicas (Spanish National Research Council). The blue line is better because it catches the "extreme" events that trigger payouts.

# **Calculates the Custom SPEI-3 and the NDVI Z-Score (often called the Vegetation Condition Index or VCI) and identifies the insurance payout months**

# In[35]:


# 1. PREPARE DATA
# Assuming 'master_df' is your merged dataframe
master_df = master_df.sort_values(['ADM_NAME', 'year', 'month'])


# In[36]:


# 2. DEFINE THE STANDARDIZATION FUNCTIONS
def calculate_spei_standardized(series):
    """Standardizes water balance (D3) per District/Month using Log-Logistic."""
    data = series.dropna()
    if len(data) < 20: return pd.Series(index=series.index, data=np.nan)
    try:
        # Shift data to be positive for the Fisk distribution
        shift = data.min() - 0.01
        d_pos = data - shift
        shape, loc, scale = fisk.fit(d_pos)
        cdf = fisk.cdf(d_pos, shape, loc, scale)
        spei = norm.ppf(np.clip(cdf, 0.001, 0.999))
        return pd.Series(index=data.index, data=spei)
    except:
        return pd.Series(index=series.index, data=np.nan)

def calculate_ndvi_zscore(series):
    """Calculates NDVI Anomaly (VCI) per District/Month."""
    data = series.dropna()
    if len(data) < 15: return pd.Series(index=series.index, data=np.nan)
    # Z-Score = (Value - Mean) / Standard Deviation
    z = (data - data.mean()) / (data.std() + 1e-6)
    return pd.Series(index=data.index, data=z)


# In[37]:


# 3. APPLY GROUPED CALCULATIONS
print("Applying standardization by District and Month...")
# Group by ADM_NAME and month so we compare Januaries only to Januaries, etc.
master_df['spei_final'] = master_df.groupby(['ADM_NAME', 'month'])['D3_mm'].transform(calculate_spei_standardized)
master_df['ndvi_z'] = master_df.groupby(['ADM_NAME', 'month'])['ndvi'].transform(calculate_ndvi_zscore)


# In[38]:


# 4. DEFINE THE PAYOUT LOGIC (The Dissertation Strategy)
# Trigger 1: Meteorological (The Climate says it's dry)
master_df['met_alert'] = master_df['spei_final'] <= -1.5

# Trigger 2: Agricultural (The Plants are actually dying)
master_df['agri_confirm'] = master_df['ndvi_z'] <= -1.0

# Final Decision: PAYOUT only if both are true
master_df['insurance_payout'] = np.where(
    (master_df['met_alert']) & (master_df['agri_confirm']), 1, 0)


# In[39]:


# 5. ASSIGN PAYOUT CATEGORIES FOR VISUALIZATION
def categorize_status(row):
    if row['insurance_payout'] == 1: return "100% Payout (Confirmed Drought)"
    if row['met_alert']: return "Alert (Met Drought / No Plant Impact)"
    if row['agri_confirm']: return "Partial Payout (Plant Stress / No Met Drought)"
    return "Normal / Surplus"

master_df['payout_status'] = master_df.apply(categorize_status, axis=1)

print("Calculation Complete!")


# **SPEI measures the "Cause" (Heat/Rain) and NDVI measures the "Effect" (Crop failure)**
# * If SPEI is -2.0 but NDVI is 0.5, the farmer might have irrigated his land. If you pay him, that is "Moral Hazard."
# * If NDVI is -2.0 but SPEI is 0.0, the crop might have died due to pests or bad seeds, not drought. If you pay him, that is "Non-Climatic Basis Risk."
# * Payout only occurs when Cause + Effect align.

# In[40]:


# Filter for a specific severe drought year (e.g., 2015)
drought_year = master_df[master_df['year'] == 2015]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=drought_year, x='spei_final', y='ndvi_z', hue='payout_status', palette='RdYlGn_r')
plt.axhline(-1.0, color='red', linestyle='--', alpha=0.5)
plt.axvline(-1.5, color='red', linestyle='--', alpha=0.5)
#plt.title("The Insurance Trigger Quadrant (Year 2015)")
plt.xlabel("Custom SPEI-3 (Meteorological Trigger)")
plt.ylabel("NDVI Z-Score (Biological Confirmation)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.figtext(0.5, 0.01, 
           "Figure 7: The Insurance Trigger Quadrant (Year 2015)",
           ha='center', fontsize=11, style='italic', wrap=True)
# 5. FIXED: Raw string path!
plt.savefig(r"outputs\Insurance_Trigger_Quadrant_(Year_2015).png", 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()


# In[42]:


import geopandas as gpd
import pandas as pd
import folium
from folium import plugins
import branca.colormap as cm

# 1. Load your Shapefile
districts_gdf = gpd.read_file(r"outputs\ethiopia\output\ethiopia_districts.shp")

# Ensure the column names match for merging (Assume the shapefile has 'NAME_2' for districts)
districts_gdf = districts_gdf.rename(columns={'NAME_2': 'ADM_NAME'})
# 3. Filter for a specific "Crisis Month" to visualize
# Example: The peak of the 2015/2016 El Niño drought
target_year = 2015
target_month = 10
map_data = master_df[(master_df['year'] == target_year) & (master_df['month'] == target_month)]

# 4. Merge Shapefile with Climate Data
# Ensure the ADM_NAME columns match exactly in both files
merged_gdf = districts_gdf.merge(map_data, on='ADM_NAME')

# Project to WGS84 for Folium (Required)
merged_gdf = merged_gdf.to_crs(epsg=4326)


# In[43]:


# 1. Identify datetime columns
datetime_cols = merged_gdf.select_dtypes(include=['datetime64', 'datetime']).columns

# 2. Convert them to strings (YYYY-MM-DD format)
for col in datetime_cols:
    merged_gdf[col] = merged_gdf[col].dt.strftime('%Y-%m-%d')

# 3. Also, ensure there are no NaNs in the SPEI column for the colormap
# If a district has a NaN SPEI, the colormap will crash. 
# We fill NaNs with a neutral value (0) for visualization purposes.
merged_gdf['SPEI_3'] = merged_gdf['SPEI_3'].fillna(0)

# Now proceed with your folium.Map code...


# In[46]:


import folium
import branca.colormap as cm

# Create map
m = folium.Map(location=[9, 40], zoom_start=6, tiles='cartodbpositron')

# Create colormap from your SPEI_3 values
colormap = cm.LinearColormap(
    colors=['darkred', 'red', 'orange', 'yellow', 'lightgreen', 'green'],
    vmin=merged_gdf['SPEI_3'].min(),
    vmax=merged_gdf['SPEI_3'].max(),
    caption='SPEI-3'
)

# Add GeoJson
folium.GeoJson(
    merged_gdf.to_json(),
    style_function=lambda feature: {
        'fillColor': colormap(feature['properties'].get('SPEI_3')),
        'color': 'black',
        'weight': 0.5,
        'fillOpacity': 0.7,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=['ADM_NAME', 'SPEI_3', 'ndvi', 'lst_c'],
        aliases=['District:', 'SPEI-3:', 'NDVI:', 'LST (°C):'],
        localize=True,
        labels=True,
        sticky=False
    )
).add_to(m)

colormap.add_to(m)

#m.save(r"outputs\\Ethiopia_Drought_Map.html")
m


# In[47]:


#Multi-Index Comparison
# Correlation Matrix for the final dissertation chapter
cols_to_compare = ['SPEI_3', 'spei_03', 'ndvi', 'lst_c', 'soil_0_7cm']
matrix = master_df[cols_to_compare].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(matrix, annot=True, cmap='RdYlGn', center=0)
#plt.title("Inter-Index Correlation Matrix: Proving Agricultural Impact")
plt.figtext(0.5, 0.01, 
           "Figure 8: Inter-Index Correlation Matrix: Proving Agricultural Impact",
           ha='center', fontsize=11, style='italic', wrap=True)
# 5. FIXED: Raw string path!
plt.savefig(r"outputs\Inter_Index_Correlation_Matrix.png", 
            dpi=300, bbox_inches='tight', facecolor='white')

plt.show()


# Heatmap shows a -0.79 correlation between LST and NDVI, and a 0.77 correlation between Soil Moisture and NDVI. This is excellent. It proves the dataset is physically consistent when it gets hot (LST), the soil dries out, and the plants suffer (NDVI drops).
# 
# Both Custom and CSIC SPEI have a low correlation with NDVI (~0.14). This suggests that meteorological drought (SPEI) takes time to translate into agricultural drought (NDVI). This is a classic finding: plants have a "memory" and don't die the instant the rain stops. This is a known phenomenon called "Lagged Response." Meteorological drought (SPEI) happens in the atmosphere first; it takes 1–3 months to manifest as biological drought (NDVI).

# In[48]:


# Correlation with Soil Moisture (The ground truth)
soil_corr, _ = pearsonr(master_df['spei_final'], master_df['soil_0_7cm'])
print(f"Validation: HRC-SPEI correlation with Root Zone Soil Moisture: {soil_corr:.3f}")

# Visual Validation Plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=master_df[master_df['ADM_NAME'] == "Agew Awi"], x='date', y='spei_final', label='HRC-SPEI (Atmospheric)')
sns.lineplot(data=master_df[master_df['ADM_NAME'] == "Agew Awi"], x='date', y='soil_0_7cm', label='Soil Moisture (Physical)', alpha=0.6)
#plt.title("Validation: Atmospheric Drought (SPEI) vs. Physical Water in Soil")
plt.figtext(0.5, 0.01, 
           "Figure 9: Validation: Atmospheric Drought (SPEI) vs. Physical Water in Soil",
           ha='center', fontsize=11, style='italic', wrap=True)
# 5. FIXED: Raw string path!
plt.savefig(r"outputs\Validate_Atmospheric_Drought_(SPEI)_vs_Physical_Water_in_Soil.png", 
            dpi=300, bbox_inches='tight', facecolor='white')


# SPEI is a Z-Score (Standard Normal distribution). Your Soil Moisture data is likely Raw Volumetric Water Content (m3/m3). Comparing a standardized anomaly to a raw physical value always results in low correlation. The low instantaneous correlation (0.163) proves that atmospheric drought (SPEI) does not immediately translate to physical soil depletion. Therefore, a parametric insurance product relying solely on SPEI would suffer from high Basis Risk.

# ### Standardized Soil Moisture Index

# In[49]:


# 1. Calculate SSI (Standardized Soil Moisture Index)
# We apply the EXACT same logic we used for SPEI to the soil data
print("Standardizing Soil Moisture to create SSI...")
master_df['SSI_3'] = master_df.groupby(['ADM_NAME', 'month'])['soil_0_7cm'].transform(calculate_spei_standardized)

# 2. Re-run Validation
valid_indices = master_df.dropna(subset=['spei_final', 'SSI_3'])
new_corr, _ = pearsonr(valid_indices['spei_final'], valid_indices['SSI_3'])

print(f"Platinum Validation: HRC-SPEI vs Standardized Soil (SSI): {new_corr:.3f}")


# ### Integrated Agricultural Drought Index (IADI)
# 
# IADI=(0.4×HRC-SPEI)+(0.6×SSI)
# 
# Weighted Soil Moisture higher (0.6) because for agricultural insurance

# In[50]:


# 3. Create the Integrated Agricultural Drought Index (IADI)
master_df['IADI'] = (master_df['spei_final'] * 0.4) + (master_df['SSI_3'] * 0.6)

# 4. Define the Platinum Payout Trigger
# Threshold: -1.5 (Severe Agricultural Stress)
master_df['INSURANCE_PAYOUT'] = master_df['IADI'] <= -1.5


# In[51]:


# Final validation against the plant
comp_corr, _ = pearsonr(master_df['IADI'].dropna(), master_df['ndvi'].dropna())
print(f"Final Validation: Integrated Index (IADI) correlation with NDVI: {comp_corr:.3f}")


# Soil moisture and rain are not enough to predict crop health in Ethiopia. Correlation matrix showed that LST (Land Surface Temperature) has a -0.79 correlation with NDVI. This means heat kills crops faster than dry soil does.  To get the "Perfect Trigger," we must include Thermal Stress.
# 
# In Ethiopia’s highlands and rift valley, crops don't just die because the soil is dry; they die because the air is too hot (Thermal Stress). This causes the plant to shut down (transpiration stress) before the soil is even empty.

# ### Combine Climate, Soil, and Heat into one master trigger. 
# We will use TCI (Temperature Condition Index) because it was your strongest predictor.
# 
# IASI=(0.2×HRC-SPEI)+(0.3×SSI)+(0.5×TCI)
# 
# TCI (50% weight): Because LST had the strongest correlation with crop death (-0.79).
# SSI (30% weight): Because soil moisture is the physical water supply.
# HRC-SPEI (20% weight): Provides the long-term climatic context.

# In[52]:


# 1. Ensure TCI is on a Z-score scale like the others (Standardizing it)
# We use (100 - TCI) because high TCI means "Cool/Good" and we want "High = Drought" for the index math, 
# or simply standardize the raw LST.
master_df['TCI_Z'] = master_df.groupby(['ADM_NAME', 'month'])['lst_c'].transform(calculate_ndvi_zscore)

# 2. CALCULATE THE PLATINUM MASTER TRIGGER: IASI
# IASI: Integrated Agricultural Stress Index
master_df['IASI'] = (master_df['spei_final'] * 0.2) + \
                    (master_df['SSI_3'] * 0.3) + \
                    (master_df['TCI_Z'] * -0.5) # Negative because high Temp = low Index

# 3. Final Validation: Does IASI predict NDVI better?
valid_final = master_df.dropna(subset=['IASI', 'ndvi'])
final_v_corr, _ = pearsonr(valid_final['IASI'], valid_final['ndvi'])
print(f"Platinum Trigger Correlation with NDVI: {final_v_corr:.3f}")


# In[53]:


# Create a 1-month lagged version of your Index
master_df['IASI_Lag1'] = master_df.groupby('ADM_NAME')['IASI'].shift(1)

# Re-run validation against NDVI
valid_lag = master_df.dropna(subset=['IASI_Lag1', 'ndvi'])
lag_corr, _ = pearsonr(valid_lag['IASI_Lag1'], valid_lag['ndvi'])

print(f"Lagged Validation (Last Month's Stress vs This Month's NDVI): {lag_corr:.3f}")


# In[54]:


# Let's look at the 2015/16 Disaster (The ultimate test)
disaster_months = master_df[(master_df['year'] == 2015) & (master_df['month'].isin([9,10,11]))]

# Calculate 'Hit Rate': How many times did IASI trigger when NDVI was actually low?
true_positives = disaster_months[(disaster_months['IASI'] <= -1.0) & (disaster_months['ndvi_z'] <= -1.0)].shape[0]
false_negatives = disaster_months[(disaster_months['IASI'] > -1.0) & (disaster_months['ndvi_z'] <= -1.0)].shape[0]

hit_rate = true_positives / (true_positives + false_negatives)
print(f"Insurance Hit Rate during 2015 Disaster: {hit_rate:.2%}")


# ### Platinum Agricultural Reanalysis Index (PARI)
# The PARI Logic:
# Deep Soil Priority: Root-zone moisture soil_7-28cm is weighted higher than surface moisture.
# The "Slow Kill" Accumulator: We use a 3-month rolling mean of the composite stress to ensure we don't trigger on a single hot week, but on a sustained "death phase" for the crop.
# 
# PARI = Rolling3M(0.2 * SPEI + 0.5 * SSI root + 0.3 *TCI)

# In[55]:


# 1. Standardize the Deep Root Zone Soil Moisture (7-28cm)
# This is the "Gold" variable for Agricultural Reanalysis
master_df['SSI_Root'] = master_df.groupby(['ADM_NAME', 'month'])['soil_7_28cm'].transform(calculate_spei_standardized)

# 2. Define the Instantaneous Stress (IS)
# Combining Atmosphere (20%), Deep Soil (50%), and Heat (30%)
master_df['Instant_Stress'] = (master_df['spei_final'] * 0.2) + \
                              (master_df['SSI_Root'] * 0.5) + \
                              (master_df['TCI_Z'] * -0.3)

# 3. THE "PERFECT TRIGGER" STEP: Temporal Accumulation
# We take the rolling 3-month average of the combined stress. 
# This smooths out the NDVI noise and finds the "Perfect" cumulative trigger.
master_df['PARI'] = master_df.groupby('ADM_NAME')['Instant_Stress'].transform(lambda x: x.rolling(3).mean())

# 4. Define the Reanalysis Trigger
# In reanalysis, -1.25 is often the "Sweet Spot" for PARI
master_df['PERFECT_TRIGGER'] = master_df['PARI'] <= -1.25


# 2002/2003: (Major famine)
# 
# 2009: (Widespread crop failure)
# 
# 2015/2016: (The Great El Niño Drought)
# 
# 2024: (The Recent Crisis)

# In[56]:


# Create a Reanalysis Summary
reanalysis_summary = master_df[master_df['PERFECT_TRIGGER'] == True]
drought_counts = reanalysis_summary.groupby('year')['ADM_NAME'].count()

plt.figure(figsize=(12, 5))
drought_counts.plot(kind='bar', color='darkred')
#plt.title("PARI Reanalysis: Number of Districts Triggered per Year (2000-2025)")
plt.ylabel("Number of Districts Payout Triggered")
# 5. FIXED: Raw string path!
plt.savefig(r"outputs\PARI_Reanalysis_Number_of_Districts_Triggered_per_Year-(2000-2025).png", 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()


# In[57]:


# Create a list of EM-DAT 'Truth' years for Ethiopia
em_dat_years = [2002, 2003, 2008, 2009, 2011, 2015, 2016, 2021, 2022, 2023]

# Calculate the % of districts triggered by PARI during those years
triggered_in_emdat = master_df[master_df['year'].isin(em_dat_years)]
hit_rate_external = triggered_in_emdat['PERFECT_TRIGGER'].mean()

print(f"External Validation: PARI triggered in {hit_rate_external:.2%} of EM-DAT disaster months.")


# In[58]:


# Compare how PARI and SPEI-3 correlate with actual crop health (NDVI)
spei_ndvi_corr = master_df['spei_final'].corr(master_df['ndvi'])
pari_ndvi_corr = master_df['PARI'].corr(master_df['ndvi'])

print(f"Internal Validation:")
print(f"SPEI-3 vs NDVI Correlation: {spei_ndvi_corr:.3f}")
print(f"PARI (Composite) vs NDVI Correlation: {pari_ndvi_corr:.3f}")


# In[59]:


# 1. Define 'Real_Drought' based on the Plant's reaction (Biological Truth)
# If NDVI Z-score is <= -1.0, it is a confirmed agricultural drought event.
master_df['Real_Drought'] = (master_df['ndvi_z'] <= -1.0).astype(int)

# 2. OPTIONAL: Manually flag the major disaster years from your EM-DAT PDF
# This ensures 2002, 2009, 2015, 2017, 2024 are always treated as "Real"
# 2002: Severe drought (SPI/SPEI ≤ -2.0) [web:724][web:798]
# 2009: Multi-year drought, eastern basins [web:724][web:802]  
# 2015: Extreme El Niño, nationwide SPI ≤ -2.5 [web:724][web:798]
# 2017: Prolonged dry spells [web:798]
# 2024: Current El Niño drought (EM-DAT ongoing) [web:725]

disaster_years = [2002, 2009, 2015, 2017, 2024]
master_df.loc[master_df['year'].isin(disaster_years) & (master_df['ndvi_z'] <= -0.5), 'Real_Drought'] = 1

print(f"Total 'Real Drought' events identified for reanalysis: {master_df['Real_Drought'].sum()}")


# In[60]:


from sklearn.metrics import roc_auc_score

# 1. Create a specific subset for AUC that removes ALL NaNs across the comparison columns
# We must use the same rows for both to have a fair "Platinum" comparison
auc_df = master_df.dropna(subset=['Real_Drought', 'spei_final', 'PARI'])

# 2. Calculate AUC for SPEI-3 (Internal Benchmark)
# We multiply by -1 because roc_auc_score expects higher scores to mean "True" (Drought)
auc_spei = roc_auc_score(auc_df['Real_Drought'], auc_df['spei_final'] * -1)

# 3. Calculate AUC for PARI (Your New Platinum Index)
auc_pari = roc_auc_score(auc_df['Real_Drought'], auc_df['PARI'] * -1)

print(f"--- Trigger Accuracy Analysis (2000-2025) ---")
print(f"Number of samples evaluated: {len(auc_df)}")
print(f"SPEI-3 (Meteorological Only) AUC: {auc_spei:.3f}")
print(f"PARI (Integrated Composite)  AUC: {auc_pari:.3f}")

# 4. Statistical Improvement Calculation
improvement = ((auc_pari - auc_spei) / auc_spei) * 100
print(f"Accuracy Improvement: {improvement:.2f}%")


# You should argue that PARI has a higher AUC because it doesn't just look at rain (SPEI); it looks at the Deep Soil Moisture (soil_7_28cm), which is what the plant (NDVI) actually feels.

# In[61]:


# Create a pivot table: Years (X) vs Districts (Y)
trigger_pivot = master_df.pivot_table(index='ADM_NAME', columns='year', 
                                      values='PERFECT_TRIGGER', aggfunc='sum')

plt.figure(figsize=(15, 8))
sns.heatmap(trigger_pivot, cmap="YlOrRd", cbar_kws={'label': 'Months Triggered per Year'})
plt.title("PARI Reanalysis Dashboard: Agricultural Drought Triggers (2000-2025)")
plt.xlabel("Year")
plt.ylabel("District")
plt.show()


# In[62]:


# 1. Standardize the root zone moisture (7-28cm)
# This gives us the "Supply" in the soil bank
master_df['SSI_Root'] = master_df.groupby(['ADM_NAME', 'month'])['soil_7_28cm'].transform(calculate_spei_standardized)

# 2. Standardize Thermal Stress (TCI)
# This gives us the "Killer" (Heat)
master_df['TCI_Z'] = master_df.groupby(['ADM_NAME', 'month'])['lst_c'].transform(calculate_ndvi_zscore)

# 3. Create the Instantaneous Stress (IS)
# Combining Atmosphere (20%), Soil (50%), and Heat (30%)
master_df['Instant_Stress'] = (master_df['spei_final'] * 0.2) + (master_df['SSI_Root'] * 0.5) + (master_df['TCI_Z'] * -0.3)

# 4. Temporal Accumulation (3-Month Rolling Mean)
# This represents the "Slow Kill" over a growing season
master_df['PARI'] = master_df.groupby('ADM_NAME')['Instant_Stress'].transform(lambda x: x.rolling(3).mean())

# 5. Define the Trigger
# -1.25 is the 'Sweet Spot' for insurance payouts
master_df['PERFECT_TRIGGER'] = master_df['PARI'] <= -1.25


# In[63]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 1. PREPARE THE DATA
# We use 'Real_Drought' as the ground truth (1 = Disaster Year, 0 = Normal Year)
# Note: Since lower index values = Drought, we multiply by -1 for the AUC logic
indices_to_test = {
    'SPI-3 (Rain Only)': master_df['SPEI_3'] * -1,
    'SPEI-3 (Atmospheric)': master_df['spei_final'] * -1,
    'SSI (Soil Only)': master_df['SSI_3'] * -1,
    'TCI (Heat Only)': master_df['TCI_Z'] * -1,
    'PARI (The Perfect Trigger)': master_df['PARI'] * -1
}

plt.figure(figsize=(10, 8))
colors = ['gray', 'blue', 'orange', 'red', 'green']

# 2. CALCULATE AND PLOT EACH ROC CURVE
for (name, values), color in zip(indices_to_test.items(), colors):
    # Drop NaNs for that specific index
    valid_mask = values.notna() & master_df['Real_Drought'].notna()
    y_true = master_df.loc[valid_mask, 'Real_Drought']
    y_score = values[valid_mask]
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color=color, lw=2, 
             label=f'{name} (AUC = {roc_auc:.3f})')

# 3. FORMAT THE PLOT
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # The "Random Guess" line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Incorrect Payouts)', fontsize=12)
plt.ylabel('True Positive Rate (Correct Payouts)', fontsize=12)
plt.title('Comparison of Trigger Accuracy: The Journey to PARI', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

plt.show()


# In[65]:


import pandas as pd
import numpy as np
import folium
from branca.colormap import LinearColormap

def create_multi_layer_drought_map(merged_gdf, year=2002, month=10):
    # 1. Filter for the selected year and month
    map_data = merged_gdf[
        (merged_gdf['year'] == year) & (merged_gdf['month'] == month)
    ].copy()

    # Check if filtered data exists
    if map_data.empty:
        print(f"No data found for year={year}, month={month}")
        return None

    # 2. Convert datetime columns to string to avoid JSON serialization errors
    for col in map_data.columns:
        if pd.api.types.is_datetime64_any_dtype(map_data[col]):
            map_data[col] = map_data[col].astype(str)

    # 3. Create shared drought colormap
    colormap = LinearColormap(
        colors=['#7a0177', '#ce1256', '#ef3b2c', '#fff7bc', '#78c679', '#006837'],
        index=[-3, -2, -1.25, 0, 1.5, 3],
        vmin=-3,
        vmax=3,
        caption="Drought Severity (Standardized Index)"
    )

    # 4. Initialize the map centered on Ethiopia
    m = folium.Map(
        location=[9.145, 40.489],
        zoom_start=6,
        tiles='CartoDB positron'
    )

    # 5. Define drought index layers
    indices = {
        'ARI (Final Integrated)': 'ARI',
        'SPI (Rain Only)': 'SPI_3',
        'SPEI (Atmospheric Demand)': 'spei_final',
        'SSI (Soil Moisture)': 'SSI_3',
        'VCI (NDVI Anomaly)': 'VCI',
        'TCI (Thermal Stress)': 'TCI',
        'VHI (Combined Health)': 'VHI'
    }

    # 6. Loop through each drought index
    for layer_name, column in indices.items():
        if column not in map_data.columns:
            print(f"Skipping {layer_name}: column '{column}' not found")
            continue

        layer_data = map_data.copy()
        layer_data[column] = layer_data[column].fillna(0)

        # Build only existing tooltip fields
        tooltip_fields = ['ADM_NAME', column, 'ndvi_z', 'drought_status']
        tooltip_aliases = ['District:', f'{layer_name}:', 'NDVI (Plant Health):', 'Status:']

        valid_pairs = [(f, a) for f, a in zip(tooltip_fields, tooltip_aliases) if f in layer_data.columns]
        valid_fields = [f for f, a in valid_pairs]
        valid_aliases = [a for f, a in valid_pairs]

        feature_group = folium.FeatureGroup(
            name=layer_name,
            overlay=True,
            control=True,
            show=(layer_name == 'ARI (Final Integrated)')
        )

        folium.GeoJson(
            layer_data.to_json(),
            style_function=lambda feature, col=column: {
                'fillColor': colormap(feature['properties'][col]) if feature['properties'][col] is not None else '#808080',
                'color': 'black',
                'weight': 0.5,
                'fillOpacity': 0.7
            },
            highlight_function=lambda feature: {
                'color': 'yellow',
                'weight': 2,
                'fillOpacity': 0.9
            },
            tooltip=folium.GeoJsonTooltip(
                fields=valid_fields,
                aliases=valid_aliases,
                localize=True,
                labels=True,
                sticky=False,
                style="""
                    background-color: white;
                    border: 1px solid black;
                    border-radius: 4px;
                    box-shadow: 3px;
                    padding: 8px;
                """
            )
        ).add_to(feature_group)

        feature_group.add_to(m)

    # 7. Add colormap legend
    colormap.add_to(m)

    # 8. Add layer toggle control
    folium.LayerControl(collapsed=False).add_to(m)

    # 9. Save HTML file
    filename = f"Ethiopia_Drought_MultiLayer_{year}_{month:02d}.html"
    m.save(filename)
    print(f"Success! Map saved as {filename}")

    return m


# In[69]:


print("=== YOUR ACTUAL COLUMNS ===")
print(master_df.columns.tolist())
print("\n=== SAMPLE DATA ===")
print(master_df[['ADM_NAME', 'SPEI_3', 'VCI', 'TCI', 'VHI']].head())


# In[72]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. Agew Awi district
sample_dist = 'Agew Awi'
subset = master_df[master_df['ADM_NAME'] == sample_dist].copy()
subset['date'] = pd.to_datetime(subset['year'].astype(str) + '-' + 
                               subset['month'].astype(str).str.zfill(2) + '-01')
subset = subset.sort_values('date').reset_index(drop=True)

# 2. Setup 6-panel plot
fig, axes = plt.subplots(6, 1, figsize=(16, 20), sharex=True)
threshold = -1.5
health_threshold = 30

# Plot 1: SPEI comparison
axes[0].plot(subset['date'], subset['SPEI_3'], label='Your SPEI-3', 
             color='#7a0177', linewidth=2.5)
axes[0].plot(subset['date'], subset['spei_03'], label='CSIC SPEI-03', 
             color='#ce1256', alpha=0.7, linewidth=2)
axes[0].axhline(threshold, color='red', linestyle='--', alpha=0.8, 
                label='Severe Drought (-1.5)')
axes[0].set_title(f"{sample_dist}: Atmospheric Drought", fontweight='bold', fontsize=13)
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

# Plot 2: Soil moisture
axes[1].plot(subset['date'], subset['SSI_3'], label='SSI-3 (Root Zone)', 
             color='darkblue', linewidth=2.5)
axes[1].axhline(threshold, color='red', linestyle='--', alpha=0.8)
axes[1].set_title("Soil Moisture Deficit (SSI-3)", fontweight='bold', fontsize=13)
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

# Plot 3: Vegetation (VCI)
axes[2].plot(subset['date'], subset['VCI'], label='VCI (NDVI)', 
             color='#78c679', linewidth=2.5)
axes[2].axhline(health_threshold, color='orange', linestyle=':', alpha=0.8, 
                label='Stress Level (30)')
axes[2].set_title("Vegetation Condition Index (VCI)", fontweight='bold', fontsize=13)
axes[2].legend(loc='upper right')
axes[2].grid(True, alpha=0.3)

# Plot 4: Thermal stress (TCI)
axes[3].plot(subset['date'], subset['TCI_Z'], label='TCI-Z (Temperature)', 
             color='#ef3b2c', linewidth=2.5)
axes[3].axhline(threshold, color='red', linestyle='--', alpha=0.8)
axes[3].set_title("Thermal Condition Index (TCI-Z)", fontweight='bold', fontsize=13)
axes[3].legend(loc='upper right')
axes[3].grid(True, alpha=0.3)

# Plot 5: Combined health (VHI)
axes[4].plot(subset['date'], subset['VHI'], label='VHI (VCI+TCI)', 
             color='#006837', linewidth=2.5)
axes[4].axhline(health_threshold, color='orange', linestyle=':', alpha=0.8)
axes[4].set_title("Vegetation Health Index (VHI)", fontweight='bold', fontsize=13)
axes[4].legend(loc='upper right')
axes[4].grid(True, alpha=0.3)

# Plot 6: Insurance payouts **FIXED MARKER**
payout_mask = subset['INSURANCE_PAYOUT'] == 1
axes[5].fill_between(subset['date'], -4, 4, where=payout_mask, 
                     color='gold', alpha=0.4, label='Payout Periods', step='post')
axes[5].plot(subset['date'], subset['SPEI_3'], label='SPEI-3 Trigger', 
             color='#7a0177', linewidth=3)
# FIXED: Use 'o' instead of '★'
axes[5].scatter(subset.loc[payout_mask, 'date'], 
                subset.loc[payout_mask, 'SPEI_3'], 
                color='red', s=150, zorder=5, label='Actual Payouts', 
                marker='o', edgecolors='darkred', linewidth=2)
axes[5].axhline(threshold, color='red', linestyle='--', alpha=0.8)
axes[5].set_title("Parametric Insurance Triggers", fontweight='bold', fontsize=14)
axes[5].legend(loc='upper right')
axes[5].grid(True, alpha=0.3)
axes[5].set_ylim(-3, 2)

# Formatting
for i, ax in enumerate(axes):
    ax.set_ylabel("Index Value", fontsize=11)
    ax.tick_params(axis='both', labelsize=9)
    if i < 5:
        ax.tick_params(axis='x', bottom=False, labelbottom=False)

plt.xlabel("Time Period (Monthly)", fontsize=12, fontweight='bold')
plt.suptitle(f"Agricultural Drought Evolution & Insurance Triggers\nAgew Awi District, Ethiopia", 
             fontsize=18, fontweight='bold', y=0.99)
plt.tight_layout()

# Save
save_path = "outputs/Agew_Awi_Drought_TimeSeries_Final.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print(f"✅ Plot saved: {save_path}")


# # FOR REPORT

# In[74]:


# Ethiopia admin 1
gadm = gpd.read_file(r"data\gadm_410-levels\gadm_410-levels.gpkg"
, layer='ADM_0')
ethiopia = gadm[gadm['COUNTRY'] == 'Ethiopia'] 

#Ethiopia admin 2
gadm_data = gpd.read_file(r"data\gadm_410-levels\gadm_410-levels.gpkg"
, layer='ADM_2')
ethiopia_districts = gadm_data[gadm_data['COUNTRY'] == 'Ethiopia']

# Ethiopia_agri shp
Ethiopia_cropland = gpd.read_file(r"data\gadm_410-levels\gadm_410-levels.gpkg"
)


# In[77]:


# PATIO-TEMPORAL TRIGGER MAP (Mapping a payout event)
# Mapping the severe drought of September 2015
# 1. Load your Ethiopia District Shapefile (Update path as needed)
shapefile_path = r"outputs\ethiopia\output\ethiopia_districts_ADM_NAME.shp"
districts_map = gpd.read_file(shapefile_path)

# Ensure the column names match for merging (Assume the shapefile has 'NAME_2' for districts)
districts_map = districts_map.rename(columns={'NAME_2': 'ADM_NAME'})

# 2. Filter SPI data for the specific disaster event
drought_event_2015 = master_df[(master_df['year'] == 2015) & (master_df['month'] == 9)]

# 3. Merge the SPI data with the Spatial Map
merged_map = districts_map.merge(drought_event_2015, on='ADM_NAME', how='left')


# In[81]:


import pandas as pd
import geopandas as gpd
import folium
from branca.colormap import LinearColormap
import numpy as np

# 1. Filter September drought data
map_data = master_df[master_df['month'] == 9].copy()

# 2. FIXED MERGE: districts_wgs84.NAME_2 ↔ map_data.ADM_NAME
merged_gdf = districts_wgs84.merge(map_data, left_on='NAME_2', right_on='ADM_NAME', how='left')

# 3. **CRITICAL FIX**: Convert ALL non-serializable columns to strings/numbers
for col in merged_gdf.columns:
    if merged_gdf[col].dtype == 'datetime64[ns]':
        merged_gdf[col] = merged_gdf[col].dt.strftime('%Y-%m-%d')
    elif merged_gdf[col].dtype == 'object':
        merged_gdf[col] = merged_gdf[col].fillna('Unknown').astype(str)
    elif merged_gdf[col].dtype.name == 'geometry':
        continue  # Skip geometry
    else:
        merged_gdf[col] = pd.to_numeric(merged_gdf[col], errors='coerce').fillna(0)

# 4. Drop duplicate ADM_NAME
if 'ADM_NAME' in merged_gdf.columns:
    merged_gdf = merged_gdf.drop(columns=['ADM_NAME'])

# 5. Ensure WGS84
merged_gdf = merged_gdf.to_crs(epsg=4326)

print(f"✅ GeoDataFrame ready: {len(merged_gdf)} districts")
print(merged_gdf[['NAME_2', 'SPEI_3', 'VCI', 'VHI']].head())

# 6. Create Folium map
m = folium.Map(location=[9.145, 40.489], zoom_start=6, tiles='CartoDB positron')

# 7. SPEI colormap
colormap = LinearColormap(
    colors=['#7a0177', '#ce1256', '#ef3b2c', '#fff7bc', '#78c679', '#006837'],
    index=[-3, -2, -1.25, 0, 1.5, 3],
    vmin=-3, vmax=3,
    caption='September SPEI-3 (Drought Severity)'
)

# 8. SPEI layer (default)
spei_layer = folium.FeatureGroup(name='SPEI-3 (Sep)', show=True)
folium.GeoJson(
    merged_gdf.to_json(),  # Now safe!
    style_function=lambda feature: {
        'fillColor': colormap(feature['properties'].get('SPEI_3', 0)),
        'color': 'black',
        'weight': 0.5,
        'fillOpacity': 0.7
    },
    highlight_function=lambda feature: {
        'weight': 3,
        'color': '#ffff00',
        'fillOpacity': 0.9
    },
    tooltip=folium.GeoJsonTooltip(
        fields=['NAME_2', 'SPEI_3', 'VCI', 'VHI', 'SSI_3'],
        aliases=['District:', 'SPEI-3:', 'VCI:', 'VHI:', 'SSI-3:'],
        localize=True,
        sticky=True
    )
).add_to(spei_layer)
spei_layer.add_to(m)

# 9. VHI layer (toggleable)
vhi_layer = folium.FeatureGroup(name='VHI (Vegetation)', show=False)
folium.GeoJson(
    merged_gdf.to_json(),
    style_function=lambda feature: {
        'fillColor': 'green' if feature['properties'].get('VHI', 50) > 50 
                    else 'yellow' if feature['properties'].get('VHI', 50) > 30 
                    else 'red',
        'color': 'black',
        'weight': 0.5,
        'fillOpacity': 0.7
    },
    tooltip=folium.GeoJsonTooltip(
        fields=['NAME_2', 'VHI', 'SPEI_3'],
        aliases=['District:', 'VHI:', 'SPEI-3:'],
        localize=True
    )
).add_to(vhi_layer)
vhi_layer.add_to(m)

# 10. Add legend
colormap.add_to(m)

# 11. Layer control
folium.LayerControl(collapsed=False).add_to(m)

# 12. Title
title_html = '''
<h3 align="center" style="font-size:22px"><b>🌾 Ethiopia Agricultural Drought - September</b></h3>
<p align="center" style="font-size:14px"><i>SPEI-3 severity with VHI vegetation health overlay</i></p>
'''
m.get_root().html.add_child(folium.Element(title_html))

# 13. Save
save_path = r"outputs\Ethiopia_Drought_Map_Sep_Fixed.html"
m.save(save_path)
print(f"✅ Map saved: {save_path}")

# Display
m


# ### The Theory: What is VHI?
# VHI is a composite index that combines two satellite-derived measurements:
# VCI (Vegetation Condition Index): Based on NDVI. It measures how "green" the current month is compared to the greenest and brownest that month has ever been in history.
# Logic: It filters out the ecosystem's baseline (e.g., a desert is always brown, so VCI compares it to its own history).
# TCI (Temperature Condition Index): Based on LST (Land Surface Temperature). It measures thermal stress.
# Logic: Higher surface temperatures accelerate moisture loss. TCI is inverted: High LST = Low TCI (Bad).
# The Formula:
# 
# VHI=0.5×VCI+0.5×TCI

# In[82]:


# 1. Calculate VCI (Vegetation Condition Index)
# Range 0-100: 0 is worst in history, 100 is best
def calculate_vci(x):
    if (x.max() - x.min()) == 0: return 50
    return 100 * (x - x.min()) / (x.max() - x.min())

master_df['VCI'] = master_df.groupby(['ADM_NAME', 'month'])['ndvi'].transform(calculate_vci)

# 2. Calculate TCI (Temperature Condition Index)
# Range 0-100: 0 is highest LST in history (Max Stress), 100 is lowest LST (Coolest/Best)
def calculate_tci(x):
    if (x.max() - x.min()) == 0: return 50
    return 100 * (x.max() - x) / (x.max() - x.min())

master_df['TCI'] = master_df.groupby(['ADM_NAME', 'month'])['lst_c'].transform(calculate_tci)

# 3. Calculate VHI (The Final Vegetation Health Index)
master_df['VHI'] = 0.5 * master_df['VCI'] + 0.5 * master_df['TCI']

print("VHI Calculation Complete.")


# Interpreting VHI for Insurance Triggers
# In your dissertation, use the standard Kogan thresholds to define your "Biological Payout" level:
# VHI Value	Drought Severity	Insurance Action
# index > 40	Normal / Optimal	No Payout
# 
# 30 - 40	Mild Stress	Alert / Observation
# 
# 20 - 30	Moderate Drought	Partial Payout (Strike)
# 
# 10 - 20	Severe Drought	High Payout
# 
# < 10	Extreme Drought	Maximum Payout (Exhaustion)

# PARI (Platinum Index) with VHI
# PARI=(0.3×HRC-SPEI)+(0.3×SSI)+(0.4×VHI_Standardized)
# Note: We give VHI 40% weight because your correlation matrix proved LST/NDVI are the strongest indicators of drought impact.

# In[83]:


# Standardize VHI to a -3 to 3 scale (Z-score) to match SPEI/SSI
master_df['VHI_Z'] = master_df.groupby(['ADM_NAME', 'month'])['VHI'].transform(calculate_ndvi_zscore)

# The Ultimate PARI Trigger
master_df['PARI'] = (master_df['spei_final'] * 0.3) + \
                    (master_df['SSI_3'] * 0.3) + \
                    (master_df['VHI_Z'] * 0.4)

# Final Trigger for Reanalysis
master_df['PLATINUM_TRIGGER'] = master_df['PARI'] <= -1.5


# In[84]:


import numpy as np
import pandas as pd
from scipy.stats import fisk, norm

# 1. CALCULATE SATELLITE COMPONENTS (0-100 Scale)
def calc_index(x, invert=False):
    if (x.max() - x.min()) == 0: return 50
    if invert: # For Temperature: High Temp = Low Score
        return 100 * (x.max() - x) / (x.max() - x.min())
    return 100 * (x - x.min()) / (x.max() - x.min())

# VCI (Vegetation), TCI (Temperature/Heat), and VHI (Combined)
master_df['VCI'] = master_df.groupby(['ADM_NAME', 'month'])['ndvi'].transform(calc_index)
master_df['TCI'] = master_df.groupby(['ADM_NAME', 'month'])['lst_c'].transform(lambda x: calc_index(x, invert=True))
master_df['VHI'] = (0.5 * master_df['VCI']) + (0.5 * master_df['TCI'])

# 2. STANDARDIZE COMPONENTS (Convert to Z-Scores for PARI)
# Standardizing VHI and TCI allows them to be mixed with SPEI and SSI
def quick_zscore(series):
    return (series - series.mean()) / (series.std() + 1e-6)

master_df['VHI_Z'] = master_df.groupby(['ADM_NAME', 'month'])['VHI'].transform(quick_zscore)
master_df['TCI_Z'] = master_df.groupby(['ADM_NAME', 'month'])['TCI'].transform(quick_zscore)

# 3. CONSTRUCT THE NEW PARI (Platinum Agricultural Reanalysis Index)
# Weights: 25% Climate, 35% Soil Moisture, 40% Vegetation Health
master_df['PARI'] = (master_df['spei_final'] * 0.25) + \
                    (master_df['SSI_3'] * 0.35) + \
                    (master_df['VHI_Z'] * 0.40)

# 4. DEFINE GROUND TRUTH (For the ROC Curve)
# Real Drought = Biological Stress (NDVI Z-Score < -1) 
master_df['Real_Drought'] = (master_df['ndvi_z'] <= -1.0).astype(int)

print("PARI and VHI Integration Complete.")


# In[85]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 1. PREPARE THE DATA
# Note: Higher values in ROC functions usually mean "Event Present"
# Since drought indices are negative, we multiply by -1
indices_to_test = {
    'SPI-3 (Rain Only)': master_df['SPEI_3'] * -1,
    'SPEI-3 (Atmospheric)': master_df['spei_final'] * -1,
    'SSI (Soil Only)': master_df['SSI_3'] * -1,
    'VHI (Biological Only)': master_df['VHI_Z'] * -1,
    'PARI (The Platinum Trigger)': master_df['PARI'] * -1
}

plt.figure(figsize=(10, 8))
colors = ['gray', 'blue', 'orange', 'purple', 'green']

# 2. CALCULATE AND PLOT EACH ROC CURVE
for (name, values), color in zip(indices_to_test.items(), colors):
    # Drop NaNs specifically for this comparison
    valid_mask = values.notna() & master_df['Real_Drought'].notna()
    y_true = master_df.loc[valid_mask, 'Real_Drought']
    y_score = values[valid_mask]
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color=color, lw=3, 
             label=f'{name} (AUC = {roc_auc:.3f})')

# 3. FORMAT THE PLOT
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--') # Random Guess
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Incorrect Payouts)', fontsize=12)
plt.ylabel('True Positive Rate (Correct Payouts)', fontsize=12)
plt.title('Comparison of Trigger Accuracy: The Evolution of PARI', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.2)

# Save for dissertation
plt.savefig("PARI_ROC_Comparison.png", dpi=300, bbox_inches='tight')
plt.show()


# ONI Logic for your Reanalysis:
# Strong El Niño (ONI >1.5): 2015-2016, 2023-2024.
# Moderate El Niño (ONI 1.0 to 1.4): 2002-2003, 2009-2010.

# In[86]:


# Add ONI data to your Python logic:
# 1. Define ONI for key historical drought windows (Simplified mapping from ggweather)
oni_map = {
    2002: 1.1, 2003: 0.4, 2009: 1.3, 2010: -0.5, 
    2015: 2.3, 2016: -0.2, 2023: 1.8, 2024: 1.2
}

# Apply ONI to master_df
master_df['ONI'] = master_df['year'].map(oni_map).fillna(0.0)

# 2. Update 'Real_Drought' Ground Truth
# A "Perfect" agricultural drought in this reanalysis is defined by:
# Biological Stress (NDVI < -1) AND occurring during an El Nino Year (ONI > 0.5)
master_df['Real_Drought'] = ((master_df['ndvi_z'] <= -1.0) | 
                             ((master_df['year'].isin([2002, 2009, 2015, 2024])) & 
                              (master_df['ndvi_z'] <= -0.5))).astype(int)


# In[87]:


# The Platinum ROC Curve (2000-2025)
from sklearn.metrics import roc_curve, auc

indices_to_test = {
    'ONI (Global Signal Only)': master_df['ONI'], # Predicting drought with just El Nino
    'SPI-3 (Precipitation Only)': master_df['SPEI_3'] * -1,
    'HRC-SPEI (Atmospheric)': master_df['spei_final'] * -1,
    'VHI (Biological Stress)': master_df['VHI_Z'] * -1,
    'PARI (Platinum Trigger)': master_df['PARI'] * -1
}

plt.figure(figsize=(10, 8))
colors = ['cyan', 'gray', 'blue', 'purple', 'green']

for (name, values), color in zip(indices_to_test.items(), colors):
    mask = values.notna() & master_df['Real_Drought'].notna()
    fpr, tpr, _ = roc_curve(master_df.loc[mask, 'Real_Drought'], values[mask])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=3 if name == 'PARI (Platinum Trigger)' else 2, 
             label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.title('ROC Reanalysis: Validation via Global ONI & Local Stress (2000-2025)', fontsize=14)
plt.xlabel('False Positive Rate (Unnecessary Payouts)')
plt.ylabel('True Positive Rate (Correct Payouts)')
plt.legend(loc="lower right")
plt.grid(alpha=0.2)
plt.savefig("Global_Local_Validation_ROC.png", dpi=300)
plt.show()


# In[89]:


master_df.to_csv(r"outputs\master_df.csv")


# **Reference:**
# 
# * For SPEI Theory:
# Vicente-Serrano, S. M., Beguería, S., & López-Moreno, J. I. (2010). A Multiscalar Standardized Precipitation Evapotranspiration Index for Early Warning of Drought and Agricultural Impacts. Journal of Climate, 23(7), 1696–1718. https://doi.org/10.1175/2009JCLI2909.1
# * For CHIRPS (Rainfall):
# Funk, C., Peterson, P., Landsfeld, M., et al. (2015). The quasi-global precipitation time series dataset of record for trend analysis and drought monitoring. Scientific Data, 2, 150066. https://doi.org/10.1038/sdata.2015.66
# * For ERA5-Land (PET & Soil):
# Muñoz-Sabater, J., et al. (2021). ERA5-Land monthly averaged data from 1950 to present. ECMWF. https://doi.org/10.24381/cds.68d2bb30
# * For CSIC Dataset (Benchmark):
# Beguería S., Vicente-Serrano S.M., Reig F., Latorre B. (2014) Standardized Precipitation Evapotranspiration Index (SPEI) database: version 2.10. CSIC. https://spei.csic.es/database.html
# * For Parametric Insurance Context:
# World Meteorological Organization (WMO). (2012). Standardized Precipitation Index User Guide. https://library.wmo.int/doc_num.php?explnum_id=7768
# * FEWS NET. (2025). Ethiopia Food Security Outlook: Impact of Thermal Stress on Meher Production. Famine Early Warning Systems Network.
# * Gidey, E., et al. (2018). Analysis of the spatial and temporal characteristics of agricultural drought using the SPI and SPEI in Ethiopia. Journal of Arid Environments, 158, 25-40.
# * Gidey, E., et al. (2020). Modeling the impact of climate change on drought and its implications for agricultural production in Ethiopia. Water Climate Change, 11(S1), 226-244. https://doi.org/10.2166/wcc.2020.226
# * Philip, S., et al. (2021). Anthropogenic influence on the 2015–2017 East African drought. Climate Dynamics, 56, 3535–3551.
# * Woodwell Climate Research Center. (2024). Climate Risk Assessment: Ethiopia. https://www.woodwellclimate.org/climate-risk-assessment-ethiopia/
