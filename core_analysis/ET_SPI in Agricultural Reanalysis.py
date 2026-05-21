#!/usr/bin/env python
# coding: utf-8

# ![Ethiopia_geology mapfile.jpg](attachment:e2ff63e7-7a71-4dd3-a6c9-c04191590e9c.jpg)
# # Ethiopia
# 
# Ethiopia's geography, climate, and agriculture create extreme drought vulnerability due to sharp elevation-driven rainfall gradients and rainfed farming reliance.
# 
# Ethiopia's central highlands (2,000-4,500m, 60% land) contrast with eastern lowlands (Afar/Somali deserts <500m) and Rift Valley basins. Highlands receive 800-2,000mm bimodal rain (Belg Mar-May, Meher Jun-Sep); lowlands <300mm unimodal. Topography creates rain shadows Amhara highlands (your safest zones) stay wetter vs. eastern basins. !(https://www.woodwellclimate.org/climate-risk-assessment-ethiopia/)
# 
# Tropical lowlands: 25-35°C year-round, peaks 40°C+ in drought. Subtropical highlands: 15-25°C. Cold highlands (>2,400m): 6-16°C. Extreme heat accelerates evapotranspiration, turning meteorological drought (low SPI) into agricultural drought via soil moisture loss. !(https://www.woodwellclimate.org/climate-risk-assessment-ethiopia/) !(https://fews.net/east-africa/ethiopia/fews-net-analysis-note/november-2025)
# 
# Drought-Prone Highlands (60% population): Teff, maize, barley; 85% rainfed. Belg failure kills short-cycle crops. Pastoral Lowlands (east): Livestock on sparse pasture; water points dry first. Rift Valley: Cash crops (sesame, coffee) but flood/drought prone.
# 
# 94% crops depend on Meher rains; SPI-3 tracks soil moisture for planting, SPI-6 streamflow/reservoirs. 1983-85/2011 events (SPI ≤ -2) caused 50-80% yield loss. Eastern basins 3-5x more frequent severe drought vs. your Amhara rankings. Climate change worsens: hotter/warmer baselines shrink growing period 10-20 days by 2050.!(https://doi.org/10.2166/wcc.2020.226)
# District SPI must weight elevation - highland triggers <-1.0, lowland <-0.8 for accurate payouts.!(https://doi.org/10.2166/wcc.2020.226)

# ### What is SPI?
# SPI is a probability-based index developed to quantify precipitation deficits across different climates. Because rainfall is not normally distributed (it is heavily skewed with many dry days and a few massive storms), SPI fits historical rainfall data to a Gamma distribution and then transforms it into a standard normal distribution (a bell curve).
# 
# Around 0: normal conditions.
# 
# About −1: moderately dry (starting to worry about drought).
# 
# About −2 or less: extremely dry (serious drought).
# 
# Positive values (e.g. +1, +2): correspondingly wet or very wet periods.

# The "Standard" Reference: WMO (2012), Standardized Precipitation Index User Guide (WMO-No. 1090), officially recommends SPI as the primary index for drought. It explicitly states that SPI-3 is the best proxy for agricultural drought because it reflects short-to-medium term moisture conditions.
# 
# The African Reanalysis Reference: Funk et al. (2015), The Climate Hazards Group InfraRed Precipitation with Station data (CHIRPS)—a new hazard record for monitoring extremes, proves that CHIRPS-derived SPI is a valid tool for identifying historical agricultural shocks in Africa.
# 
# Agricultural Specificity: Guttman, N. B. (1999), Accepting the Standardized Precipitation Index: A Calculation Algorithm, argues that while SPI-1 is for meteorology, SPI-3 is for agriculture because crop health is the result of accumulated moisture over the 60–90 day growing cycle.

# ### Why SPI-3 for Agricultural Triggers?
# SPI can be calculated on different timescales (1, 3, 6, 12 months). For your dissertation, SPI-3 (a 3-month rolling accumulation) is the optimal proxy for agricultural drought. It reflects the soil moisture accumulation over a standard 90-day crop growing cycle. If SPI-3 hits a severe deficit during the "grain-filling" phase of a crop, crop failure is almost guaranteed.

# ### The Data
# For pixel-by-pixel analysis, you are likely using a gridded dataset like CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data).
# Resolution: CHIRPS provides data at ~5km (0.05°) resolution.
# Depth: 25 years of historical data to accurately fit the Gamma distribution.

# In[1]:


#!pip install climate-indices SPI
# OR
#!pip install standard-precip


# In[2]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.stats import gamma, norm
import numpy as np
import seaborn as sns
import warnings
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
warnings.filterwarnings('ignore')


# In[5]:


# Ethiopia admin 1
gadm = gpd.read_file("C:\\Users\\FlawiyaShirishMore\\OneDrive - Africa Specialty Risks Ltd\\ASR-Parametric_Research_Study\\africa_risk\\Drought\\data\\gadm_410-levels\\gadm_410-levels.gpkg"
, layer='ADM_0')
ethiopia = gadm[gadm['COUNTRY'] == 'Ethiopia'] 

#Ethiopia admin 2
gadm_data = gpd.read_file("C:\\Users\\FlawiyaShirishMore\\OneDrive - Africa Specialty Risks Ltd\\ASR-Parametric_Research_Study\\africa_risk\\Drought\\data\\gadm_410-levels\\gadm_410-levels.gpkg"
, layer='ADM_2')
ethiopia_districts = gadm_data[gadm_data['COUNTRY'] == 'Ethiopia']

# Ethiopia_agri shp
Ethiopia_cropland = gpd.read_file(r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk\Drought\Output\ethiopia\output\ethiopia_filtered_data.shp"
)


# In[6]:


# Load your long format data
df= pd.read_csv("C:\\Users\\FlawiyaShirishMore\\OneDrive - Africa Specialty Risks Ltd\\ASR-Parametric_Research_Study\\africa_risk\\Drought\\Output\\ethiopia\\content\\ethiopia_rainfall_master_cleaned.csv")
print(df.columns.tolist())
print(df.info())


# In[7]:


# EDA: Seasonality Profile (Justifies SPI-3 Meher trigger for dissertation)
monthly_climatology = df.groupby('month')['precip_mm'].mean()

plt.figure(figsize=(10, 6))  # Bigger for thesis
sns.barplot(x=monthly_climatology.index, y=monthly_climatology.values, 
            color='skyblue', edgecolor='navy', linewidth=0.8)

plt.xlabel("Month", fontsize=12)
plt.ylabel("Average Precipitation (mm)", fontsize=12)
#plt.title("Average Monthly Rainfall Across Ethiopian Districts", fontsize=14, fontweight='bold', pad=20)

# CAPTION at bottom (perfect placement y=0.02)
plt.figtext(0.5, 0.009, "Figure 2: Average Monthly Rainfall Across Ethiopian Districts",
            ha='center', fontsize=10, style='italic', wrap=True)

# SAVE FIRST - BEFORE tight_layout()
plt.savefig(r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk\Drought\Output\Ethiopia_Monthly_Rainfall_Climatology.png", 
            dpi=300, bbox_inches='tight', facecolor='white')

plt.tight_layout()
plt.show()


# Figure 2 illustrates the average monthly precipitation across Ethiopian districts, revealing a distinct seasonal climatology. Rainfall is minimal from November to February (the dry Bega season). A smaller, secondary peak occurs around April and May (the Belg short rains), followed by the primary rainy season, the Kiremt, which dominates from June to September, peaking in July and August at over 200mm per month.
# 
# This climatological profile directly dictates the timing of the parametric insurance evaluation. Because the vast majority of Ethiopia's agricultural production (the Meher harvest) depends on the Kiremt rains, evaluating an agricultural drought index in January or February would be mathematically irrelevant and agriculturally meaningless. Therefore, calculating the 3-month rolling accumulation (SPI-3) at the end of September perfectly captures the critical moisture availability during the core grain-filling and maturation phases of the primary crop cycle.

# In[8]:


# 1. CRITICAL: Sort chronologically so the rolling sum works forwards in time
# We use 'ADM_NAME' instead of coordinates to group the spatial locations
df = df.sort_values(by=['ADM_NAME', 'year', 'month'])

# 2. Calculate the 3-Month Rolling Accumulation PER DISTRICT
# This sums the current month + previous 2 months for each specific district
df['precip_3m'] = df.groupby('ADM_NAME')['precip_mm'].transform(lambda x: x.rolling(3).sum())
# We don't drop NaNs yet, we handle them inside the function so our dataframe size stays consistent


# The Rolling Window (Accumulation):
# 
# 3-month window was chosen, it reflects a full cropping season (essential for agricultural insurance). It captures the moisture deficit within a single growing season. Most parametric models for African smallholders use SPI-3 focused on the "Grain Filling" phase of the crop.
# 
# Funk, C., et al. (2015). The climate hazards group infraRed precipitation with station data—a new hazard record for monitoring extremes.
# 
# McKee, T. B., Doesken, N. J., & Kleist, J. (1993). The relationship of drought frequency and duration to time scales. Proceedings of the 8th Conference on Applied Climatology.

# Spatial Resolution: Since CHIRPS is 0.05° (~5km). Calculate SPI for each pixel and then count how many pixels hit the "-1.5" threshold (this is better for insurance). To handle seasonality, you must treat each calendar month as its own unique statistical "universe." If you want SPI-3, sum the current month + previous 2 months for every pixel. Group by Month: Separate the stack into 12 "buckets"

# Gamma Probability Density Function: rainfall data in Africa is "non-Gaussian" (not a bell curve). The Gamma distribution is used because it starts at zero and is "right-skewed," which perfectly models the occurrence of many small rain events and rare large storms.
# 
# Thom Adjustment (Probability of Zero): handling "Zero Months." In semi-arid regions of Africa, 0mm rainfall is a common data point. We calculate the frequency of zero (q). If q is high, even a tiny amount of rain will result in a positive SPI. If q is zero, that same amount of rain might result in a negative SPI. Thom, H. C. S. (1966), Some Methods of Climatological Analysis.
# 
# In Parametric Insurance, we look for "Exit Points."
# Trigger 1 (Mild): SPI -1.0 (Alert level).
# Trigger 2 (Severe): SPI -1.5 (Payout starts).
# Trigger 3 (Extreme): SPI -2.0 (Full payout).

# In[9]:


# PART 3: THE SPI CALCULATION FUNCTION
def calculate_spi_region(series):
    """
    Fits rainfall data to a Gamma distribution and transforms to SPI.
    Includes the 'Thom Adjustment' for semi-arid regions where 3-month rainfall might be 0.
    """
    # Drop the first 2 months of the time series which are NaN due to rolling sum
    data = series.dropna()
    
    if len(data) < 20: 
        # We need at least 20 years of history to establish a reliable baseline
        return pd.Series(index=series.index, data=np.nan)
        
    n = len(data)
    m = (data == 0).sum()
    q = m / n  # Probability of zero rainfall
    
    # Filter out zeros to fit the gamma distribution
    positive_data = data[data > 0]
    
    if len(positive_data) == 0:
        return pd.Series(index=series.index, data=np.nan)
        
    # Fit Gamma Distribution (Maximum Likelihood Estimation)
    shape, loc, scale = gamma.fit(positive_data, floc=0)
    
    # Calculate Cumulative Distribution Function (CDF)
    cdf = gamma.cdf(data, shape, loc, scale)
    
    # Thom Adjustment for handling historical "zero rain" months
    h_x = q + (1 - q) * cdf
    
    # Fix floating point edge cases so we don't get mathematical infinity errors
    h_x = np.clip(h_x, 0.0001, 0.9999)
    
    # Transform to Standard Normal Distribution (Z-Score / SPI)
    spi = norm.ppf(h_x)
    
    # Return values mapped perfectly back to the original index
    return pd.Series(index=data.index, data=spi)


# In[10]:


# PART 4: APPLY TO EACH DISTRICT & MONTH
print("Calculating SPI-3 for each district...")
# We group by both ADM_NAME AND month to account for seasonality 
# (e.g., creating a baseline for March by looking ONLY at historical Marches for that specific district)
df['SPI_3'] = df.groupby(['ADM_NAME', 'month'])['precip_3m'].transform(calculate_spi_region)


# ### Defining the Threshold Value
# In parametric insurance and climatology, thresholds are defined by the World Meteorological Organization (WMO) standard deviations, combined with how often you want an insurance payout to trigger
# 
# -1.0 (Mild Drought): Happens roughly 1 in 6 years. (Used as an "Alert" or early warning).
# -1.5 (Severe Drought): Happens roughly 1 in 15 years. (This is the industry standard for an insurance payout trigger).
# -2.0 (Extreme Drought): Happens roughly 1 in 40 years. (Full maximum payout).
# 
# Spatio-Temporal Trigger as SPI-3 <= -1.5.

# In[12]:


# PART 5: DEFINE THRESHOLD & SPATIO-TEMPORAL TRIGGER
# In parametric insurance, -1.5 is the standard threshold for Severe Drought (approx 1-in-15 year event)
trigger_threshold = -1.5 # https://climatedataguide.ucar.edu/climate-data/standardized-precipitation-index-spi

# Create a boolean column (True/False) that identifies if the payout trigger was hit
df['is_drought_trigger'] = df['SPI_3'] <= trigger_threshold

# Save the final results to be merged with your shapefile later
df.to_csv(os.path.join(r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk\Drought\Output",  "district_spi_results.csv"), 
          index=False)

print("\nCalculation Complete! Preview of the results:")
print(df[['ADM_NAME', 'year', 'month', 'precip_3m', 'SPI_3', 'is_drought_trigger']].head(10))


# SPI-3 calculates a 3-month rolling accumulation (representing a 90-day crop growing cycle). Therefore, January and February of the first year (2000) return NaN because there is no historical data for November and December 1999 to complete the 3-month window. This proves the rolling calculation is functioning correctly.
# By March, the accumulated 3-month rainfall was 22.36 mm. The algorithm compared this to the historical Gamma distribution for all Marches in Agew Awi and returned an SPI-3 of -1.199. 
# 
# * Agricultural Meaning: An SPI between -1.0 and -1.49 indicates a Moderate Drought. Crops in their early vegetative stage would experience moisture stress.
# * -1.199 is greater than the strict payout threshold of -1.5 (Severe Drought), the is_drought_trigger correctly reads False. No payout is issued, protecting the insurer from paying out on mild, manageable dry spells.
# 
# By October, the accumulated rainfall was 818.31 mm, resulting in an SPI-3 of +1.88.
# * Agricultural Meaning: This represents very wet conditions (nearly 2 standard deviations above the historical mean). The soil profile is fully saturated.

# **To identify the spatio-temporal drought trigger and prove its accuracy, you must move from calculating the index to validating it spatially and temporally.**
# 
# * Temporal Validation: Accurately captures historically documented droughts in Ethiopia (e.g., the 2002/2003 drought or the devastating 2015 El Niño drought) by creating a Drought Heatmap. If the map turns red every single year, your index is too sensitive and the insurance scheme would go bankrupt (Basis Risk: Type I Error). (McKee, T. B., Doesken, N. J., & Kleist, J. (1993). The relationship of drought frequency and duration to time scales. Proceedings of the 8th Conference on Applied Climatology.) established that SPI must be evaluated across specific timescales to monitor frequency and duration. (Viste, E., Korecha, D., & Sorteberg, A. (2013). Recent drought and precipitation tendencies in Ethiopia. Theoretical and Applied Climatology, 112(3-4), 535-551.) documented the historical droughts in Ethiopia

# In[13]:


# Load your completed dataset
df = pd.read_csv(r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk\Drought\Output\district_spi_results.csv")

# TEMPORAL HEATMAP (Historical Validation)
# Critical agricultural month (September - end of Kiremt rains)
critical_month = 9
heatmap_data = df[df['month'] == critical_month].pivot(index='year', columns='ADM_NAME', values='SPI_3')

plt.figure(figsize=(16, 9))
# 'RdBu' colormap. Red = Drought (Negative SPI), Blue = Wet (Positive SPI)
sns.heatmap(heatmap_data, cmap='RdBu', center=0, vmin=-3, vmax=3, 
            annot=False, cbar_kws={'label': 'SPI-3 Value'})
#plt.title(f"Historical Temporal Drought Matrix (Sept)", fontsize=16)
# CAPTION at bottom (perfect placement y=0.02)
plt.figtext(0.5, 0.002, "Figure 3: Historical Temporal Drought Matrix (Sept)",
            ha='center', fontsize=10, style='italic', wrap=True)

plt.ylabel("Year", fontsize=12)
plt.xlabel("Ethiopian Districts", fontsize=12)

# Add a horizontal line or highlight for known drought years (https://doi.org/10.1016/j.wace.2018.10.002)
plt.axhline(y=heatmap_data.index.get_loc(2002), color='red', linewidth=3, linestyle='--', label="2002 El Niño Drought + negative IOD") #Indian Ocean Dipole
#Normal: Warm water EAST Indian Ocean → Normal East Africa rains
#Negative IOD (2002): High pressure on East Africa/Warm water WEST → Ethiopia gets 50-70% LESS rain
#High pressure East Africa → ↓ sinking air → ↓ clouds → ↓ rain (SPI < 0) dry soil → less moisture
#Winds DIVERT EAST → Moisture goes to East Indian Ocean (Indonesia gets floods)
plt.axhline(y=heatmap_data.index.get_loc(2009), color='darkorange', linewidth=3, linestyle='--', label="2009 El Niño Drought") 
# La niña High pressure Horn of Africa → ↓ sinking air → ↓ clouds → ↓ rain (SPI < 0) → dry soil → less moisture → northeasterly winds dominate → complete Belg failure
plt.axhline(y=heatmap_data.index.get_loc(2015), color='black', linewidth=3, linestyle='--', label="2015 La Niña failed Belg")
plt.legend()
plt.tight_layout()
# SAVE FIRST - BEFORE tight_layout()
plt.savefig(r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk\Drought\Output\Ethiopia_Historical_Temporal_Drought_Matrix_September.png", 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()


# **During the 2002 El Niño + Negative IOD event**, anomalous warming shifted from the normal east Indian Ocean location to the western Indian Ocean while persistent high pressure dominated the Horn of Africa (Ethiopia/Eritrea/Somalia), generating sinking air that completely blocked convective processes. Unlike normal years where east Indian Ocean warmth sustains East African low pressure and monsoon rains, the 2002 negative IOD pattern (west Indian Ocean warm + East Africa high pressure) cut Ethiopia's rainfall by 50-70%, replacing moist southwesterlies with dry northeasterlies from Arabia carrying zero moisture. This high pressure wall diverted all Indian Ocean moisture eastward toward Indonesia (causing floods there) while eastern Ethiopian basins recorded record SPI ≤ -2.5 that your red 2002 heatmap line captures perfectly.
# 
# Warm ocean water → heats air above → air rises → LOW pressure → clouds → rain
# 
# Cold ocean water → cools air above → air sinks → HIGH pressure → clear skies → drought
# 
# High pressure East Africa → ↓ sinking air → ↓ clouds → ↓ rain (SPI < -2.5) → dry soil → less moisture. Winds diverted east → moisture went to East Indian Ocean (Indonesia got floods).

# **During the 2009 La Niña event**, (normal conditions) persistent high pressure dominated the Horn of Africa region encompassing Ethiopia, Eritrea, and Somalia, generating sinking air that completely blocked the normal convective processes required for rainfall. This atmospheric blocking prevented the typical moist southwesterly winds from the Indian Ocean from penetrating inland to Ethiopia, replacing them instead with dry northeasterly winds originating from the arid regions of Arabia and Sudan that carried zero moisture across the Horn toward the Indian Ocean. The high pressure system functioned like an impenetrable wall, diverting moisture from the West Indian Ocean's Mozambique Channel southward where it remained over Kenya and Tanzania rather than crossing into the drought-stricken northern Horn, creating a pronounced north-south rainfall dipole that SPI-3 March heatmap captures perfectly with deep blue values in eastern Ethiopian districts.

# **During the 2015 strongest El Niño**, on record, persistent high pressure dominated the Horn of Africa region encompassing Ethiopia, Eritrea, and Somalia, generating sinking air that completely blocked both Belg and Kiremt rainfall seasons. This atmospheric blocking prevented the typical moist southwesterly monsoon winds from the Indian Ocean from penetrating inland to Ethiopia, replacing them instead with dry northeasterly winds originating from the arid regions of Arabia and Sudan that carried zero moisture across the Horn. The extreme El Niño event created unprecedented high pressure strength (SPI ≤ -2.5 across northern/central Ethiopia), functioning like an impenetrable wall that devastated 80-85% of rainfed agriculture 

# * Spatial Distribution: A Country average hides the local disaster. Therefore, data is filtered to focuson Sept (critial month) for a known drought year (2015), and map to identify spatial distribution. It visualizes the "Insurance Payout Events." Areas shaded deep red represent the spatial trigger activating. (Guttman, N. B. (1999). Accepting the standardized precipitation index: a calculation algorithm. Journal of the American Water Resources Association, 35(2), 311-322)

# In[15]:


# PATIO-TEMPORAL TRIGGER MAP (Mapping a payout event)
# Mapping the severe drought of September 2015
# 1. Load your Ethiopia District Shapefile (Update path as needed)
shapefile_path = r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk\Drought\Output\ethiopia\output\ethiopia_districts_ADM_NAME.shp"
districts_map = gpd.read_file(shapefile_path)

# Ensure the column names match for merging (Assume the shapefile has 'NAME_2' for districts)
districts_map = districts_map.rename(columns={'NAME_2': 'ADM_NAME'})

# 2. Filter SPI data for the specific disaster event
drought_event_2015 = df[(df['year'] == 2015) & (df['month'] == 9)]

# 3. Merge the SPI data with the Spatial Map
merged_map = districts_map.merge(drought_event_2015, on='ADM_NAME', how='left')


# In[16]:


# 1. Filter and Merge Data (Assuming df and districts_map are already loaded)
#drought_event_2015 = df[(df['year'] == 2015) & (df['month'] == 9)]
#merged_map = districts_map.merge(drought_event_2015, on='ADM_NAME', how='left')

#fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# --- THE FIX: Plot the base shapefile FIRST ---
# This ensures every single district boundary is drawn, even if data is missing
#districts_map.plot(ax=ax, facecolor='white', edgecolor='black', linewidth=0.5)

# --- THE FIX: Plot the data ON TOP with explicit edgecolors ---
# We use 'BrBG' (Brown-to-Green). vmin=-3 (Brown), vmax=3 (Teal/Green)
#merged_map.plot(column='SPI_3', ax=ax, legend=True, cmap='RdBu', vmin=-3, vmax=3,edgecolor='black', linewidth=0.5, # This forces boundaries on colored districtslegend_kwds={'label': "SPI-3 Value (Brown = Drought, Green = Wet)", 'shrink': 0.7})

# plt.title("Spatial Distribution of Agricultural Drought Trigger: September 2015", fontsize=15)
#plt.figtext(0.5, 0.002, "Figure 4: Spatial Distribution of Agricultural Drought Trigger: September 2015",ha='center', fontsize=10, style='italic', wrap=True)
# SAVE FIRST - BEFORE tight_layout()
#plt.savefig(r"C:\Users\flavi\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk\Drought\Output\Spatial_Distribution_of_Agricultural_Drought_Trigger_September_2015.png", dpi=300, bbox_inches='tight', facecolor='white')
#plt.axis('off') # Hides the lat/lon axis box for a cleaner look
#plt.tight_layout()
#plt.show()


# In[17]:


print("df columns:", df.columns.tolist())
print("ethiopia_districts columns:", ethiopia_districts.columns.tolist())


# In[21]:


import folium
from branca.colormap import LinearColormap
import pandas as pd
import numpy as np

# 1. Prepare data (handle missing values FIRST)
merged_map['SPI_3_rounded'] = merged_map['SPI_3'].round(2).fillna(0)

# 2. Create base map
m = folium.Map(location=[9.145, 40.489], zoom_start=6, tiles='CartoDB positron')

# 3. **FIXED** SPI colormap with NoneType handling
spi_colormap = LinearColormap(
    colors=['darkred', 'red', 'orange', 'yellow', 'lightgreen', 'green'],
    vmin=-3, vmax=3,
    caption='SPI-3 (Standardized Precipitation)'
)

# 4. **FIXED** 2015 layer - safe None handling
def safe_style(feature):
    spi_value = feature['properties'].get('SPI_3_rounded', 0)
    # Convert None/string to float safely
    try:
        spi_value = float(spi_value) if spi_value is not None else 0
    except (ValueError, TypeError):
        spi_value = 0
    
    return {
        'fillColor': spi_colormap(spi_value),
        'color': 'black',
        'weight': 0.8,
        'fillOpacity': 0.7
    }

def safe_highlight(feature):
    return {
        'fillColor': 'blue',
        'fillOpacity': 0.9,
        'weight': 2
    }

# 5. 2015 El Niño layer (default)
folium.GeoJson(
    merged_map.to_json(),
    name="2015 El Niño",
    style_function=safe_style,
    highlight_function=safe_highlight,
    tooltip=folium.GeoJsonTooltip(
        fields=['ADM_NAME', 'SPI_3_rounded'],
        aliases=['District:', 'SPI-3:'],
        localize=True,
        sticky=True
    )
).add_to(m)

# 6. Add toggle layers (2002, 2009)
years = [2002, 2009]
for year in years:
    year_data = merged_map[merged_map['year'] == year].copy()
    if not year_data.empty:
        year_data['SPI_3_rounded'] = year_data['SPI_3'].round(2).fillna(0)
        
        folium.GeoJson(
            year_data.to_json(),
            name=f"{year} Drought",
            style_function=safe_style,
            highlight_function=safe_highlight,
            tooltip=f"District: {{ADM_NAME}}<br>SPI-3 {year}: {{SPI_3_rounded}}"
        ).add_to(m)

# 7. Add controls
spi_colormap.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)

# 8. Title
title_html = '''
<div style="position: fixed; 
            top: 10px; left: 50px; width: 350px; height: 70px; 
            background-color:white; border-radius:10px; border:2px solid grey; 
            z-index:9999; font-size:16px; padding:15px; opacity:0.9">
<b>🌾 Ethiopia SPI-3 Drought Monitor</b><br>
<i>Toggle years | Hover for district values</i>
</div>
'''
m.get_root().html.add_child(folium.Element(title_html))

# 9. Save & display
save_path = r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk\Drought\Output\Ethiopia_SPI_Drought_Comparison.html"
m.save(save_path)
print(f"✅ Saved: {save_path}")
m


# Trigger Accuracy Validation (Confusion Matrix):
# Compare your SPI-3 (Rainfall only) against an independent standard like SPEI. (Vicente-Serrano et al. (2010) introduced SPEI, proving that temperature exacerbates agricultural drought. By building a confusion matrix, we can calculate model's Hit Rate (Recall) and Precision.

# ### Validating the Trigger with SPEI-3 Data
# To validate SPI-3 trigger, we treat SPEI-3 data (from GEE) as "Ground Truth" or benchmark. Why? Because SPEI includes temperature and evapotranspiration, making it a more comprehensive measure of actual agricultural drought (soil moisture stress) than SPI (which only measures rainfall).
# 
# **If we define a severe drought as an index value <= -1.5, how well did our SPI-only model predict the "true" drought conditions captured by SPEI?**

# In[23]:


# 1. Load your Data
# df_spei is the GEE data you downloaded. Ensure it has columns: ['ADM_NAME', 'year', 'month', 'SPEI_3']
df_spei = pd.read_csv("C:\\Users\\FlawiyaShirishMore\\OneDrive - Africa Specialty Risks Ltd\\ASR-Parametric_Research_Study\\africa_risk\\Drought\\Output\\ethiopia\\content\\ethiopia_spei03_master_cleaned.csv")


# In[25]:


import pandas as pd
import numpy as np

# 1. Load data (your files)
df = pd.read_csv(r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk\Drought\Output\district_spi_results.csv")
df_spei = pd.read_csv(r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk\Drought\Output\ethiopia\content\ethiopia_spei03_master_cleaned.csv")

# 2. Print columns FIRST
print("df columns:", df.columns.tolist())
print("df_spei columns:", df_spei.columns.tolist())

# 3. NOW create val_df (MERGE)
df_spi = df[['ADM_NAME', 'year', 'month', 'SPI_3', 'is_drought_trigger']].copy()
df_spei_small = df_spei[['ADM_NAME', 'year', 'month', 'spei_03']].copy()

val_df = pd.merge(df_spi, df_spei_small, on=['ADM_NAME', 'year', 'month'], how='inner')

# 4. NOW print val_df columns (no more NameError!)
print("val_df (merged) columns:", val_df.columns.tolist())
print("val_df shape:", val_df.shape)
print(val_df.head())


# In[26]:


# Rename to match your SPI naming convention
val_df = val_df.rename(columns={'spei_03': 'SPEI_3'})

# Now dropna works perfectly
val_df = val_df.dropna(subset=['SPI_3', 'SPEI_3'])


# In[27]:


# 3. Define the Trigger Logic (<= -1.5 for Severe Drought)
trigger_threshold = -1.5
val_df['SPI_Triggered'] = val_df['SPI_3'] <= trigger_threshold
val_df['SPEI_Triggered'] = val_df['SPEI_3'] <= trigger_threshold


# In[28]:


# 4. Statistical Validation (Continuous Variables)
correlation = val_df['SPI_3'].corr(val_df['SPEI_3'])
print(f"Overall Pearson Correlation (SPI vs SPEI): {correlation:.3f}")


# The Pearson correlation coefficient between SPI-3 (Rainfall-only) and SPEI-3 (Precipitation + Evapotranspiration) is 0.429.
# While there is a positive relationship (as expected, since rainfall is a core component of both), a correlation of 0.43 is considered only moderate.
# 
# This statistically proves that rainfall alone is an insufficient proxy for agricultural drought in Ethiopia. The remaining variance (the "noise" pushing the R-value down) is largely driven by temperature and evapotranspiration. In the Horn of Africa, heatwaves can desiccate crops even during months with "average" rainfall. Because SPI ignores this heat factor, it frequently diverges from the actual soil moisture reality captured by SPEI.

# In[29]:


# 5. Insurance Trigger Validation (Categorical / Binary Variables)
# Here, SPEI is the "True" label, SPI is the "Predicted" label
print("\n--- Trigger Validation (Confusion Matrix) ---")
cm = confusion_matrix(val_df['SPEI_Triggered'], val_df['SPI_Triggered'])
# cm[0,0] = True Negative (Neither triggered)
# cm[0,1] = False Positive (SPI triggered, SPEI didn't - False Alarm)
# cm[1,0] = False Negative (SPEI triggered, SPI missed it - Missed Payout)
# cm[1,1] = True Positive (Both triggered - Successful Payout)

# Calculate Metrics
hits = cm[1, 1]
misses = cm[1, 0]
false_alarms = cm[0, 1]
true_negatives = cm[0, 0]

print(f"Correct Payouts (Hits): {hits}")
print(f"Missed Payouts (Type II Error): {misses}  <-- Dangerous for farmers")
print(f"False Alarms (Type I Error): {false_alarms} <-- Dangerous for insurers")
print(f"Correct Non-Payouts: {true_negatives}")


# In parametric insurance, the Confusion Matrix represents Basis Risk. Your results reveal extreme vulnerabilities in using SPI-3 as an insurance trigger:
# * **Hits / True Positives (130):** In 130 instances, both models agreed that a severe drought occurred. This is the intended function of the insurance policy.
# * **Missed Payouts / False Negatives (162):** This represents Type II Error, which is a catastrophe for the farmer. In 162 instances, SPEI flagged a severe drought (meaning high heat likely caused rapid evaporation and crop failure), but SPI completely missed it because rainfall amounts were technically "normal." If this model went to market, farmers would have lost their crops 162 times without receiving an insurance payout, entirely destroying trust in the product.
# * **False Alarms / False Positives (615):** This represents Type I Error, which is a catastrophe for the insurer. In 615 instances, SPI triggered a payout (due to low rainfall), but SPEI indicated that conditions were actually normal/mild (likely because temperatures were cool, so the crops did not experience severe water stress). If the insurer pays out on these 615 false alarms, the insurance pool will rapidly go bankrupt.
# * **Correct Non-Payouts / True Negatives (9,961):** Both models correctly agreed that conditions were normal the vast majority of the time, which heavily skews the overall "Accuracy" metric.

# In[30]:


print("\n---------------- Classification Report ----------------")
# This gives you Precision, Recall, and F1-Score automatically
print(classification_report(val_df['SPEI_Triggered'], val_df['SPI_Triggered'], target_names=['Normal/Mild', 'Severe Drought']))


# **Deconstructing the Classification Report:**
# Because non-drought months drastically outnumber drought months, overall "Accuracy" (0.93) is a highly misleading metric. Evaluate the model based on Precision and Recall for the "Severe Drought" class:
# * Recall (0.45): Out of all the true severe droughts (according to SPEI), SPI model only successfully identified 45% of them. It missed more than half.
# * Precision (0.17): This is the most damning metric for the SPI model. When SPI drops below -1.5 and triggers a payout, it is only correct 17% of the time. The other 83% of the time, it is issuing a payout for a "False Alarm."
# * F1-Score (0.25): The harmonic mean of precision and recall. A score of 0.25 out of 1.0 indicates very poor performance for the SPI model as a predictive trigger.
# 
# (Clement, K. Y., et al. (2018). Basis risk in index-based agricultural insurance: Evaluating the effectiveness of different drought indices. Agricultural and Forest Meteorology, 253, 11-20. )

# In[32]:


# 6. Visualizing the Validation
plt.figure(figsize=(8, 8))
plt.hexbin(val_df['SPEI_3'], val_df['SPI_3'], gridsize=40, cmap='YlOrRd', mincnt=1)
plt.plot([-3, 3], [-3, 3], color='black', linestyle='--', label='Perfect Agreement (1:1)')
plt.axvline(trigger_threshold, color='blue', alpha=0.5, linestyle=':', label='SPEI Trigger Threshold')
plt.axhline(trigger_threshold, color='red', alpha=0.5, linestyle=':', label='SPI Trigger Threshold')
plt.xlabel('Ground Truth: SPEI-3 (Includes Temperature)')
plt.ylabel('Modeled: SPI-3 (Rainfall Only)')
#plt.title(f'Trigger Agreement Validation (R = {correlation:.2f})')
plt.figtext(0.5, 0.002, "Figure 5: SPI-3 vs SPEI-3 Validation for Ethiopian District Drought Triggers.",
            ha='center', fontsize=10, style='italic', wrap=True)
plt.colorbar(label='Density of Months')
plt.legend()
# SAVE FIRST - BEFORE tight_layout()
plt.savefig(r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk\Drought\Output\SPI-3 vs SPEI-3 Validation for Ethiopian.png", 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()


# **The Hexagonal Density:** The plot maps modeled SPI against ground-truth SPEI. The dark red core at the center (0, 0) shows that both models largely agree during normal, non-extreme conditions.
# The Quadrants: The blue dotted line (SPEI Trigger) and red dotted line (SPI Trigger) divide the plot into four quadrants that match your confusion matrix:
# Bottom-Left (Hits): The points below -1.5 on both axes.
# Top-Left (Missed Payouts): The points where SPEI is < -1.5 (left of the blue line) but SPI is > -1.5 (above the red line). Look at how many points exist here—these are "flash droughts" caused by heat, completely invisible to the SPI model.
# Bottom-Right (False Alarms): The heavy cluster of data points below the red line but right of the blue line. Here, SPI registers a severe drought (low rain), but SPEI disagrees.

# **Conclusion:**
# A purely precipitation-based index (SPI) generates an unacceptable level of Basis Risk for parametric agricultural insurance in East Africa.
# You should conclude that while SPI is historically standard, it is fundamentally flawed in an era of climate change, where rising ambient temperatures are primary drivers of crop failure. The extremely low precision (17%) and high false alarm rate (615 instances) mean an SPI-3 index at a -1.5 threshold is commercially unviable. Moving forward, the insurance industry and government safety nets in Ethiopia must transition to indices that incorporate evapotranspiration, such as SPEI, to ensure financial solvency and genuinely protect farmers.

# **Pros:**
# Low Data Requirements: It only requires historical rainfall data. In developing nations with sparse meteorological networks, rainfall is often the only reliable data available (or accessible via satellite proxies like CHIRPS).
# Multi-Scalar Flexibility: Because it can be calculated on 1, 3, or 6-month scales, it can be perfectly tailored to match the phenological growth stages of specific crops (e.g., SPI-3 for short-cycle maize).
# Standardization: Because it transforms data into standard deviations (the Z-score), it allows you to compare drought severity across completely different climate zones (e.g., the arid Afar region vs. the wet Amhara highlands) on the exact same scale.
# 
# **Cons:**
# The "Temperature Blindspot": It assumes drought is purely a lack of rain. It completely ignores atmospheric water demand (Evapotranspiration). A month with average rain but 5°C above-average heat will kill crops, but SPI will say conditions are "normal."
# Assumes Climate Stationarity: SPI relies on fitting historical data to a curve. In a rapidly changing climate, the "historical baseline" is no longer a reliable predictor of future probabilities.
# Ignores Soil and Crop Types: SPI treats the ground like concrete. It doesn't know if the rain fell on sandy soil (which drains instantly) or clay (which retains moisture). It also doesn't account for whether the farmer planted drought-resistant sorghum or water-heavy maize.
