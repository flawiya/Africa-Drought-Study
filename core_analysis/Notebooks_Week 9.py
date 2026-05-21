# %% [markdown]
# # African Maize Drought Analysis: Gamma-SSI Pipeline
# ### Developed for Africa Specialty Risks Ltd (ASR)
# 
# **Author:** Flawiya Shirish More  
# **Date:** May 21, 2026  
# 
# ## 🔬 Scientific Justification
# Traditional drought indices like the SPI (Precipitation) often fail to account for the actual "Supply" of water available to a crop's root zone. This pipeline implements the **Standardized Soil Moisture Index (SSI)** using a **Gamma-CDF transformation**.
# 
# ### Key Methodological Choices:
# 1. **Root Zone Focus**: We utilize ERA5-Land Layer 2 (7-28cm) soil moisture, which corresponds to the critical depth for maize during flowering and grain-filling.
# 2. **Gamma Distribution**: Unlike precipitation, soil moisture is often non-Gaussian. Fitting a 3-parameter Gamma distribution (Shape, Scale, and Probability of Zero) allows for a more robust standardization across diverse African soil types (AghaKouchak, 2014).
# 3. **The 25% establishment Rule**: The first 25% of the growing season is excluded from the "Risk Window" because drought during early establishment is less impactful on final yield than stress during reproductive phases.
# 
# ## 📚 Primary References
# - **AghaKouchak, A. (2014)**: A multivariate approach for drought monitoring and analysis using standardized indices. *HESS*.
# - **McKee, T. B., et al. (1993)**: The relationship of drought frequency and duration to time scales.
# - **Svoboda, M., et al. (2002)**: The Drought Monitor. *BAMS*.

# %%
import os
import json
import math
import numpy as np
import geopandas as gpd
import pandas as pd
from scipy import stats as scipy_stats
from pathlib import Path
import plotly.express as px
import matplotlib.pyplot as plt

# =============================================================================
# 1. CONFIGURATION & PATH HANDLING
# =============================================================================

# We use Raw Strings (r"") to handle Windows backslashes and spaces safely.
BASE_DIR = Path(r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk")
DATA_DIR = BASE_DIR / "Drought" / "data"
OUTPUT_DIR = BASE_DIR / "Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Specific Data Paths
GADM_PATH = DATA_DIR / "africa_agricultural_domain_2019" / "africa_agricultural_domain_2019.shp"
GEOGLAM_PATH = DATA_DIR / "GEOGLAM_CM4EW_Calendars_V1.4" / "GEOGLAM_CM4EW_Calendars_V1.4.shp"
ERA5_PATH = DATA_DIR / "Africa_Agri_districts_ERA5_LAND_DAILY_AGGR_2000_2026_timeseries.csv"

# Actuarial Parameters
SSI_THRESHOLD = -1.0  # WMO Moderate Drought onset
MIN_OBS_FOR_FIT = 15  # Minimum years of data required to fit a Gamma curve

# %% [markdown]
# ## Step 1 & 2: Data Loading and Crop Filtering
# We filter the GEOGLAM crop calendar specifically to 'Maize 1'. Maize 1 represents the primary cereal crop cycle across the majority of sub-Saharan Africa.

# %%
def load_and_filter_data():
    print("STEP 1: Loading GADM Districts & GEOGLAM Calendars...")
    gdf_districts = gpd.read_file(GADM_PATH)
    gdf_geoglam = gpd.read_file(GEOGLAM_PATH)
    
    # Standardize names
    gdf_districts["ADM_NAME"] = gdf_districts["ADM_NAME"].astype(str).str.strip().str.upper()
    
    print("STEP 2: Filtering GEOGLAM to 'Maize 1'...")
    gdf_maize1 = gdf_geoglam[gdf_geoglam["crop"] == "Maize 1"].copy()
    
    return gdf_districts, gdf_maize1

# %% [markdown]
# ## Step 3 & 4: Spatial Joining & Risk Window Logic
# A spatial join is performed using **District Centroids** to ensure each administrative unit is mapped to exactly one crop calendar zone. 
# 
# **Risk Window Logic**: The "established" phase of the crop is more resilient. We calculate the duration of the season and skip the first 25% to focus on the **Reproductive and Maturity phases** (the 75% risk window).

# %%
def process_risk_windows(gdf_districts, gdf_maize1):
    print("STEP 3: Performing Spatial Join via Centroids...")
    gdf_centroids = gdf_districts.to_crs(epsg=3857).copy()
    gdf_centroids["geometry"] = gdf_centroids.centroid
    gdf_centroids = gdf_centroids.to_crs(epsg=4326)
    
    # Inner join to keep only agricultural districts
    valid_maize = gpd.sjoin(gdf_centroids, gdf_maize1.to_crs(epsg=4326), how="inner", predicate="within")

    print("STEP 4: Calculating Crop-Specific Risk Windows (Establishing 25% Delay)...")
    def _calc_risk(row):
        p, h = row["planting"], row["endofseaso"]
        duration = (h - p) if h >= p else (365 - p) + h
        delay = duration * 0.25
        risk_start = int(round((p + delay) % 365))
        if risk_start == 0: risk_start = 365
        return pd.Series([risk_start, h], index=["risk_start_doy", "risk_end_doy"])

    valid_maize[["risk_start_doy", "risk_end_doy"]] = valid_maize.apply(_calc_risk, axis=1)
    return valid_maize

# %% [markdown]
# ## Step 5 & 6: Gamma-SSI Calculation
# This is the core engine of the research. 
# 1. We group soil moisture by **(District, Day of Year)**.
# 2. We handle the **"Zero-Mass" problem**: Soil can reach a physical lower limit where it stays at the same low value. 
# 3. We fit the Gamma distribution and transform to the **Standard Normal Space**.

# %%
def compute_gamma_ssi(valid_maize):
    print("STEP 5: Loading ERA5 Soil Moisture & Merging with Risk Windows...")
    df_era5 = pd.read_csv(ERA5_PATH)
    df_era5["feature_id"] = df_era5["feature_id"].astype(str).str.strip().str.upper()

    # Merge ERA5 with the Risk Windows calculated in Step 4
    df_merged = df_era5.merge(
        valid_maize[["ADM_NAME", "risk_start_doy", "risk_end_doy"]],
        left_on="feature_id", right_on="ADM_NAME", how="inner"
    )

    # Filter to only include days WITHIN the crop season
    mask_std = (df_merged["risk_start_doy"] <= df_merged["risk_end_doy"]) & \
               (df_merged["doy"] >= df_merged["risk_start_doy"]) & \
               (df_merged["doy"] <= df_merged["risk_end_doy"])
    mask_wrp = (df_merged["risk_start_doy"] > df_merged["risk_end_doy"]) & \
               ((df_merged["doy"] >= df_merged["risk_start_doy"]) | (df_merged["doy"] <= df_merged["risk_end_doy"]))
    
    df_filtered = df_merged[mask_std | mask_wrp].copy()

    print("STEP 6: Computing Daily Gamma-CDF SSI...")
    df_filtered["SSI"] = np.nan
    groups = df_filtered.groupby(["feature_id", "doy"])

    for (dist, doy), group in groups:
        vals = group["volumetric_soil_water_layer_2"].dropna().values
        if len(vals) < MIN_OBS_FOR_FIT: continue

        nonzero = vals[vals > 0]
        q_zero = (len(vals) - len(nonzero)) / len(vals)

        try:
            # Fit Gamma distribution (standard for SSI research)
            alpha, loc, beta = scipy_stats.gamma.fit(nonzero, floc=0)
            
            p = q_zero + (1.0 - q_zero) * scipy_stats.gamma.cdf(group["volumetric_soil_water_layer_2"], alpha, loc=0, scale=beta)
            p = np.clip(p, 0.001, 0.999)
            df_filtered.loc[group.index, "SSI"] = scipy_stats.norm.ppf(p)
        except:
            continue

    return df_filtered.dropna(subset=["SSI"])

# %% [markdown]
# ## Step 7 & 8: Aggregation & Insurance Severity Bins
# We aggregate daily failures into an annual "Drought Day Count". We then classify them into the **US Drought Monitor (Svoboda et al., 2002)** standard categories (D1-D5).

# %%
def aggregate_and_classify(df_filtered):
    print("STEP 7: Aggregating to Annual Drought-Day Counts...")
    df_filtered["is_drought"] = (df_filtered["SSI"] <= -1.5).astype(int) # Severe trigger
    
    df_annual = df_filtered.groupby(["feature_id", "year"])["is_drought"].sum().reset_index()
    df_annual.rename(columns={"is_drought": "Drought_Days"}, inplace=True)

    print("STEP 8: Classifying Drought Severity (WMO Standard)...")
    bins = [0, 1, 10, 20, 35, 55, 365]
    labels = ["No Drought", "D1-Abnormally Dry", "D2-Moderate", "D3-Severe", "D4-Extreme", "D5-Exceptional"]
    df_annual["Category"] = pd.cut(df_annual["Drought_Days"], bins=bins, labels=labels, include_lowest=True)
    
    return df_annual

# %% [markdown]
# ## Final Step: Reporting and Mapping
# Generates an animated map showing the "Drought Pulse" of Africa over 25 years.

# %%
def main():
    # Execute Pipeline
    districts, maize1 = load_and_filter_data()
    valid_maize = process_risk_windows(districts, maize1)
    df_ssi = compute_gamma_ssi(valid_maize)
    df_final = aggregate_and_classify(df_ssi)
    
    # Save results
    output_csv = OUTPUT_DIR / "African_Maize_Drought_History_2000_2025.csv"
    df_final.to_csv(output_csv, index=False)
    
    print(f"\n✅ PIPELINE COMPLETE.")
    print(f"📍 Final records saved to: {output_csv}")
    
    # Summary Table for the README
    print("\n--- Summary Distribution (2000-2025) ---")
    print(df_final["Category"].value_counts(normalize=True).sort_index() * 100)

if __name__ == "__main__":
    main()