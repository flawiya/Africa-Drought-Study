import os
import numpy as np
import geopandas as gpd
import pandas as pd
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
# Logic to jump from core_analysis up to the project root then into data
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Exact File Paths based on your setup
GADM_PATH = os.path.join(DATA_DIR, "africa_agricultural_domain_2019.shp")
GEOGLAM_PATH = os.path.join(DATA_DIR, "GEOGLAM_CM4EW_Calendars_V1.4.shp")
ERA5_PATH = os.path.join(DATA_DIR, "Africa_Agri_districts_ERA5_LAND_DAILY_AGGR_2000_2026_timeseries.csv")

# Constants
MIN_OBS_FOR_FIT = 10
# We define our list of thresholds here
THRESHOLDS = {
    'Neg1_0': -1.0,
    'Neg1_5': -1.5,
    'Neg2_0': -2.0
}

# ---------------------------------------------------------------------------
# CORE FUNCTIONS
# ---------------------------------------------------------------------------

def load_shapefiles():
    print(f"STEP 1: Loading shapefiles from {DATA_DIR}")
    gdf_districts = gpd.read_file(GADM_PATH)
    gdf_districts["ADM_NAME"] = gdf_districts["ADM_NAME"].astype(str).str.strip().str.upper()
    gdf_geoglam = gpd.read_file(GEOGLAM_PATH)
    return gdf_districts, gdf_geoglam

def filter_maize1(gdf_geoglam):
    print("STEP 2: Filtering GEOGLAM to Maize 1")
    return gdf_geoglam[gdf_geoglam["crop"] == "Maize 1"].copy()

def join_districts_to_maize1(gdf_districts, gdf_maize1):
    print("STEP 3: Spatial join via centroids")
    gdf_centroids = gdf_districts.to_crs(epsg=3857).copy()
    gdf_centroids["geometry"] = gdf_centroids.centroid
    gdf_centroids = gdf_centroids.to_crs(epsg=4326)
    if gdf_maize1.crs != gdf_centroids.crs:
        gdf_maize1 = gdf_maize1.to_crs(epsg=4326)
    districts_maize = gpd.sjoin(gdf_centroids, gdf_maize1, how="inner", predicate="within")
    return districts_maize[(districts_maize["planting"] > 0) & (districts_maize["harvest"] > 0)].copy()

def calculate_risk_windows(valid_maize):
    print("STEP 4: Calculating risk windows")
    def _calc_risk(row):
        p, h = row["planting"], row["endofseaso"]
        duration = (h - p) if h >= p else (365 - p) + h
        delay = duration * 0.25
        return pd.Series([int(round((p + delay) % 365)), h], index=["risk_start_doy", "risk_end_doy"])
    valid_maize[["risk_start_doy", "risk_end_doy"]] = valid_maize.apply(_calc_risk, axis=1)
    return valid_maize

def load_and_merge_era5(valid_maize):
    print("STEP 5: Loading ERA5 soil moisture & merging")
    df_era5 = pd.read_csv(ERA5_PATH)
    df_era5["feature_id"] = df_era5["feature_id"].astype(str).str.strip().str.upper()
    df_merged = df_era5.merge(valid_maize[["ADM_NAME", "risk_start_doy", "risk_end_doy"]], left_on="feature_id", right_on="ADM_NAME", how="inner")
    
    mask = ((df_merged["risk_start_doy"] <= df_merged["risk_end_doy"]) & (df_merged["doy"] >= df_merged["risk_start_doy"]) & (df_merged["doy"] <= df_merged["risk_end_doy"])) | \
           ((df_merged["risk_start_doy"] > df_merged["risk_end_doy"]) & ((df_merged["doy"] >= df_merged["risk_start_doy"]) | (df_merged["doy"] <= df_merged["risk_end_doy"])))
    
    df_filtered = df_merged[mask].copy()
    df_filtered["crop_year"] = df_filtered["year"]
    df_filtered.loc[(df_filtered["risk_start_doy"] > df_filtered["risk_end_doy"]) & (df_filtered["doy"] <= df_filtered["risk_end_doy"]), "crop_year"] -= 1
    return df_filtered

def compute_daily_gamma_ssi(df_filtered):
    print("STEP 6: Computing daily Gamma-CDF SSI (heavy processing)...")
    df_filtered = df_filtered.copy()
    df_filtered["SSI"] = np.nan
    groups = df_filtered.groupby(["feature_id", "doy"])

    for (district, doy), group in groups:
        values = group["volumetric_soil_water_layer_2"].dropna().values
        if len(values) < MIN_OBS_FOR_FIT: continue
        nonzero = values[values > 0]
        q_zero = (len(values) - len(nonzero)) / len(values)
        if len(nonzero) < 5: continue
        try:
            alpha, loc, beta = scipy_stats.gamma.fit(nonzero, floc=0)
            idx = group.index
            sm_vals = df_filtered.loc[idx, "volumetric_soil_water_layer_2"].values
            p = np.where(sm_vals <= 0, q_zero / 2.0, q_zero + (1.0 - q_zero) * scipy_stats.gamma.cdf(sm_vals, alpha, loc=0, scale=beta))
            df_filtered.loc[idx, "SSI"] = scipy_stats.norm.ppf(np.clip(p, 0.001, 0.999))
        except: continue
    return df_filtered.dropna(subset=["SSI"])

def aggregate_multi_threshold(df_filtered):
    print(f"STEP 7: Aggregating for multiple thresholds: {list(THRESHOLDS.values())}")
    
    # Calculate binary drought columns for each threshold
    for label, value in THRESHOLDS.items():
        df_filtered[f'is_drought_{label}'] = (df_filtered['SSI'] <= value).astype(int)
    
    # Define how to aggregate (Sum the binary drought markers)
    agg_dict = {f'is_drought_{label}': 'sum' for label in THRESHOLDS.keys()}
    
    # Group by district and crop year
    df_annual = df_filtered.groupby(['feature_id', 'crop_year']).agg(agg_dict).reset_index()
    
    # Rename columns to be more readable
    rename_dict = {f'is_drought_{label}': f'Drought_Days_SSI_{abs(value)}' for label, value in THRESHOLDS.items()}
    rename_dict['crop_year'] = 'year'
    df_annual = df_annual.rename(columns=rename_dict)
    
    # Final filter for valid years
    return df_annual[(df_annual["year"] >= 2000) & (df_annual["year"] <= 2025)]

# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    gdf_districts, gdf_geoglam = load_shapefiles()
    gdf_maize1 = filter_maize1(gdf_geoglam)
    valid_maize = join_districts_to_maize1(gdf_districts, gdf_maize1)
    valid_maize = calculate_risk_windows(valid_maize)
    df_filtered = load_and_merge_era5(valid_maize)
    
    # Heavy processing happens here
    df_with_ssi = compute_daily_gamma_ssi(df_filtered)
    
    # Multi-threshold aggregation
    df_annual = aggregate_multi_threshold(df_with_ssi)

    # Save the final file
    output_name = "Africa_Maize_Drought_Annual_MultiThreshold.csv"
    df_annual.to_csv(output_name, index=False)
    
    print(f"\n✅ Success! Multi-threshold file saved as: {output_name}")
    print("Columns included:")
    print("- Drought_Days_SSI_1.0 (Moderate)")
    print("- Drought_Days_SSI_1.5 (Severe)")
    print("- Drought_Days_SSI_2.0 (Extreme)")