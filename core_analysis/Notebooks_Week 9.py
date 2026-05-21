#!/usr/bin/env python
# coding: utf-8

# new-file.py — Gamma-SSI Drought Analysis Pipeline
# ==================================================
# Full pipeline from raw data to daily SSI values for African maize districts.
# 
# STEPS:
#   1. Load GADM districts + GEOGLAM V1.4 crop calendar shapefiles
#   2. Filter GEOGLAM to Maize 1; spatial-join with districts via centroids
#   3. Calculate risk window per district (skip first 25% of season)
#   4. Load ERA5 daily soil moisture; merge with district risk windows
#   5. Compute daily SSI via Gamma-CDF transformation per (district, DOY)
#   6. Count drought days per district per crop-year
# 
# METHOD — Gamma-CDF SSI (per AghaKouchak 2014, McKee et al. 1993):
#   For each (district, day-of-year):
#     1. Collect all years of daily soil moisture for that DOY
#     2. Fit a Gamma distribution (shape α, scale β) to non-zero values
#     3. For each observation: p = q_zero + (1 − q_zero) × Gamma_CDF(x; α, β)
#     4. SSI = Φ⁻¹(p)  (inverse standard normal)
# 
#   This ensures SSI = −1.5 universally means "6.7th percentile event"
#   regardless of the local soil moisture distribution shape.
# 
# References:
#   - McKee, Doesken & Kleist (1993), 8th Conf. Applied Climatology
#   - AghaKouchak (2014), HESS, 18(7), 2515–2526
#   - GEOGLAM CM4EW Calendars V1.4 (Zenodo)
#   - GADM Agricultural Domain 2019

# In[33]:


import os
import json
import numpy as np
import geopandas as gpd
import pandas as pd
from scipy import stats as scipy_stats
import plotly.express as px
import matplotlib.pyplot as plt

# Configuration
#BASE_DIR = os.getcwd()
GADM_PATH = os.path.join(r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk\Drought\data\africa_agricultural_domain_2019\africa_agricultural_domain_2019.shp")
GEOGLAM_PATH = os.path.join(r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk\Drought\data\GEOGLAM_CM4EW_Calendars_V1.4\GEOGLAM_CM4EW_Calendars_V1.4.shp")
ERA5_PATH = os.path.join(r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk\Drought\data\Africa_Agri_districts_ERA5_LAND_DAILY_AGGR_2000_2026_timeseries.csv")

SSI_THRESHOLD = -1.0  # WMO Moderate Drought onset
MIN_OBS_FOR_FIT = 10


# In[34]:


def load_shapefiles():
    """
    Load both shapefiles and standardise the district name column.

    Returns:
        gdf_districts: GeoDataFrame of GADM agricultural districts
        gdf_geoglam:   GeoDataFrame of GEOGLAM crop calendar polygons
    """
    print("=" * 70)
    print("STEP 1: Loading shapefiles")
    print("=" * 70)

    # Load GADM districts
    print(f"\n  Loading GADM districts from:\n    {GADM_PATH}")
    gdf_districts = gpd.read_file(GADM_PATH)

    # Standardise district names to uppercase (for consistent matching later)
    gdf_districts["ADM_NAME"] = (
        gdf_districts["ADM_NAME"].astype(str).str.strip().str.upper()
    )
    print(f"  → {len(gdf_districts)} district polygons loaded")
    print(f"  → CRS: {gdf_districts.crs}")
    print(f"  → Columns: {gdf_districts.columns.tolist()}")

    # Load GEOGLAM crop calendar
    print(f"\n  Loading GEOGLAM V1.4 from:\n    {GEOGLAM_PATH}")
    gdf_geoglam = gpd.read_file(GEOGLAM_PATH)
    print(f"  → {len(gdf_geoglam)} calendar polygons loaded")
    print(f"  → CRS: {gdf_geoglam.crs}")
    print(f"  → Crops available: {sorted(gdf_geoglam['crop'].unique())}")

    return gdf_districts, gdf_geoglam


# In[35]:


# Call the function and store the results
gdf_districts, gdf_geoglam = load_shapefiles()

# Optional: View the first 5 rows of the district data to confirm it loaded
gdf_districts.head()


# In[36]:


# ---------------------------------------------------------------------------
# STEP 2: Filter GEOGLAM to Maize 1 only
# ---------------------------------------------------------------------------
def filter_maize1(gdf_geoglam):
    """
    Filter the GEOGLAM calendar to only 'Maize 1' entries.

    GEOGLAM contains calendars for many crops (Wheat, Rice, Maize 1, Maize 2, etc.).
    We isolate Maize 1 because it is the primary maize growing season across Africa.

    Returns:
        gdf_maize1: GeoDataFrame containing only Maize 1 calendar polygons
    """
    print("\n" + "=" * 70)
    print("STEP 2: Filtering GEOGLAM to Maize 1")
    print("=" * 70)

    # Filtering the data
    gdf_maize1 = gdf_geoglam[gdf_geoglam["crop"] == "Maize 1"].copy()

    # Printing detailed logs
    print(f"\n  Total GEOGLAM polygons: {len(gdf_geoglam)}")
    print(f"  Maize 1 polygons:      {len(gdf_maize1)}")
    print(f"\n  Maize 1 calendar columns:")
    print(f"    planting    — DOY when planting begins (1–365)")
    print(f"    vegetative  — DOY when vegetative growth begins")
    print(f"    harvest     — DOY when harvest begins")
    print(f"    endofseaso  — DOY when season ends")
    
    print(f"\n  Countries covered by Maize 1:")
    countries = sorted(gdf_maize1["country"].unique())
    for c in countries:
        n_regions = gdf_maize1[gdf_maize1["country"] == c]["region"].nunique()
        print(f"    • {c} ({n_regions} regions)")

    return gdf_maize1


# In[37]:


# Call the function using the data loaded in Step 1
gdf_maize1 = filter_maize1(gdf_geoglam)

# Optional: Preview the filtered data
gdf_maize1.head()


# In[38]:


# ---------------------------------------------------------------------------
# STEP 3: Compute district centroids and spatial join with Maize 1
# ---------------------------------------------------------------------------
def join_districts_to_maize1(gdf_districts, gdf_maize1):
    """
    Spatially join GADM districts to Maize 1 calendar zones using centroids.

    Why centroids?
      - District polygons may overlap multiple GEOGLAM zones
      - Using the centroid gives a single, deterministic assignment
      - Centroid is computed in EPSG:3857 (metric) for accuracy, then
        reprojected back to WGS84 (EPSG:4326) for the spatial join

    After the join, we filter to districts with VALID calendar data:
      - planting > 0  (zero means "no data" in GEOGLAM)
      - harvest > 0

    Returns:
        valid_maize: GeoDataFrame of districts with valid Maize 1 calendar data
    """
    print("\n" + "=" * 70)
    print("STEP 3: Spatial join — districts to Maize 1 calendar via centroids")
    print("=" * 70)

    # Compute centroids in a projected CRS (metres) for geometric accuracy
    print("\n  Computing district centroids (EPSG:3857 → EPSG:4326) …")
    gdf_centroids = gdf_districts.to_crs(epsg=3857).copy()
    gdf_centroids["geometry"] = gdf_centroids.centroid
    gdf_centroids = gdf_centroids.to_crs(epsg=4326)

    # Ensure GEOGLAM is also in WGS84
    if gdf_maize1.crs != gdf_centroids.crs:
        gdf_maize1 = gdf_maize1.to_crs(epsg=4326)

    # Spatial join: find which Maize 1 polygon contains each district centroid
    print("  Performing spatial join (centroid 'within' Maize 1 polygon) …")
    districts_maize = gpd.sjoin(
        gdf_centroids, gdf_maize1, how="inner", predicate="within"
    )

    print(f"\n  Results:")
    print(f"    Total GADM districts:               {len(gdf_districts)}")
    print(f"    Districts matched to Maize 1:       {len(districts_maize)}")
    print(f"    Districts NOT matched (no Maize 1): {len(gdf_districts) - len(districts_maize)}")

    # Filter out districts where GEOGLAM has placeholder zeros
    valid_maize = districts_maize[
        (districts_maize["planting"] > 0) & (districts_maize["harvest"] > 0)
    ].copy()

    n_invalid = len(districts_maize) - len(valid_maize)
    print(f"\n    Districts with valid planting/harvest dates: {len(valid_maize)}")
    print(f"    Districts with zero (placeholder) dates:     {n_invalid}")

    # Summary of calendar dates
    print(f"\n  Calendar date summary (DOY) for valid districts:")
    print(f"    Planting    — min: {valid_maize['planting'].min()}, "
          f"max: {valid_maize['planting'].max()}, "
          f"mean: {valid_maize['planting'].mean():.0f}")
    print(f"    Harvest     — min: {valid_maize['harvest'].min()}, "
          f"max: {valid_maize['harvest'].max()}, "
          f"mean: {valid_maize['harvest'].mean():.0f}")
    print(f"    End-of-season — min: {valid_maize['endofseaso'].min()}, "
          f"max: {valid_maize['endofseaso'].max()}, "
          f"mean: {valid_maize['endofseaso'].mean():.0f}")

    # Show a few example districts
    print(f"\n  Sample districts (first 10):")
    sample_cols = ["ADM_NAME", "COUNTRY", "country", "region", "planting", "harvest", "endofseaso"]
    available_cols = [c for c in sample_cols if c in valid_maize.columns]
    print(valid_maize[available_cols].head(10).to_string(index=False))

    return valid_maize


# In[39]:


# Execute Step 3
valid_maize = join_districts_to_maize1(gdf_districts, gdf_maize1)

# Preview results to ensure names and calendar dates matched correctly
valid_maize.head()


# In[40]:


# ---------------------------------------------------------------------------
# STEP 4: Calculate risk window for each district
# ---------------------------------------------------------------------------
def calculate_risk_windows(valid_maize):
    """
    For each district, compute the RISK WINDOW within the growing season.

    Why skip the first 25%?
      - The early season (planting → early vegetative) is when crops are
        establishing roots. Soil moisture stress during this phase is less
        impactful on final yield than stress during flowering/grain-fill.
      - The remaining 75% of the season (vegetative → end-of-season) is
        the "risk window" where drought has the most impact on yield.

    How it works:
      - Season duration = endofseaso − planting (handles year-wrap)
      - Skip = 25% of duration
      - risk_start_doy = planting + skip (mod 365)
      - risk_end_doy = endofseaso

    Returns:
        valid_maize with two new columns: risk_start_doy, risk_end_doy
    """
    print("\n" + "=" * 70)
    print("STEP 4: Calculating risk windows (skip first 25% of season)")
    print("=" * 70)

    valid_maize = valid_maize.copy()

    def _calc_risk(row):
        p = row["planting"]
        h = row["endofseaso"]

        # Season duration, handling year-wrap (e.g., planting in Nov, harvest in Apr)
        duration = (h - p) if h >= p else (365 - p) + h

        # Skip first 25% of season
        delay = duration * 0.25
        risk_start = int(round((p + delay) % 365))
        if risk_start == 0: risk_start = 365 # Keep DOY 1-365 range
        risk_end = h

        return pd.Series([risk_start, risk_end], index=["risk_start_doy", "risk_end_doy"])

    valid_maize[["risk_start_doy", "risk_end_doy"]] = valid_maize.apply(_calc_risk, axis=1)

    # Summary
    print(f"\n  Risk windows computed for {len(valid_maize)} districts")
    print(f"\n  Example (first 5):")
    print(valid_maize[["ADM_NAME", "planting", "endofseaso", "risk_start_doy", "risk_end_doy"]].head().to_string(index=False))

    # Check for year-wrapped seasons
    n_wrapped = (valid_maize["risk_start_doy"] > valid_maize["risk_end_doy"]).sum()
    n_standard = len(valid_maize) - n_wrapped
    print(f"\n  Standard seasons (start ≤ end): {n_standard}")
    print(f"  Year-wrapped seasons (start > end, e.g. Nov→Apr): {n_wrapped}")

    return valid_maize


# In[41]:


# Execute Step 4
valid_maize = calculate_risk_windows(valid_maize)

# Preview results
valid_maize[['ADM_NAME', 'planting', 'endofseaso', 'risk_start_doy', 'risk_end_doy']].head()


# In[42]:


# ---------------------------------------------------------------------------
# STEP 5: Load ERA5 and merge with district risk windows
# ---------------------------------------------------------------------------
def load_and_merge_era5(valid_maize):
    """
    Load the ERA5-Land daily soil moisture CSV and merge with district
    risk windows. Keep only rows where the day-of-year falls WITHIN
    the district's risk window.

    Also assigns crop_year:
      - For standard seasons (e.g. May→Oct): crop_year = calendar year
      - For year-wrapped seasons (e.g. Nov→Apr): days in Jan–Apr are
        assigned to the PREVIOUS year's crop season

    Returns:
        df_filtered: DataFrame with columns [feature_id, year, doy, crop_year,
                     volumetric_soil_water_layer_2, risk_start_doy, risk_end_doy]
    """
    print("\n" + "=" * 70)
    print("STEP 5: Loading ERA5 soil moisture & merging with risk windows")
    print("=" * 70)

    # Load ERA5 CSV
    print(f"\n  Loading ERA5 from:\n    {ERA5_PATH}")
    df_era5 = pd.read_csv(ERA5_PATH)
    
    # Ensure ID is uppercase for matching
    df_era5["feature_id"] = df_era5["feature_id"].astype(str).str.strip().str.upper()
    
    print(f"  → {len(df_era5):,} rows, {df_era5['feature_id'].nunique()} districts")
    print(f"  → Year range: {df_era5['year'].min()}–{df_era5['year'].max()}")
    print(f"  → Columns: {df_era5.columns.tolist()}")

    # Prepare the lookup table from valid_maize
    valid_maize_lookup = valid_maize.copy()
    valid_maize_lookup["ADM_NAME"] = valid_maize_lookup["ADM_NAME"].astype(str).str.strip().str.upper()

    # Merge: each ERA5 row gets its district's specific risk window dates
    print(f"\n  Merging ERA5 rows with {len(valid_maize_lookup)} district risk windows …")
    df_merged = df_era5.merge(
        valid_maize_lookup[["ADM_NAME", "risk_start_doy", "risk_end_doy"]],
        left_on="feature_id",
        right_on="ADM_NAME",
        how="inner",
    )
    print(f"  → Rows after merge (districts with Maize 1 calendar): {len(df_merged):,}")

    # Filter: keep only days that fall within the risk window
    # Two cases: standard season (start ≤ end) and wrapped season (start > end)
    print("\n  Filtering to days within risk window …")

    mask_standard = (
        (df_merged["risk_start_doy"] <= df_merged["risk_end_doy"])
        & (df_merged["doy"] >= df_merged["risk_start_doy"])
        & (df_merged["doy"] <= df_merged["risk_end_doy"])
    )
    mask_wrapped = (
        (df_merged["risk_start_doy"] > df_merged["risk_end_doy"])
        & (
            (df_merged["doy"] >= df_merged["risk_start_doy"])
            | (df_merged["doy"] <= df_merged["risk_end_doy"])
        )
    )
    df_filtered = df_merged[mask_standard | mask_wrapped].copy()
    print(f"  → Rows within risk window: {len(df_filtered):,}")
    print(f"    (Dropped {len(df_merged) - len(df_filtered):,} out-of-season rows)")

    # Assign crop_year
    # This is vital for insurance: it groups the entire season into one year.
    # For wrapped seasons: days early in the calendar year (Jan–Apr) belong
    # to the PREVIOUS year's planting season.
    df_filtered["crop_year"] = df_filtered["year"].copy()
    wrapped_mask = df_filtered["risk_start_doy"] > df_filtered["risk_end_doy"]
    early_in_year = df_filtered["doy"] <= df_filtered["risk_end_doy"]
    df_filtered.loc[wrapped_mask & early_in_year, "crop_year"] = (
        df_filtered.loc[wrapped_mask & early_in_year, "year"] - 1
    )

    print(f"\n  Crop year assigned:")
    print(f"    Year range in data: {df_filtered['year'].min()}–{df_filtered['year'].max()}")
    print(f"    Crop-year range:    {df_filtered['crop_year'].min()}–{df_filtered['crop_year'].max()}")
    print(f"    Districts:          {df_filtered['feature_id'].nunique()}")

    return df_filtered


# In[44]:


# Execute Step 5
# Ensure ERA5_PATH was defined in your first configuration cell
df_filtered = load_and_merge_era5(valid_maize)

# Preview the filtered and crop-year assigned data
#df_filtered.head()


# In[45]:


import numpy as np
from scipy import stats as scipy_stats

# Ensure these constants are set (from your project logic)
MIN_OBS_FOR_FIT = 10 
SSI_THRESHOLD = -1.5  # Your severe drought threshold


# In[46]:


# ---------------------------------------------------------------------------
# STEP 6: Compute daily SSI via Gamma-CDF transformation
# ---------------------------------------------------------------------------
def compute_daily_gamma_ssi(df_filtered):
    """
    Compute the Standardised Soil-moisture Index (SSI) for each daily
    observation using a Gamma-CDF transformation.
    """
    print("\n" + "=" * 70)
    print("STEP 6: Computing daily Gamma-CDF SSI")
    print("=" * 70)
    print(f"\n  Method: Gamma distribution fitted per (district, DOY)")
    print(f"  Baseline: Full record (all available years for each district)")
    print(f"  Min observations for reliable fit: {MIN_OBS_FOR_FIT}")

    df_filtered = df_filtered.copy()
    df_filtered["SSI"] = np.nan

    # Group by (district, DOY) — each group gets its own Gamma fit
    groups = df_filtered.groupby(["feature_id", "doy"])
    n_groups = len(groups)
    n_fitted = 0
    n_skipped = 0

    print(f"  Total (district × DOY) groups to fit: {n_groups:,}")
    print(f"\n  Fitting Gamma distributions … (This may take a minute)")

    for (district, doy), group in groups:
        values = group["volumetric_soil_water_layer_2"].dropna().values

        # Skip if not enough observations for a reliable fit
        if len(values) < MIN_OBS_FOR_FIT:
            n_skipped += 1
            continue

        # Separate zero and non-zero values
        nonzero = values[values > 0]
        n_zeros = len(values) - len(nonzero)
        q_zero = n_zeros / len(values)  # probability of observing zero

        if len(nonzero) < 5:
            n_skipped += 1
            continue

        # Fit Gamma distribution to non-zero values
        try:
            # floc=0 forces the location parameter to 0 (standard for SSI)
            alpha, loc, beta = scipy_stats.gamma.fit(nonzero, floc=0)
        except Exception:
            n_skipped += 1
            continue

        # Validate parameters
        if alpha <= 0 or beta <= 0 or np.isnan(alpha) or np.isnan(beta):
            n_skipped += 1
            continue

        n_fitted += 1

        # Compute SSI for every observation in this group
        idx = group.index
        sm_vals = df_filtered.loc[idx, "volumetric_soil_water_layer_2"].values
        ssi_vals = np.full(len(sm_vals), np.nan)

        for i, sm_val in enumerate(sm_vals):
            if pd.isna(sm_val):
                continue

            # Mixed distribution CDF logic
            if sm_val <= 0:
                # Probability = half the zero mass (McKee et al. 1993 convention)
                p = q_zero / 2.0
            else:
                # P(X ≤ x) = P(X=0) + P(X>0) * P(X≤x | X>0)
                p = q_zero + (1.0 - q_zero) * scipy_stats.gamma.cdf(
                    sm_val, alpha, loc=0, scale=beta
                )

            # Clamp probability to avoid infinity (±3.0 is usually plenty)
            p = np.clip(p, 0.001, 0.999)

            # Inverse standard normal: transform probability → SSI (Z-score)
            ssi_vals[i] = scipy_stats.norm.ppf(p)

        df_filtered.loc[idx, "SSI"] = ssi_vals

    # Cleanup: Drop rows where calculation failed
    n_before = len(df_filtered)
    df_filtered = df_filtered.dropna(subset=["SSI"])
    n_dropped = n_before - len(df_filtered)

    print(f"\n  Results:")
    print(f"    (district × DOY) groups fitted:  {n_fitted:,}")
    print(f"    Groups skipped:                  {n_skipped:,}")
    print(f"    Total daily SSI values created:  {len(df_filtered):,}")
    print(f"\n  SSI Statistics:")
    print(f"    Mean:  {df_filtered['SSI'].mean():.4f}  (Target: 0)")
    print(f"    Std:   {df_filtered['SSI'].std():.4f}   (Target: 1)")
    print(f"\n  Drought Check (SSI ≤ {SSI_THRESHOLD}):")
    n_drought = (df_filtered["SSI"] <= SSI_THRESHOLD).sum()
    print(f"    Count: {n_drought:,} drought days identified.")

    return df_filtered


# In[47]:


# Execute Step 6
df_filtered = compute_daily_gamma_ssi(df_filtered)

# Check the new SSI column
df_filtered[['feature_id', 'year', 'doy', 'SSI']].head()


# In[48]:


# ---------------------------------------------------------------------------
# STEP 7: Aggregate to annual drought-day counts
# ---------------------------------------------------------------------------
def aggregate_annual_drought_days(df_filtered):
    """
    Count the number of drought days per district per crop-year.

    A "drought day" is any day where SSI is less than or equal to the threshold.
    This creates the 'Intensity' and 'Frequency' metrics for our Risk Zones.

    Returns:
        df_annual: DataFrame with columns [feature_id, year, Drought_Days]
    """
    print("\n" + "=" * 70)
    print("STEP 7: Aggregating to annual drought-day counts")
    print("=" * 70)
    print(f"\n  Current Threshold: SSI ≤ {SSI_THRESHOLD}")

    df_filtered = df_filtered.copy()
    
    # Create the binary 'Trigger' column (1 if drought, 0 if not)
    df_filtered["is_drought"] = (df_filtered["SSI"] <= SSI_THRESHOLD).astype(int)

    # Group by District and Crop-Year to sum the total drought days per season
    df_annual = (
        df_filtered.groupby(["feature_id", "crop_year"])["is_drought"]
        .sum()
        .reset_index()
        .rename(columns={"is_drought": "Drought_Days", "crop_year": "year"})
    )

    # Keep only 2000–2025 to avoid partial seasons at the data edges
    df_annual = df_annual[(df_annual["year"] >= 2000) & (df_annual["year"] <= 2025)].copy()

    # Print summary statistics for verification
    print(f"\n  Results:")
    print(f"    Total District-Year records:     {len(df_annual):,}")
    print(f"    Unique districts processed:      {df_annual['feature_id'].nunique()}")
    print(f"    Year range:                      {df_annual['year'].min()}–{df_annual['year'].max()}")
    print(f"    Avg drought days per season:     {df_annual['Drought_Days'].mean():.1f}")
    print(f"    Highest count in a single year:  {df_annual['Drought_Days'].max()}")
    
    # Calculate how many districts stayed "Safe" across all 25 years
    zero_districts = (df_annual.groupby('feature_id')['Drought_Days'].sum() == 0).sum()
    print(f"    Districts with 0 drought days (total 25yr history): {zero_districts}")

    return df_annual


# In[49]:


# Execute Step 7
df_annual = aggregate_annual_drought_days(df_filtered)

# Preview the annual time series for a few districts
df_annual.sort_values(by=['feature_id', 'year']).head(10)


# In[50]:


# ---------------------------------------------------------------------------
# SEVERITY BIN DEFINITIONS
# ---------------------------------------------------------------------------

# Drought severity bins: (lower, upper, label)
# Aligned with the US Drought Monitor D0–D4 scale (Svoboda et al. 2002)
DROUGHT_BINS = [
    (0,    0,   "No Drought"),          
    (1,   10,   "D1 – Abnormally Dry"),
    (11,  20,   "D2 – Moderate Drought"),
    (21,  35,   "D3 – Severe Drought"),
    (36,  55,   "D4 – Extreme Drought"),
    (56, None,  "D5 – Exceptional Drought"),
]

def classify_drought_severity(drought_days):
    """Map a drought-day count to a severity category."""
    for lower, upper, label in DROUGHT_BINS:
        if upper is None:
            return label
        if lower <= drought_days <= upper:
            return label
    return "D5 – Exceptional Drought"

def get_category_order():
    """Return ordered list of severity labels."""
    return [label for _, _, label in DROUGHT_BINS]

def get_color_map(category_order):
    """Assign hex colors to severity bins for mapping."""
    SEVERITY_COLORS = {
        "No Drought":                "rgba(220,220,220,0.25)",
        "D1 – Abnormally Dry":       "#FFEDA0",   
        "D2 – Moderate Drought":     "#FEB24C",   
        "D3 – Severe Drought":       "#FC4E2A",   
        "D4 – Extreme Drought":      "#BD0026",   
        "D5 – Exceptional Drought":  "#4A0010",   
    }
    return {cat: SEVERITY_COLORS.get(cat, "#999999") for cat in category_order}


# In[51]:


# ---------------------------------------------------------------------------
# STEP 8: Classifying drought severity
# ---------------------------------------------------------------------------
def bin_drought_days(df_annual):
    """
    Add a 'Drought_Category' column to df_annual based on severity bins.
    """
    print("\n" + "=" * 70)
    print("STEP 8: Classifying drought severity (WMO 2012 / Svoboda et al. 2002)")
    print("=" * 70)

    df_annual = df_annual.copy()
    
    # Apply the classification logic to every row
    df_annual["Drought_Category"] = df_annual["Drought_Days"].apply(classify_drought_severity)

    # Print the final distribution of events across your 25-year history
    category_order = get_category_order()
    print(f"\n  Severity distribution across all 25 years (district-years):")
    for cat in category_order:
        count = (df_annual["Drought_Category"] == cat).sum()
        pct = 100 * count / len(df_annual)
        print(f"    {cat:<30s}  {count:>6,} ({pct:>5.1f}%)")

    return df_annual


# In[52]:


# Execute Step 8 using the df_annual created in Step 7
df_annual = bin_drought_days(df_annual)

# Preview the results
#df_annual.sort_values(by='Drought_Days', ascending=False).head(10)


# In[54]:


import plotly.express as px
import json

# ---------------------------------------------------------------------------
# GROUP COLOUR PALETTE (consistent across all maps)
# ---------------------------------------------------------------------------
GROUP_COLOR_MAP = {
    "Jan–Feb planters": "#1b9e77",
    "Mar–Apr planters": "#d95f02",
    "May–Jun planters": "#7570b3",
    "Jul–Aug planters": "#e7298a",
    "Sep–Oct planters": "#66a61e",
    "Nov–Dec planters": "#e6ab02",
}

def _build_group_legend_html(district_to_group):
    """Build legend HTML items for only the groups actually present in data."""
    present_groups = sorted(set(district_to_group.values()))
    lines = []
    for g in present_groups:
        color = GROUP_COLOR_MAP.get(g, "#999999")
        lines.append(
            f'  <div class="legend-item"><div class="legend-color" style="background:{color}"></div> {g}</div>'
        )
    return "\n".join(lines)

def _build_group_colors_js(district_to_group):
    """Build JS object literal for only the groups actually present in data."""
    present_groups = sorted(set(district_to_group.values()))
    entries = []
    for g in present_groups:
        color = GROUP_COLOR_MAP.get(g, "#999999")
        entries.append(f'  "{g}": "{color}"')
    return "{\n" + ",\n".join(entries) + "\n}"

def _build_short_group_legend_html(district_to_group):
    """Build legend for inter-group map using shortened names (no 'planters')."""
    present_groups = sorted(set(district_to_group.values()))
    lines = []
    for g in present_groups:
        color = GROUP_COLOR_MAP.get(g, "#999999")
        short = g.replace(" planters", "")
        lines.append(
            f'  <div class="legend-item"><div class="legend-color" style="background:{color}"></div> {short}</div>'
        )
    return "\n".join(lines)

def _build_short_group_colors_js(district_to_group, suffix=""):
    """Build JS object for inter-group map using shortened group keys, with optional hex suffix."""
    present_groups = sorted(set(district_to_group.values()))
    entries = []
    for g in present_groups:
        color = GROUP_COLOR_MAP.get(g, "#999999")
        short = g.replace(" planters", "")
        entries.append(f'  "{short}": "{color}{suffix}"')
    return "{\n" + ",\n".join(entries) + "\n}"


# In[57]:


def get_category_order():
    return ["No Drought", "D1 – Abnormally Dry", "D2 – Moderate Drought", 
            "D3 – Severe Drought", "D4 – Extreme Drought", "D5 – Exceptional Drought"]

def get_color_map(category_order):
    SEVERITY_COLORS = {
        "No Drought":                "rgba(220,220,220,0.25)",
        "D1 – Abnormally Dry":       "#FFEDA0",
        "D2 – Moderate Drought":     "#FEB24C",
        "D3 – Severe Drought":       "#FC4E2A",
        "D4 – Extreme Drought":      "#BD0026",
        "D5 – Exceptional Drought":  "#4A0010",
    }
    return {cat: SEVERITY_COLORS.get(cat, "#999999") for cat in category_order}


# In[58]:


# ---------------------------------------------------------------------------
# STEP 9: Generate animated choropleth map
# ---------------------------------------------------------------------------
def generate_animated_map(df_annual, gdf_districts):
    """
    Produce an animated Plotly choropleth HTML map showing drought severity
    per district per year (2000–2025), with a year slider/play button.
    """
    print("\n" + "=" * 70)
    print("STEP 9: Generating animated choropleth map")
    print("=" * 70)

    # These functions must be defined in your notebook
    category_order = get_category_order()
    color_map = get_color_map(category_order)

    # Simplify district geometry for faster browser rendering
    print("\n  Simplifying geometries for map …")
    gdf_plot = gdf_districts.copy()
    gdf_plot["ADM_NAME"] = gdf_plot["ADM_NAME"].astype(str).str.strip().str.upper()
    gdf_plot["geometry"] = gdf_plot["geometry"].simplify(tolerance=0.1, preserve_topology=True)

    # Filter to districts that actually have drought data
    valid_names = df_annual["feature_id"].unique()
    gdf_plot = gdf_plot[gdf_plot["ADM_NAME"].isin(valid_names)].copy()
    gdf_plot = gdf_plot.drop_duplicates("ADM_NAME")
    print(f"  Districts in map: {len(gdf_plot)}")

    # Merge annual data with spatial geometry
    final_gdf = gdf_plot[["ADM_NAME", "geometry"]].merge(
        df_annual, left_on="ADM_NAME", right_on="feature_id", how="inner"
    )
    final_gdf = final_gdf.to_crs(epsg=4326)

    # Build GeoJSON for Plotly
    temp_gdf = (
        final_gdf[["ADM_NAME", "geometry"]]
        .drop_duplicates("ADM_NAME")
        .set_index("ADM_NAME")
    )
    geojson_payload = json.loads(temp_gdf.geometry.to_json())

    # Build the animated Plotly figure
    print(f"  Building Plotly choropleth ({final_gdf['ADM_NAME'].nunique()} districts) …")

    fig = px.choropleth(
        final_gdf,
        geojson=geojson_payload,
        locations="ADM_NAME",
        color="Drought_Category",
        animation_frame="year",
        category_orders={"Drought_Category": category_order},
        color_discrete_map=color_map,
        title=(
            "<b>African Maize Drought Severity (Gamma-SSI Method, 2000–2025)</b>"
            "<br><sup>Days with SSI ≤ -1.0 during Maize 1 risk window</sup>"
        ),
        labels={"Drought_Category": "Drought Severity"},
    )

    # Style the map for a professional look
    fig.update_traces(marker_line_color="white", marker_line_width=0.2)
    fig.update_geos(
        visible=True,
        scope="africa",
        showland=True,
        landcolor="#D3D3D3",
        showocean=True,
        oceancolor="white",
        fitbounds="locations",
    )
    fig.update_layout(
        margin={"r": 0, "t": 80, "l": 0, "b": 60},
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend_title_text="Drought Severity",
        font=dict(family="Inter, sans-serif")
    )

    # --- FIXED PATH LOGIC ---
    # Instead of using a missing BASE_DIR, we use the current folder
    output_filename = "Africa_Maize_GammaSSI_Drought_Map.html"
    output_path = os.path.join(os.getcwd(), output_filename)
    
    fig.write_html(output_path, include_plotlyjs="cdn")
    
    file_size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\n  ✅ Map saved to your current folder: {output_path}")
    print(f"     File size: {file_size_mb:.1f} MB")

    return output_path


# In[59]:


# Now run the function
generate_animated_map(df_annual, gdf_districts)


# In[60]:


import math
import json

# ---------------------------------------------------------------------------
# STEP 10: Assign each district to a season group
# ---------------------------------------------------------------------------
def classify_season_groups(valid_maize):
    """
    Groups districts by the MONTH in which planting begins. 
    This follows the Africa Risk Capacity (ARC) operational model for 
    parametric insurance.
    """
    print("\n" + "=" * 70)
    print("STEP 10: Grouping districts by crop-season type")
    print("=" * 70)

    valid = valid_maize.copy()
    # Approximate month from DOY
    valid["plant_month"] = valid["planting"].apply(
        lambda d: min(12, math.ceil(d / 30.44)) if d > 0 else 0
    )

    # Bimonthly grouping (6 groups for the whole continent)
    BIMONTH_LABELS = {
        (1, 2):   "Jan–Feb planters",
        (3, 4):   "Mar–Apr planters",
        (5, 6):   "May–Jun planters",
        (7, 8):   "Jul–Aug planters",
        (9, 10):  "Sep–Oct planters",
        (11, 12): "Nov–Dec planters",
    }

    def _get_bimonth_group(month):
        for (m1, m2), label in BIMONTH_LABELS.items():
            if month in (m1, m2):
                return label
        return "Unknown"

    valid["season_group"] = valid["plant_month"].apply(_get_bimonth_group)

    # Build groups dict (only include groups with ≥2 districts for correlation)
    groups = {}
    for grp_name, sub in valid.groupby("season_group"):
        if grp_name == "Unknown": continue
        members = sub["ADM_NAME"].tolist()
        if len(members) >= 2:
            groups[grp_name] = members

    print(f"\n  Season groups identified: {len(groups)}")
    for name, members in sorted(groups.items()):
        print(f"    {name:<22s}: {len(members):>5} districts")

    # Create a lookup for later mapping
    district_to_group = {}
    for grp_name, members in groups.items():
        for d in members:
            district_to_group[d] = grp_name

    return groups, district_to_group


# In[81]:


# ---------------------------------------------------------------------------
# STEP 11a: Compute Pearson correlation within groups
# ---------------------------------------------------------------------------
def compute_within_group_correlation(df_annual, groups):
    """
    Calculates how closely districts 'move together' into drought.
    High correlation = Systemic Risk Zone.
    """
    print("\n" + "=" * 70)
    print("STEP 11: Computing within-group district drought correlation")
    print("=" * 70)

    min_years = 5  # Require at least 5 years of data for a valid correlation
    results = {}

    for group_name, district_list in groups.items():
        # Filter annual data to districts in this specific planting group
        df_grp = df_annual[df_annual["feature_id"].isin(district_list)].copy()

        # Pivot to: Rows = Years, Columns = Districts
        pivot = df_grp.pivot_table(index="year", columns="feature_id", values="Drought_Days")

        # Keep districts that meet our data-length criteria
        valid_cols = pivot.columns[pivot.notna().sum() >= min_years]
        pivot = pivot[valid_cols]

        if len(valid_cols) < 2:
            continue

        # Fill missing years with the mean (conservative padding)
        pivot_filled = pivot.fillna(pivot.mean())

        # Perform Pearson Correlation
        corr = pivot_filled.corr()
        results[group_name] = (corr, valid_cols.tolist())
        
    print("  Correlation matrices calculated for all groups.")
    return results


# In[82]:


def _build_group_legend_html(district_to_group):
    """Build legend HTML items for only the groups actually present in data."""
    GROUP_COLOR_MAP = {
        "Jan–Feb planters": "#1b9e77",
        "Mar–Apr planters": "#d95f02",
        "May–Jun planters": "#7570b3",
        "Jul–Aug planters": "#e7298a",
        "Sep–Oct planters": "#66a61e",
        "Nov–Dec planters": "#e6ab02",
    }
    present_groups = sorted(set(district_to_group.values()))
    lines = []
    for g in present_groups:
        color = GROUP_COLOR_MAP.get(g, "#999999")
        lines.append(f'  <div class="legend-item"><div class="legend-color" style="background:{color}"></div> {g}</div>')
    return "\n".join(lines)

def _build_group_colors_js(district_to_group):
    """Build JS object literal for colors."""
    GROUP_COLOR_MAP = {
        "Jan–Feb planters": "#1b9e77",
        "Mar–Apr planters": "#d95f02",
        "May–Jun planters": "#7570b3",
        "Jul–Aug planters": "#e7298a",
        "Sep–Oct planters": "#66a61e",
        "Nov–Dec planters": "#e6ab02",
    }
    present_groups = sorted(set(district_to_group.values()))
    entries = [f'  "{g}": "{GROUP_COLOR_MAP.get(g, "#999999")}"' for g in present_groups]
    return "{\n" + ",\n".join(entries) + "\n}"


# In[83]:


def _write_correlation_html(geojson, corr_lookup, ts_data, district_to_group, district_to_country):
    """Write the standalone interactive correlation HTML map (EXACT .PY VERSION)."""
    import json
    import os

    geojson_str = json.dumps(geojson)
    corr_str = json.dumps(corr_lookup)
    ts_str = json.dumps(ts_data)
    groups_str = json.dumps(district_to_group)
    countries_str = json.dumps(district_to_country)

    group_legend_html = _build_group_legend_html(district_to_group)
    group_colors_js = _build_group_colors_js(district_to_group)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Maize District Drought Correlation (Gamma-SSI)</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  body {{ margin:0; font-family: 'Inter', 'Segoe UI', sans-serif; }}
  #map {{ width:100%; height:100vh; }}
  .info-panel {{
    position:absolute; top:10px; right:10px; z-index:1000;
    background:white; padding:14px 18px; border-radius:8px;
    box-shadow:0 2px 12px rgba(0,0,0,0.15); max-width:360px;
    font-size:13px; line-height:1.5;
  }}
  .legend {{
    position:absolute; bottom:30px; left:10px; z-index:1000;
    background:white; padding:12px 16px; border-radius:8px;
    box-shadow:0 2px 12px rgba(0,0,0,0.15); font-size:12px;
  }}
  .legend-item {{ display:flex; align-items:center; margin:3px 0; }}
  .legend-color {{ width:20px; height:14px; margin-right:8px; border:1px solid #ccc; }}
  .ts-chart {{ margin-top:10px; }}
  .ts-bar {{ display:inline-block; width:8px; margin:0 1px; background:#e74c3c;
             vertical-align:bottom; border-radius:2px 2px 0 0; }}
</style>
</head>
<body>
<div id="map"></div>
<div class="info-panel" id="info">
  <h3>🌽 Maize District Drought Correlation</h3>
  <p><b>Click a district</b> to see correlated districts within its crop-season group.</p>
  <div id="details"></div>
</div>
<div class="legend">
  <b>Correlation with selected district</b>
  <div class="legend-item"><div class="legend-color" style="background:#08519c"></div> r ≥ 0.8 (very high)</div>
  <div class="legend-item"><div class="legend-color" style="background:#3182bd"></div> 0.6 ≤ r < 0.8 (high)</div>
  <div class="legend-item"><div class="legend-color" style="background:#6baed6"></div> 0.4 ≤ r < 0.6 (moderate)</div>
  <div style="margin-top:8px;"><b>Season groups (no selection)</b></div>
  {group_legend_html}
</div>
<script>
const geojson = {geojson_str};
const corrData = {corr_str};
const tsData = {ts_str};
const districtGroups = {groups_str};
const groupColors = {group_colors_js};

const map = L.map('map').setView([0, 20], 4);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_noclabels/{{z}}/{{x}}/{{y}}@2x.png').addTo(map);

function corrColor(r) {{
  if (r >= 0.8) return "#08519c";
  if (r >= 0.6) return "#3182bd";
  if (r >= 0.4) return "#6baed6";
  if (r >= 0.2) return "#bdd7e7";
  return "#eff3ff";
}}

function makeSparkline(dist) {{
  if (!tsData[dist]) return "";
  const years = Object.keys(tsData[dist]).sort();
  const vals = years.map(y => tsData[dist][y]);
  const maxV = Math.max(...vals, 1);
  let html = '<div class="ts-chart">';
  vals.forEach((v, i) => {{
    const h = Math.max(2, (v / maxV) * 40);
    html += `<div class="ts-bar" style="height:${{h}}px" title="${{years[i]}}: ${{v}} days"></div>`;
  }});
  html += '</div>';
  return html;
}}

const layerLookup = {{}};
const geoLayer = L.geoJSON(geojson, {{
  style: (f) => ({{ fillColor: groupColors[f.properties.season_group], weight: 0.5, color: "#999", fillOpacity: 0.6 }}),
  onEachFeature: (f, l) => {{
    const name = f.properties.ADM_NAME;
    layerLookup[name] = l;
    l.on('click', () => {{
      const corrs = corrData[name] || {{}};
      geoLayer.eachLayer(layer => layer.setStyle({{fillColor: groupColors[layer.feature.properties.season_group], fillOpacity:0.6}}));
      l.setStyle({{fillColor: "#ffd700", fillOpacity: 1}});
      Object.entries(corrs).forEach(([d, r]) => {{
          if(layerLookup[d]) layerLookup[d].setStyle({{fillColor: corrColor(r), fillOpacity: 0.8}});
      }});
      document.getElementById('details').innerHTML = `<h3>${{name}}</h3>` + makeSparkline(name);
    }});
  }}
}}).addTo(map);
</script>
</body>
</html>"""
    output_path = os.path.join(os.getcwd(), "Africa_Maize_GammaSSI_Correlation_Map.html")
    with open(output_path, "w", encoding="utf-8") as f: f.write(html)
    print(f"✅ Correlation map saved: {'C:/Users/FlawiyaShirishMore/OneDrive - Africa Specialty Risks Ltd/ASR-Parametric_Research_Study/africa_risk/Drought/Output'}")


# In[84]:


generate_correlation_map(df_annual, valid_maize, gdf_districts, groups, district_to_group, district_to_country)


# In[85]:


import pandas as pd
import numpy as np
import json

# ---------------------------------------------------------------------------
# STEP 12: Inter-group (Continental) Correlation Analysis
# ---------------------------------------------------------------------------
def generate_intergroup_correlation_map(df_annual, valid_maize, gdf_districts, district_to_group, district_to_country):
    """
    Compute and map correlation between ALL district pairs across Africa.
    """
    print("\n" + "=" * 70)
    print("STEP 12: Computing inter-group (continental) correlation matrix")
    print("=" * 70)

    # All districts in our 25-year history
    all_districts = df_annual["feature_id"].unique()
    
    # Pivot full matrix: Rows = Years, Columns = Districts
    pivot = df_annual.pivot_table(
        index="year", columns="feature_id", values="Drought_Days"
    )

    # Require at least 5 years of data per district
    min_years = 5
    valid_cols = pivot.columns[pivot.notna().sum() >= min_years]
    pivot = pivot[valid_cols]
    print(f"  Districts meeting data criteria: {len(valid_cols)}")

    # Conservative approach: Fill missing years with the district's mean
    pivot_filled = pivot.fillna(pivot.mean())

    # Compute full Pearson correlation matrix
    corr = pivot_filled.corr()
    print(f"  Continental correlation matrix created: {corr.shape[0]}x{corr.shape[1]}")

    # Prepare Geometry for the map
    print("\n  Preparing map geometry …")
    gdf_plot = gdf_districts.copy()
    gdf_plot["ADM_NAME"] = gdf_plot["ADM_NAME"].astype(str).str.strip().str.upper()
    gdf_plot["geometry"] = gdf_plot["geometry"].simplify(tolerance=0.05, preserve_topology=True)
    gdf_plot = gdf_plot.to_crs(epsg=4326)

    valid_set = set(valid_cols)
    gdf_plot = gdf_plot[gdf_plot["ADM_NAME"].isin(valid_set)].drop_duplicates("ADM_NAME")
    
    # Attach group and country labels for tooltips
    gdf_plot["season_group"] = gdf_plot["ADM_NAME"].map(district_to_group).fillna("Unknown")
    gdf_plot["COUNTRY_LABEL"] = gdf_plot["ADM_NAME"].map(district_to_country).fillna("Unknown")

    # Build GeoJSON
    geojson = json.loads(
        gdf_plot[["ADM_NAME", "COUNTRY_LABEL", "geometry", "season_group"]].to_json()
    )

    # Build correlation lookup for the interactive map
    corr_lookup = {}
    for dist in valid_cols:
        row = corr.loc[dist].drop(dist)
        corr_lookup[dist] = {
            k: round(v if not np.isnan(v) else 0.0, 2)
            for k, v in row.items()
        }

    # Build time series data for the sparkline charts
    ts_data = {}
    for dist, sub in df_annual.groupby("feature_id"):
        if dist in valid_set:
            ts_data[dist] = dict(zip(
                sub["year"].astype(int).tolist(),
                sub["Drought_Days"].astype(int).tolist()
            ))

    # Save to HTML
    print("  Generating final interactive HTML …")
    _write_intergroup_html(geojson, corr_lookup, ts_data, district_to_group, district_to_country)


# In[94]:


import os
import json
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# STEP 12: Inter-group (Continental) Correlation Dashboard (Evans 1996)
# ---------------------------------------------------------------------------

def _write_intergroup_html(geojson, corr_lookup, district_to_country):
    """Write the interactive continental correlation dashboard with Evans (1996) standards."""
    
    geojson_str = json.dumps(geojson)
    corr_str = json.dumps(corr_lookup)
    countries_str = json.dumps(district_to_country)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>African Maize — Systemic Risk Correlation (Evans 1996)</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  body {{ margin:0; font-family:'Segoe UI', Roboto, Helvetica, Arial, sans-serif; display: flex; background: #f0f2f5; }}
  #map {{ width:70%; height:100vh; background: #cbd5e0; }}
  #sidebar {{ 
    width:30%; height:100vh; background:#ffffff; border-left:1px solid #e2e8f0; 
    overflow-y:auto; padding:25px; box-sizing: border-box; box-shadow: -2px 0 10px rgba(0,0,0,0.1);
  }}
  h2 {{ font-size:22px; color:#b2182b; margin-top:0; border-bottom: 3px solid #eee; padding-bottom:10px; }}
  h4 {{ font-size:14px; color:#4a5568; margin-top:25px; margin-bottom:10px; text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid #edf2f7; }}
  
  .selected-card {{ background: #fff5f5; padding: 15px; border-radius: 8px; border: 1px solid #feb2b2; margin-bottom: 20px; }}
  .district-title {{ font-size: 24px; font-weight: bold; color: #2d3748; }}
  .country-title {{ font-size: 16px; color: #718096; margin-bottom: 5px; }}
  
  .corr-table {{ width:100%; border-collapse:collapse; font-size:12px; }}
  .corr-table td {{ padding:8px 4px; border-bottom:1px solid #f7fafc; }}
  .r-val {{ font-weight:bold; text-align:right; font-family: monospace; font-size: 14px; }}
  
  .legend {{
    position:absolute; bottom:40px; left:20px; z-index:1000;
    background:white; padding:15px; border-radius:8px; font-size:12px; box-shadow:0 4px 12px rgba(0,0,0,0.15);
    border: 1px solid #e2e8f0;
  }}
  .legend-item {{ display:flex; align-items:center; margin:6px 0; }}
  .legend-color {{ width:24px; height:14px; margin-right:12px; border-radius:3px; border: 0.5px solid #999; }}
  .footer-ref {{ font-size: 10px; color: #a0aec0; margin-top: 15px; font-style: italic; line-height: 1.4; }}
</style>
</head>
<body>
<div id="map"></div>
<div id="sidebar">
  <h2>Risk Explorer</h2>
  <div id="details">
    <p style="color:#718096;">Click any district on the map to visualize its <b>Spatial Correlation (r)</b> with the rest of the continent using Evans (1996) standards.</p>
  </div>
  <div id="tables"></div>
</div>

<div class="legend">
  <b style="font-size:13px;">Correlation Strength (Evans 1996)</b>
  <div class="legend-item"><div class="legend-color" style="background:#b2182b"></div> 0.80 - 1.00 (Very Strong)</div>
  <div class="legend-item"><div class="legend-color" style="background:#ef8a62"></div> 0.60 - 0.79 (Strong)</div>
  <div class="legend-item"><div class="legend-color" style="background:#4daf4a"></div> 0.40 - 0.59 (Moderate)</div>
  <div class="legend-item"><div class="legend-color" style="background:#9ecae1"></div> 0.20 - 0.39 (Weak)</div>
  <div class="legend-item"><div class="legend-color" style="background:#f0f0f0"></div> 0.00 - 0.19 (Negligible)</div>
  <div class="footer-ref">Scientific Standard: Evans, J. D. (1996). <br>Straightforward Statistics for the Behavioral Sciences.</div>
</div>

<script>
const geojson = {geojson_str};
const corrData = {corr_str};
const countryLookup = {countries_str};

const map = L.map('map').setView([0, 20], 4);

// Tile layer that keeps country boundaries and names visible
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}@2x.png', {{
  attribution: '&copy; CARTO'
}}).addTo(map);

function getEvansColor(r) {{
  const val = Math.abs(r);
  if (val >= 0.8) return "#b2182b"; // DEEP RED
  if (val >= 0.6) return "#ef8a62"; // ORANGE/RED
  if (val >= 0.4) return "#4daf4a"; // GREEN
  if (val >= 0.2) return "#9ecae1"; // LIGHT BLUE
  return "#f0f0f0"; // GREY (Negligible)
}}

let layerLookup = {{}};

const geoLayer = L.geoJSON(geojson, {{
  style: {{ fillColor: "#ffffff", weight: 0.8, color: "#a0aec0", fillOpacity: 0.5 }},
  onEachFeature: (f, l) => {{
    const name = f.properties.ADM_NAME;
    layerLookup[name] = l;
    l.bindTooltip(`<b>${{name}}</b> (${{countryLookup[name] || '...'}})`);
    l.on('click', (e) => {{
        L.DomEvent.stopPropagation(e);
        selectDistrict(name);
    }});
  }}
}}).addTo(map);

function selectDistrict(name) {{
  const corrs = corrData[name] || {{}};
  const selCountry = countryLookup[name] || "Unknown";

  // Color the whole map based on correlation to selected district
  geoLayer.eachLayer(layer => {{
    const targetName = layer.feature.properties.ADM_NAME;
    if (targetName === name) {{
        layer.setStyle({{ fillColor: "#000000", fillOpacity: 0.9, weight: 2, color: "#000" }});
        layer.bringToFront();
    }} else if (corrs[targetName] !== undefined) {{
        layer.setStyle({{ 
            fillColor: getEvansColor(corrs[targetName]), 
            fillOpacity: 0.8, 
            weight: 0.3, 
            color: "#fff" 
        }});
    }} else {{
        layer.setStyle({{ fillColor: "#ffffff", fillOpacity: 0.1, weight: 0.5, color: "#cbd5e0" }});
    }}
  }});

  // Update Sidebar
  document.getElementById('details').innerHTML = `
    <div class="selected-card">
      <div class="country-title">${{selCountry}}</div>
      <div class="district-title">${{name}}</div>
    </div>
  `;

  const sorted = Object.entries(corrs).sort((a, b) => b[1] - a[1]);
  const topPos = sorted.slice(0, 10);
  const topNeg = [...sorted].reverse().slice(0, 10);

  let html = "<h4>Top 10 Positive Correlations</h4><table class='corr-table'>";
  topPos.forEach(([d, r]) => {{
    const c = countryLookup[d] || "Unknown";
    html += `<tr><td>${{d}} (${{c}})</td><td class='r-val' style='color:${{getEvansColor(r)}}'>${{r.toFixed(3)}}</td></tr>`;
  }});
  html += "</table>";

  html += "<h4>Top 10 Negative (Diversifiers)</h4><table class='corr-table'>";
  topNeg.forEach(([d, r]) => {{
    const c = countryLookup[d] || "Unknown";
    html += `<tr><td>${{d}} (${{c}})</td><td class='r-val' style='color:#718096'>${{r.toFixed(3)}}</td></tr>`;
  }});
  html += "</table>";

  document.getElementById('tables').innerHTML = html;
}}

map.on('click', () => {{
    geoLayer.eachLayer(l => l.setStyle({{ fillColor: "#ffffff", weight: 0.8, color: "#a0aec0", fillOpacity: 0.5 }}));
    document.getElementById('details').innerHTML = '<p>Click a district on the map to begin analysis.</p>';
    document.getElementById('tables').innerHTML = '';
}});
</script>
</body>
</html>"""

    output_path = os.path.join(os.getcwd(), "Africa_Maize_GammaSSI_InterGroup_Correlation_Map.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Inter-District Correlation Map Generated: {output_path}")

def generate_intergroup_correlation_map(df_annual, valid_maize, gdf_districts, district_to_country):
    print("\nSTEP 12: Running Continental Correlation Logic (Standard: Evans 1996)...")
    
    # 1. Create Pivot Table
    pivot = df_annual.pivot_table(index="year", columns="feature_id", values="Drought_Days")
    
    # 2. Filter for districts with at least 5 years of data
    valid_cols = pivot.columns[pivot.notna().sum() >= 5]
    pivot_filtered = pivot[valid_cols].fillna(pivot[valid_cols].mean())
    
    # 3. Calculate Pearson r Matrix
    corr_matrix = pivot_filtered.corr()
    
    # 4. Prepare Map Geometry
    gdf_plot = gdf_districts.copy()
    gdf_plot["ADM_NAME"] = gdf_plot["ADM_NAME"].astype(str).str.strip().str.upper()
    valid_set = set(valid_cols)
    gdf_plot = gdf_plot[gdf_plot["ADM_NAME"].isin(valid_set)].drop_duplicates("ADM_NAME")
    gdf_plot["geometry"] = gdf_plot["geometry"].simplify(tolerance=0.04)
    gdf_plot = gdf_plot.to_crs(epsg=4326)
    
    geojson = json.loads(gdf_plot[["ADM_NAME", "geometry"]].to_json())
    
    # 5. Build lookup dictionary (Corrected dictionary syntax)
    corr_lookup = {} 
    for dist in valid_cols:
        row = corr_matrix.loc[dist].drop(dist)
        # Store correlations to show full map range
        corr_lookup[dist] = {k: round(float(v), 3) for k, v in row.items()}
    
    # 6. Generate the HTML
    _write_intergroup_html(geojson, corr_lookup, district_to_country)

# --- EXECUTION ---
generate_intergroup_correlation_map(df_annual, valid_maize, gdf_districts, district_to_country)


# In[93]:


generate_intergroup_correlation_map(
    df_annual, 
    valid_maize, 
    gdf_districts, 
    district_to_group, 
    district_to_country
)


# In[90]:


import os
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

def _write_group_matrix_png(groups, corr_matrix, pval_matrix, group_counts):
    """Generate a professional PNG heatmap of group-level Spearman ρ."""
    
    n = len(groups)
    group_descriptions = list(groups)
    group_district_counts = [int(group_counts.get(g, 0)) for g in groups]

    # Labels for the axes
    y_labels = [f"Grp {i+1}: {group_descriptions[i]} ({group_district_counts[i]} dist)" for i in range(n)]
    x_labels = [f"Grp {i+1}: {group_descriptions[i].replace(' planters','')}" for i in range(n)]

    # Diverging color map (Red = Positive Corr, Blue = Negative/Diversified)
    cmap = plt.cm.RdBu_r
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    im = ax.imshow(corr_matrix, cmap=cmap, norm=norm, aspect="equal")

    # Add the text values and significance stars to each cell
    for i in range(n):
        for j in range(n):
            val = corr_matrix[i, j]
            if not np.isnan(val):
                sig = "**" if pval_matrix[i, j] < 0.01 else ("*" if pval_matrix[i, j] < 0.05 else "")
                text = f"{val:.2f}{sig}"
                color = "white" if abs(val) > 0.5 else "#1a1a2e"
                ax.text(j, i, text, ha="center", va="center", fontsize=11, fontweight="bold", color=color)

    # Styling axes
    ax.set_xticks(range(n))
    ax.set_xticklabels(x_labels, fontsize=9, rotation=45, ha="left")
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.set_yticks(range(n))
    ax.set_yticklabels(y_labels, fontsize=9)

    # Grid and Colorbar
    fig.colorbar(im, ax=ax, shrink=0.7).set_label("Spearman ρ", fontweight="bold")
    ax.set_title("Season Group Correlation Matrix (Systemic Risk Analysis)\n", fontsize=14, fontweight="bold")

    # --- FIXED PATH HERE ---
    output_path = os.path.join(os.getcwd(), "Africa_Maize_GammaSSI_Group_Correlation_Matrix.png")
    
    plt.savefig(output_path, bbox_inches="tight", facecolor="white")
    print(f"  ✅ Heatmap saved to folder: {'C:/Users/FlawiyaShirishMore/OneDrive - Africa Specialty Risks Ltd/ASR-Parametric_Research_Study/africa_risk/Drought/Output'}")
    plt.show() # This displays the map directly in your Jupyter Notebook


# In[89]:


from scipy.stats import spearmanr

# Execute the final step
generate_group_correlation_matrix(df_annual, district_to_group)


# In[ ]:




