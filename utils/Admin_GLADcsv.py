import os
import ee
import geemap
import fiona
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import shutil

# =============================================================================
# 0. MASTER CONFIGURATION
# =============================================================================
# Change this to: "africa", "ethiopia", or "zambia" to run different parts
RUN_MODE = "ethiopia" 

PROJECT_ID = 'vernal-parser-412016'
BASE_DIR = Path(r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk")
GADM_PATH = BASE_DIR / "Drought" / "data" / "gadm_410-levels" / "gadm_410-levels.gpkg"
OUTPUT_DIR = BASE_DIR / "Drought" / "Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def initialize_gee():
    try:
        ee.Initialize(project=PROJECT_ID)
        print(f"✅ GEE Initialized: {PROJECT_ID}")
    except:
        print("🔑 Authenticating Earth Engine...")
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)

def safe_ee_to_df(fc):
    """
    Manual replacement for geemap.ee_to_df to bypass the Pandas/Geemap conflict bug.
    """
    print("📥 Downloading data from Google Earth Engine...")
    # Drop geometry on server side to minimize data transfer size
    fc_no_geom = fc.select(['.*'], retainGeometry=False)
    
    # Get info manually
    info = fc_no_geom.getInfo()
    features = [f['properties'] for f in info['features']]
    return pd.DataFrame(features)

# =============================================================================
# BLOCK 1: BOUNDARY RECONSTRUCTION (GADM HARMONIZATION)
# =============================================================================
def reconstruct_boundaries(mode):
    print(f"📂 Processing Boundaries for Mode: {mode}")
    
    if mode == "africa":
        adm2 = gpd.read_file(GADM_PATH, layer='ADM_2')
        adm1 = gpd.read_file(GADM_PATH, layer='ADM_1')
        africa_iso3 = ["DZA","AGO","BEN","BWA","BFA","BDI","CMR","CPV","CAF","TCD","COM","COG","COD","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE","ESH"]
        
        main_africa = adm2[adm2['GID_0'].isin(africa_iso3) & (adm2['GID_0'] != 'LBY')].copy()
        libya = adm1[adm1['GID_0'] == 'LBY'].copy()
        
        # Standardize columns for Africa
        main_africa = main_africa.rename(columns={'GID_0': 'ISO3', 'NAME_0': 'COUNTRY', 'NAME_2': 'ADM_NAME'})
        libya = libya.rename(columns={'GID_0': 'ISO3', 'NAME_0': 'COUNTRY', 'NAME_1': 'ADM_NAME'})
        
        return pd.concat([main_africa[['ISO3', 'COUNTRY', 'ADM_NAME', 'geometry']], 
                          libya[['ISO3', 'COUNTRY', 'ADM_NAME', 'geometry']]])

    elif mode == "ethiopia":
        adm2 = gpd.read_file(GADM_PATH, layer='ADM_2')
        eth = adm2[adm2['COUNTRY'] == 'Ethiopia'].copy()
        return eth.rename(columns={'GID_0': 'ISO3', 'NAME_2': 'ADM_NAME'})

    elif mode == "zambia":
        adm2 = gpd.read_file(GADM_PATH, layer='ADM_2')
        zambia_all = adm2[adm2['GID_0'] == 'ZMB'].copy()
        southern_districts = ['Chikankata', 'Chirundu', 'Choma', 'Gwembe', 'Itezhi-Tezhi', 'Kalomo', 'Kazungula', 'Livingstone', 'Mazabuka', 'Monze', 'Namwala', 'Pemba', 'Siavonga', 'Sinazongwe', 'Zimba']
        zmb_south = zambia_all[zambia_all['NAME_2'].isin(southern_districts)].copy()
        return zmb_south.rename(columns={'GID_0': 'ISO3', 'NAME_0': 'COUNTRY', 'NAME_2': 'ADM_NAME'})

# =============================================================================
# BLOCK 2: EARTH ENGINE METRICS (GLAD 2019)
# =============================================================================
def calculate_glad_metrics(gdf, mode):
    initialize_gee()
    print(f"🛰️ Calculating GLAD 2019 metrics for {mode}...")
    
    # Convert local map to GEE FeatureCollection
    geojson = json.loads(gdf.to_json())
    fc = ee.FeatureCollection(geojson)
    
    glad_2019 = ee.ImageCollection("users/potapovpeter/Global_cropland_2019").mosaic()

    def add_metrics(feature):
        geom = feature.geometry()
        # GLAD pixel value 1 = Cropland. multiply(pixelArea) gives area in m2
        stats = glad_2019.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(), 
            geometry=geom, 
            scale=50, 
            maxPixels=1e9, 
            bestEffort=True
        )
        crop_m2 = ee.Number(stats.get('b1', 0))
        total_m2 = geom.area(10)
        return feature.set({
            'cropland_km2': crop_m2.divide(1e6),
            'total_area_km2': total_m2.divide(1e6),
            'cropland_pct': crop_m2.divide(total_m2).multiply(100)
        })

    processed = fc.map(add_metrics)
    
    # Use the safe conversion function instead of geemap.ee_to_df
    return safe_ee_to_df(processed)

# =============================================================================
# BLOCK 3: RESEARCH FILTERING (The 95% Rule)
# =============================================================================
def apply_research_filter(df, mode):
    print(f"📊 Filtering districts for {mode}...")
    
    # Fill NaNs in cropland area with 0 to prevent sorting errors
    df['cropland_km2'] = df['cropland_km2'].fillna(0)
    
    if mode == "zambia":
        # We always keep all 15 Southern districts for the Zambia sub-study
        return df
    
    # Sort districts by cropland size
    df = df.sort_values(by=['COUNTRY', 'cropland_km2'], ascending=[True, False])
    
    # Calculate cumulative coverage per country
    df['total_country_crop'] = df.groupby('COUNTRY')['cropland_km2'].transform('sum')
    df['cum_sum'] = df.groupby('COUNTRY')['cropland_km2'].cumsum()
    df['cum_pct'] = df['cum_sum'] / df['total_country_crop']
    
    # Keep top 95% of national cropland coverage OR any district with >30% density
    return df[(df['cum_pct'] <= 0.95) | (df['cropland_pct'] >= 30)].copy()

# =============================================================================
# BLOCK 4: RAINFALL EXTRACTION (CHIRPS) - UTILITY ONLY
# =============================================================================
def export_chirps_rainfall(year_range):
    initialize_gee()
    print(f"🌧️ Submitting Rainfall Export Tasks for years {year_range}...")
    
    # Important: This assumes you have already uploaded your filtered districts 
    # as an asset called 'africa-glad-50-2019' in your GEE account.
    asset_id = f"projects/{PROJECT_ID}/assets/africa-glad-50-2019"
    agri_districts = ee.FeatureCollection(asset_id)
    chirps_daily = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select('precipitation')

    for year in year_range:
        months = ee.List.sequence(1, 12)
        def calc_mo(m):
            monthly_total = chirps_daily.filter(ee.Filter.calendarRange(year, year, 'year')).filter(ee.Filter.calendarRange(m, m, 'month')).sum()
            stats = monthly_total.reduceRegions(collection=agri_districts, reducer=ee.Reducer.mean(), scale=5566)
            return stats.map(lambda f: f.set({'month': m, 'year': year, 'precip_mm': f.get('mean')}))
        
        yearly_stats = ee.FeatureCollection(months.map(calc_mo)).flatten()
        task = ee.batch.Export.table.toDrive(
            collection=yearly_stats, 
            description=f"CHIRPS_Rainfall_{year}", 
            folder="Agri_Drought_Study_Data",
            fileNamePrefix=f"rainfall_{year}", 
            fileFormat='CSV',
            selectors=['ISO3', 'COUNTRY', 'ADM_NAME', 'year', 'month', 'precip_mm']
        )
        task.start()
        print(f"   🚀 GEE task started for year: {year}")

# =============================================================================
# MAIN EXECUTION CONTROL
# =============================================================================
if __name__ == "__main__":
    
    # STEP 1: Process administrative boundaries
    boundaries = reconstruct_boundaries(RUN_MODE)
    
    # STEP 2: Google Earth Engine cropland metrics calculation
    raw_results = calculate_glad_metrics(boundaries, RUN_MODE)
    
    # STEP 3: Apply the 95% scientific filter
    final_filtered_df = apply_research_filter(raw_results, RUN_MODE)
    
    # STEP 4: Save final results
    output_filename = OUTPUT_DIR / f"{RUN_MODE}_final_study_districts.csv"
    final_filtered_df.to_csv(output_filename, index=False)
    
    print("\n" + "="*40)
    print(f"✅ SUCCESS: Mode '{RUN_MODE}' Complete.")
    print(f"📍 Districts saved: {len(final_filtered_df)}")
    print(f"📂 File: {output_filename}")
    print("="*40)

    # Optional: Run rainfall export for all years (2000-2024)
    # export_chirps_rainfall(range(2000, 2025))
    
    print("\n✨ All operations finished.")