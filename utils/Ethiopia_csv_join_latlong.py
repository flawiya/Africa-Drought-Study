import geopandas as gpd
import pandas as pd
import os

# ==========================================
# 1. DEFINE PATHS (Check these carefully!)
# ==========================================
# I have added the 'ethiopia\output' subfolder which was used in your previous scripts
shp_path = r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk\Drought\Output\ethiopia\output\africa_agricultural_domain_2019.shp"
csv_input = r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk\Drought\Output\ethiopia\content\ethiopia_rainfall_master_cleaned.csv"
csv_output = r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk\Drought\Output\ethiopia\content\ethiopia_rainfall_master_cleaned_with_latlon.csv"

# ==========================================
# 2. PATH VERIFICATION (Debug Step)
# ==========================================
if not os.path.exists(shp_path):
    print(f"❌ ERROR: Shapefile not found at: {shp_path}")
    print("Please check if the file is inside a subfolder or if the name is correct.")
    # Stop the script if file is missing
    exit()
else:
    print("✅ Shapefile found! Proceeding...")

# ==========================================
# 3. LOAD AND CALCULATE ACCURATE COORDINATES
# ==========================================
# Load districts
districts = gpd.read_file(shp_path)

# Logic Fix: To get accurate centroids, we project to meters (UTM), 
# calculate the center, then project back to degrees (WGS84).
districts_utm = districts.to_crs(epsg=32637) # Ethiopia UTM
districts_utm['centroid'] = districts_utm.geometry.centroid

# Convert ONLY the centroids back to WGS84 degrees
districts_wgs84 = districts_utm.set_geometry('centroid').to_crs(epsg=4326)

# Extract lat/lon
districts_wgs84['lat_wgs84'] = districts_wgs84.geometry.y
districts_wgs84['lon_wgs84'] = districts_wgs84.geometry.x

# Create a clean lookup table
centroids = districts_wgs84[['ADM_NAME', 'lat_wgs84', 'lon_wgs84']].drop_duplicates()

# ==========================================
# 4. MERGE WITH RAINFALL DATA
# ==========================================
print("Merging data...")
df = pd.read_csv(csv_input)
df_merged = df.merge(centroids, on='ADM_NAME', how='left')

# Save
df_merged.to_csv(csv_output, index=False)

print(f"✅ SUCCESS! File saved to: {csv_output}")
print(df_merged[['ADM_NAME', 'lat_wgs84', 'lon_wgs84']].head())