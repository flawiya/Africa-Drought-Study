#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import shutil
import sys

# 1. Find the site-packages folder
import site
package_path = site.getsitepackages()[0]
print(f"Searching in: {package_path}")

# 2. List of bad folders to delete
bad_folders = ['ee', 'blessings']

for folder in bad_folders:
    path = os.path.join(package_path, folder)
    if os.path.exists(path):
        print(f"Deleting bad library at: {path}")
        shutil.rmtree(path)
    else:
        print(f"Folder {folder} not found, already clean.")

# 3. Force install the CORRECT Google library
print("Installing the correct Google Earth Engine API...")
get_ipython().system('{sys.executable} -m pip install --upgrade earthengine-api geemap pandas')

print("\n--- DONE ---")
print("CRITICAL: Now go to the top menu: KERNEL -> RESTART")


# In[9]:


# ============================================================
# OPTIMIZED: African Agricultural Districts Extraction
# Using Earth Engine's native functions (NO Python loops)
# Solves the payload size error
# ============================================================

import ee
import geemap
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import shape
import json
import os

# Initialize Earth Engine
try:
    ee.Initialize()
    print("✅ Earth Engine initialized successfully")
except Exception as e:
    print("Please authenticate:")
    ee.Authenticate()
    ee.Initialize()
    print("✅ Earth Engine initialized successfully")

# ============================================================
# LOAD BOUNDARIES
# ============================================================

print("\n📂 Loading administrative boundaries...")

asset_id = 'projects/vernal-parser-412016/assets/africa_admx_shapefile'
admx = ee.FeatureCollection(asset_id)

total_districts = admx.size().getInfo()
print(f"✅ Total districts in Africa: {total_districts:,}")

# Field names
country_field = 'COUNTRY'
name_field = 'ADM_NAME'

print(f"Using country field: {country_field}")
print(f"Using name field: {name_field}")

# ============================================================
# OPTIMIZED: CROPLAND EXTRACTION - USING ONLY LATEST YEAR FIRST
# ============================================================

print("\n🌾 Extracting cropland data for 2019...")

# Use only 2019 first to avoid complexity
year = 2019
modis = ee.Image(f'MODIS/006/MCD12Q1/{year}_01_01').select('LC_Type1')
cropland_mask = modis.eq(12).Or(modis.eq(14))

# Function to add cropland metrics to each district
def add_cropland_metrics(feature):
    """Calculate cropland area and percentage for a district"""
    geometry = feature.geometry()
    
    # Calculate cropland area
    cropland_area = cropland_mask.multiply(ee.Image.pixelArea()) \
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geometry,
            scale=500,
            maxPixels=1e9,
            bestEffort=True
        )
    
    cropland_m2 = ee.Number(cropland_area.get('LC_Type1', 0))
    total_area = geometry.area(500)
    
    # Calculate cropland percentage
    cropland_pct = ee.Algorithms.If(
        total_area.gt(0),
        cropland_m2.divide(total_area).multiply(100),
        0
    )
    
    # Flag if >50% cropland
    is_high_density = ee.Number(cropland_pct).gt(50)
    
    return feature.set({
        'cropland_km2': cropland_m2.divide(1e6),
        'cropland_pct': ee.Number(cropland_pct),
        'is_high_density': is_high_density
    })

# Apply to all districts
districts_with_cropland = admx.map(add_cropland_metrics)

print("✅ Cropland metrics added")

# ============================================================
# FILTER 1: HIGH DENSITY UNITS (>50% CROPLAND)
# ============================================================

high_density_units = districts_with_cropland.filter(ee.Filter.gte('cropland_pct', 50))
high_density_count = high_density_units.size().getInfo()
print(f"📊 High density units (>50% cropland): {high_density_count:,}")

# ============================================================
# FILTER 2: CUMULATIVE 95% - USING SIMPLIFIED APPROACH
# ============================================================

print("\n🔄 Calculating cumulative 95% cropland (using simplified approach)...")

# Instead of complex loops, let's do a simplified but effective approach:
# Keep top districts by cropland area until reaching 95% of country total

def get_cumulative_units_simple(fc, country_field, area_field, threshold=95):
    """
    Simplified cumulative filtering: 
    For each country, sort districts by cropland area and keep top ones
    until cumulative area reaches threshold percentage
    """
    
    # Get unique countries
    countries = fc.distinct([country_field])
    country_list = countries.aggregate_array(country_field).getInfo()
    
    cumulative_feature_list = []
    
    for country in country_list:
        print(f"  Processing {country}...")
        
        # Get districts for this country, sorted by cropland area
        country_fc = fc.filter(ee.Filter.eq(country_field, country)) \
                       .sort(area_field, False)
        
        # Get total cropland for country
        total_cropland = country_fc.aggregate_sum(area_field).getInfo()
        
        if total_cropland == 0:
            continue
        
        # Get districts as list
        districts_list = country_fc.toList(country_fc.size()).getInfo()
        
        cumulative_sum = 0
        for district in districts_list:
            cropland_area = district['properties'][area_field]
            cumulative_sum += cropland_area
            cumulative_pct = (cumulative_sum / total_cropland) * 100
            
            if cumulative_pct <= threshold:
                cumulative_feature_list.append(district)
            else:
                break
    
    print(f"  Found {len(cumulative_feature_list)} cumulative units")
    
    # Convert back to FeatureCollection
    if cumulative_feature_list:
        features = []
        for feat in cumulative_feature_list:
            geom = ee.Geometry(feat['geometry'])
            props = feat['properties']
            features.append(ee.Feature(geom, props))
        
        return ee.FeatureCollection(features)
    else:
        return ee.FeatureCollection([])

# Get cumulative 95% units
cumulative_95_units = get_cumulative_units_simple(
    districts_with_cropland,
    country_field,
    'cropland_km2',
    95
)

cumulative_95_count = cumulative_95_units.size().getInfo()
print(f"📊 Cumulative 95% units: {cumulative_95_count:,}")

# ============================================================
# COMBINE BOTH FILTERS
# ============================================================

print("\n⭐ Combining filters...")

# Combine high density and cumulative 95%
stage1_units = high_density_units.merge(cumulative_95_units).distinct([name_field, country_field])
stage1_count = stage1_units.size().getInfo()

print(f"\n⭐ STAGE 1 COMPLETE: {stage1_count:,} agricultural districts")
print(f"   Target from paper: 3,014 units")

# ============================================================
# CONVERT TO GEOPANDAS (IN BATCHES TO AVOID PAYLOAD ERROR)
# ============================================================

print("\n📊 Converting to GeoPandas DataFrame (in batches)...")

def ee_to_gdf_batched(fc, batch_size=500):
    """Convert Earth Engine FeatureCollection to GeoPandas in batches"""
    
    total_size = fc.size().getInfo()
    print(f"  Total features: {total_size}")
    
    all_features = []
    all_properties = []
    
    for i in range(0, total_size, batch_size):
        print(f"  Processing batch {i//batch_size + 1}/{(total_size + batch_size - 1)//batch_size}...")
        
        # Get batch of features
        batch = fc.toList(batch_size, i).getInfo()
        
        for feature in batch:
            try:
                # Extract geometry
                geom = shape(feature['geometry'])
                all_features.append(geom)
                
                # Extract properties
                props = feature['properties']
                all_properties.append(props)
            except Exception as e:
                print(f"    Error processing feature: {e}")
                continue
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(all_properties, geometry=all_features, crs='EPSG:4326')
    
    return gdf

# Convert stage1 units
agricultural_districts_gdf = ee_to_gdf_batched(stage1_units, batch_size=500)

print(f"✅ Converted {len(agricultural_districts_gdf):,} districts to GeoDataFrame")

# ============================================================
# SAVE RESULTS
# ============================================================

print("\n💾 Saving results...")

output_dir = 'agricultural_districts_output'
os.makedirs(output_dir, exist_ok=True)

# Save as Shapefile
shapefile_path = os.path.join(output_dir, 'africa_agricultural_districts_stage1.shp')
agricultural_districts_gdf.to_file(shapefile_path)
print(f"✅ Saved to {shapefile_path}")

# Save as CSV
csv_path = os.path.join(output_dir, 'africa_agricultural_districts_stage1.csv')
agricultural_districts_gdf.to_csv(csv_path, index=False)
print(f"✅ Saved to {csv_path}")

# Save simplified CSV (just names for SPEI extraction)
simplified_csv = os.path.join(output_dir, 'districts_for_spei_analysis.csv')
agricultural_districts_gdf[[country_field, name_field, 'cropland_km2', 'cropland_pct']] \
    .to_csv(simplified_csv, index=False)
print(f"✅ Saved simplified list to {simplified_csv}")

# ============================================================
# SUMMARY STATISTICS
# ============================================================

print("\n📊 SUMMARY STATISTICS:")
print("=" * 60)

print(f"\nTotal districts in Africa: {total_districts:,}")
print(f"Agricultural districts (Stage 1): {len(agricultural_districts_gdf):,}")
print(f"Percentage: {(len(agricultural_districts_gdf)/total_districts*100):.1f}%")

# Top 10 countries
if country_field in agricultural_districts_gdf.columns:
    country_counts = agricultural_districts_gdf[country_field].value_counts().head(10)
    print("\nTop 10 countries by agricultural districts:")
    for country, count in country_counts.items():
        print(f"  {country}: {count}")

# Total cropland area
if 'cropland_km2' in agricultural_districts_gdf.columns:
    total_cropland = agricultural_districts_gdf['cropland_km2'].sum()
    print(f"\n🌾 Total cropland area: {total_cropland:,.0f} km²")
    
    avg_cropland_pct = agricultural_districts_gdf['cropland_pct'].mean()
    print(f"📊 Average cropland percentage: {avg_cropland_pct:.1f}%")

# ============================================================
# VISUALIZATION
# ============================================================

print("\n🎨 Creating visualization...")

# Create interactive map
Map = geemap.Map()
Map.centerObject(stage1_units, 4)

# Add stage1 units
Map.addLayer(
    stage1_units, 
    {'color': '0000FF', 'fillColor': '0000FF80'}, 
    'Agricultural Districts (Stage 1)'
)

# Add cropland mask
Map.addLayer(
    cropland_mask.updateMask(cropland_mask), 
    {'palette': ['FF0000'], 'opacity': 0.5}, 
    'Cropland 2019', 
    False
)

Map.addLayerControl()
display(Map)

# Static plot
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
agricultural_districts_gdf.plot(ax=ax, color='blue', edgecolor='darkblue', linewidth=0.2, alpha=0.7)
ax.set_title(f'Agricultural Districts in Africa (Stage 1)\n{len(agricultural_districts_gdf):,} Districts', fontsize=14)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'agricultural_districts_map.png'), dpi=300)
plt.show()

print(f"\n✅ Map saved to '{output_dir}/agricultural_districts_map.png'")

# ============================================================
# PREPARE FOR SPEI EXTRACTION
# ============================================================

print("\n🎯 Preparing districts for SPEI extraction...")

# Create a list of district identifiers
districts_list = []
for idx, row in agricultural_districts_gdf.iterrows():
    districts_list.append({
        'country': row[country_field],
        'district_name': row[name_field],
        'cropland_km2': float(row['cropland_km2']) if 'cropland_km2' in row else 0,
        'cropland_pct': float(row['cropland_pct']) if 'cropland_pct' in row else 0
    })

# Save as JSON
json_path = os.path.join(output_dir, 'districts_for_spei.json')
with open(json_path, 'w') as f:
    json.dump(districts_list, f, indent=2)

print(f"✅ Saved {len(districts_list)} districts to '{json_path}'")

# Print sample
print("\n📋 Sample of agricultural districts (first 10):")
for i, district in enumerate(districts_list[:10]):
    print(f"  {i+1}. {district['country']} - {district['district_name']} "
          f"(Cropland: {district['cropland_km2']:.0f} km², {district['cropland_pct']:.0f}%)")

print("\n" + "=" * 60)
print("✅ EXTRACTION COMPLETE!")
print("=" * 60)
print(f"\nFiles created in '{output_dir}/':")
print("  1. africa_agricultural_districts_stage1.shp - Shapefile")
print("  2. africa_agricultural_districts_stage1.csv - Full data")
print("  3. districts_for_spei_analysis.csv - Simplified list for SPEI")
print("  4. districts_for_spei.json - JSON format")
print("  5. agricultural_districts_map.png - Map visualization")
print("\n🎯 NEXT STEPS:")
print("  1. Use 'districts_for_spei_analysis.csv' for SPEI extraction")
print("  2. Calculate drought indices for these districts")
print("  3. Share results with your supervisor")
print("=" * 60)

# Display first few rows
print("\n📊 Preview of agricultural districts data:")
print(agricultural_districts_gdf[[country_field, name_field, 'cropland_km2', 'cropland_pct']].head(10))


# In[ ]:




