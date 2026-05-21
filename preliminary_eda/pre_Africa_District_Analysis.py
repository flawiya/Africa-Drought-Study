#!/usr/bin/env python
# coding: utf-8

# In[1]:


# STEP 1: Install libraries (Colab needs this every time)
get_ipython().system('pip install geemap earthengine-api pandas')

import ee
import geemap
import pandas as pd
from google.colab import drive

# STEP 2: Enter your Project ID here
# Your colleague will need to put THEIR project ID here
MY_PROJECT_ID = 'vernal-parser-412016'

try:
    ee.Initialize(project=MY_PROJECT_ID)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=MY_PROJECT_ID)

# STEP 3: The Analysis
print("Loading Africa Districts and Land Cover...")

africa_bounds = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(ee.Filter.eq('wld_rgn', 'Africa'))
districts = ee.FeatureCollection("FAO/GAUL/2015/level2").filterBounds(africa_bounds)
worldcover = ee.ImageCollection("ESA/WorldCover/v100").first()
cropland_mask = worldcover.eq(40)

print("Calculating statistics (1-2 minutes)...")
districts_with_area = districts.map(lambda f: f.set('total_area_km2', f.geometry().area().divide(1e6)))

stats = cropland_mask.reduceRegions(
    collection=districts_with_area,
    reducer=ee.Reducer.sum(),
    scale=100
)

# Download clean columns
cols = ['ADM0_NAME', 'ADM2_NAME', 'ADM2_CODE', 'sum', 'total_area_km2']
df = geemap.ee_to_df(stats.select(cols, retainGeometry=False))

# Math Logic
df['crop_area_km2'] = df['sum'] * 0.01
df['density'] = df['crop_area_km2'] / df['total_area_km2']
df = df[df['crop_area_km2'] > 0].copy()
df['total_country_crop'] = df.groupby('ADM0_NAME')['crop_area_km2'].transform('sum')
df = df.sort_values(['ADM0_NAME', 'crop_area_km2'], ascending=[True, False])
df['cum_sum'] = df.groupby('ADM0_NAME')['crop_area_km2'].cumsum()
df['cum_pct'] = df['cum_sum'] / df['total_country_crop']

# Final Filter
selected_df = df[(df['cum_pct'] <= 0.95) | (df['density'] >= 0.50)]

# STEP 4: Save and Download
selected_df.to_csv("africa_study_districts.csv", index=False)
print(f"Success! Saved {len(selected_df)} districts.")

# This line pops up a download window in the browser
from google.colab import files
files.download('africa_study_districts.csv')


# In[ ]:




