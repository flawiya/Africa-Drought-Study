#!/usr/bin/env python
# coding: utf-8

# In[1]:


import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import gamma, norm
import warnings
import os
warnings.filterwarnings('ignore')


# In[2]:


import geopandas as gpd
import pandas as pd
import numpy as np

# Now define your file path and load GADM data
gadm_file_path = r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk\Drought\data\gadm_410-levels\gadm_410-levels.gpkg"

print("Loading GADM data...")
# Read the Geopackage using the exact layer name for districts (ADM_2)
gadm_data = gpd.read_file(gadm_file_path, layer='ADM_2')

# Filter the dataset to extract ONLY Ethiopia
print("Filtering for Ethiopia...")
ethiopia_districts = gadm_data[gadm_data['COUNTRY'] == 'Ethiopia']

print(f"✅ Loaded {len(ethiopia_districts)} Ethiopian districts")
print("Columns:", ethiopia_districts.columns.tolist())
print(ethiopia_districts.head())


# In[3]:


# Path to your other shapefile
agri_district_path = (r"C:/Users/FlawiyaShirishMore/OneDrive - Africa Specialty Risks Ltd/ASR-Parametric_Research_Study/africa_risk/Drought/Output/ethiopia/output/africa_agricultural_domain_2019.shp")
# Load the shapefile
agri_district = gpd.read_file(agri_district_path)

# Ensure the second shapefile has the exact same coordinate system as the GADM data
if agri_district.crs != ethiopia_districts.crs:
    print("Aligning coordinate systems...")
    agri_district = agri_district.to_crs(ethiopia_districts.crs)    


# In[4]:


agri_district.head()


# In[5]:


import pandas as pd

# Official Zambia Southern Province 15 districts
southern_province_districts = [
    'Chikankata', 'Chirundu', 'Choma', 'Gwembe', 'Itezhi-Tezhi', 
    'Kalomo', 'Kazungula', 'Livingstone', 'Mazabuka', 'Monze', 
    'Namwala', 'Pemba', 'Siavonga', 'Sinazongwe', 'Zimba'
]

# Load your agri_district shapefile (already loaded as df presumably)
# df = gpd.read_file('agri_district.shp')  # if not already loaded

# Filter for Zambia only
zambia_df = agri_district[agri_district['ISO3'] == 'ZMB']

# Find exact matches
exact_matches = zambia_df[zambia_df['ADM_NAME'].isin(southern_province_districts)]['ADM_NAME'].unique()

print("=== SOUTHERN PROVINCE 15 DISTRICTS MATCHING ===")
print(f"Target districts: 15")
print(f"Found in agri_district file: {len(exact_matches)}")
print(f"\nMATCHING NAMES:")
print(sorted(exact_matches))
print(f"\nMISSING NAMES ({15-len(exact_matches)}):")
missing = [d for d in southern_province_districts if d not in exact_matches]
print(missing)

# Show first few Zambia district names to check spelling
print(f"\nFirst 10 Zambia ADM_NAMEs in your file:")
print(zambia_df['ADM_NAME'].head(10).tolist())


# In[6]:


# Clip the 'other_data' so it only contains features inside 'ethiopia_districts'
ethiopia_other_data = gpd.clip(agri_district, ethiopia_districts)


# In[7]:


# Create a figure and axis for the plot
fig, ax = plt.subplots(figsize=(12, 12))

# 1. Plot the Ethiopia District Boundaries (Base map)
# facecolor='none' makes the inside empty, edgecolor='black' draws the borders
ethiopia_districts.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5, label='Districts')

# 2. Plot the filtered shapefile data on top
# You can customize the color. If it's point data, adjust markersize. 
# If it's polygon data, adjust facecolor/alpha.
ethiopia_other_data.plot(ax=ax, color='#4a7c59', markersize=10, alpha=0.7, label='My Data')

ax.annotate('N', xy=(0.87, 1.0), xytext=(0.87, 0.93),
                arrowprops=dict(facecolor='black', width=5, headwidth=15),
                ha='center', va='center', fontsize=15, 
                xycoords='axes fraction')

#plt.title("Cropland Map of Ethiopia on District scale", fontsize=15)
ax.set_axis_off()

# CAPTION at bottom (perfect placement y=0.02)
plt.figtext(0.5, 0.01, "Figure 1: Cropland Distribution, Ethiopian Districts (Admin 2)",
            ha='center', fontsize=10, style='italic', wrap=True)

# Full path - SAVE BEFORE SHOW
plt.savefig(r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk\Drought\Output\ethiopia_filtered_data.png", 
            dpi=300)

# Show the map
plt.show()

# 1. Filter out the line/point artifacts created during clipping
# Keep ONLY polygons and multipolygons
allowed_geometries = ['Polygon', 'MultiPolygon']
ethiopia_other_data_clean = ethiopia_other_data[ethiopia_other_data.geometry.geom_type.isin(allowed_geometries)]

# 2. Now try saving it again!
ethiopia_other_data_clean.to_file(r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk\Drought\Output\ethiopia_filtered_data.shp")


# In[8]:


print(f"Your district count: {len(ethiopia_districts)}")
print("Sample districts:", ethiopia_districts['NAME_2'].head().tolist())


# In[ ]:




