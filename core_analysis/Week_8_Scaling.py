#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD ERA5 SOIL MOISTURE DATA (CSV)
era5_path = r"data\Africa_Agri_districts_ERA5_LAND_DAILY_AGGR_2000_2026_timeseries.csv"
df_era5 = pd.read_csv(era5_path)

# 2. LOAD GADM DISTRICT SHAPEFILE (.shp)
gadm_path = r"data\africa_agricultural_domain_2019\africa_agricultural_domain_2019.shp"
gdf_districts = gpd.read_file(gadm_path)

# 3. STANDARDIZE AND OPTIMIZE
df_era5['feature_id'] = df_era5['feature_id'].astype(str).str.strip().str.upper()
gdf_districts['ADM_NAME'] = gdf_districts['ADM_NAME'].astype(str).str.strip().str.upper()

# Simplify geometry
gdf_districts['geometry'] = gdf_districts['geometry'].simplify(tolerance=0.02, preserve_topology=True)

# 4. PRINT DATA SIZE / STRUCTURE
print("----- ERA5 DATA -----")
print(f"Rows: {df_era5.shape[0]}")
print(f"Columns: {df_era5.shape[1]}")
print("Column names:")
print(df_era5.columns.tolist())
print("\nData types:")
print(df_era5.dtypes)
print("\nMemory usage (MB):")
print(df_era5.memory_usage(deep=True).sum() / 1024**2)
print("\nFirst 5 rows:")
print(df_era5.head())

print("\n----- SHAPEFILE DATA -----")
print(f"Rows (polygons): {gdf_districts.shape[0]}")
print(f"Columns: {gdf_districts.shape[1]}")
print("Column names:")
print(gdf_districts.columns.tolist())
print("\nCRS:")
print(gdf_districts.crs)
print("\nGeometry types:")
print(gdf_districts.geom_type.value_counts())
print("\nBounds:")
print(gdf_districts.total_bounds)
print("\nFirst 5 rows:")
print(gdf_districts.head())

# 5. PLOT SHAPEFILE
fig, ax = plt.subplots(figsize=(12, 10))
gdf_districts.plot(ax=ax, edgecolor='black', linewidth=0.2, color='lightgreen')
ax.set_title("Agricultural District Shapefile")
ax.set_axis_off()
plt.show()


# In[2]:


import geopandas as gpd

# 1. Load your shapefiles
district_shp = gpd.read_file(r"data\africa_agricultural_domain_2019\africa_agricultural_domain_2019.shp") # Your ag districts
geoglam_shp = gpd.read_file(r"./Notebooks\GEOGLAM_CM4EW_Calendars_V1.4\GEOGLAM_CM4EW_Calendars_V1.4\GEOGLAM_CM4EW_Calendars_V1.4.shp")  # GEOGLAM calendar

# Make sure they share the same Coordinate Reference System (CRS)
district_shp = district_shp.to_crs(geoglam_shp.crs)

# 2. Convert your districts into points (Centroids) for the spatial join
district_centroids = district_shp.copy()

# Note: Sometimes calculating centroids gives a warning about projected CRSs. 
# We use .centroid directly here.
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    district_centroids['geometry'] = district_centroids.centroid

# 3. Perform the Spatial Join
# This attaches the GEOGLAM data to your centroid points
joined_centroids = gpd.sjoin(district_centroids, geoglam_shp, how="left", predicate="within")

# 4. Re-attach the original district Polygon geometry 
# THE FIX: Instead of a complex join, we just replace the 'geometry' column 
# in our joined dataset with the original polygon geometries from district_shp!
joined_centroids['geometry'] = district_shp['geometry']
final_joined_shp = joined_centroids

# 5. Extract ONLY the Maize Districts
# Note: Check the exact column name in your GEOGLAM shapefile. 
# It might be 'Crop Type', 'Crop', or 'crop_name'. Change it below if needed.
crop_column_name = 'crop' # Update this if your column is named differently

maize_districts = final_joined_shp[
    final_joined_shp[crop_column_name].str.contains('Maize', case=False, na=False)
]

# Plot to verify
maize_districts.plot(column=crop_column_name, legend=True, figsize=(10,10), cmap='Set1')


# In[3]:


# Look at the columns to find the names of the Start and End dates
print(maize_districts.columns)


# In[4]:


# (Replace 'Start_Column' and 'End_Column' with the actual names from your output)
print(maize_districts[['ADM_NAME', 'planting', 'harvest']].head())


# In[5]:


# Check what format the dates are in
print(maize_districts[['ADM_NAME', 'crop', 'planting', 'harvest', 'endofseaso']].head(10))
# Look at the actual data format


# In[6]:


import pandas as pd
import numpy as np

# 1. Filter OUT the districts where planting or harvest is 0.0
# This leaves us with only the true Maize-growing districts
maize_active = maize_districts[(maize_districts['planting'] > 0) & (maize_districts['harvest'] > 0)].copy()

# 2. Calculate the total length of the growth period
# We use modulo 365 (%) because sometimes planting is in November (Day 300) 
# and harvest is in February (Day 60) of the next year.
maize_active['Season_Length'] = (maize_active['harvest'] - maize_active['planting']) % 365

# 3. Calculate the 25% delay (how many days to skip)
maize_active['Delay_Days'] = (maize_active['Season_Length'] * 0.25).astype(int)

# 4. Calculate the Adjusted Start Date
maize_active['Adjusted_Start'] = (maize_active['planting'] + maize_active['Delay_Days']) % 365

# Let's look at the cleaned data with our new adjusted dates!
print(maize_active[['ADM_NAME', 'crop', 'planting', 'Adjusted_Start', 'harvest', 'Season_Length']].head(10))


# In[7]:


import pandas as pd

era5_path = r"data\Africa_Agri_districts_ERA5_LAND_DAILY_AGGR_2000_2026_timeseries.csv"

# Load the dataset using pandas
era5_df = pd.read_csv(era5_path)

# Print out the columns and the first few rows
print("--- ERA5 Data Columns ---")
print(era5_df.info())

print("\n--- First 5 Rows ---")
print(era5_df.head())


# In[8]:


import xarray as xr

# 1. Convert your DataFrame back into an xarray Dataset
# This creates a new 'ds' object from the 'era5_df' you already have
ds = era5_df.to_xarray()

# 2. Set the variable name for Layer 2 (ERA5 short name)
soil_moisture_var = 'volumetric_soil_water_layer_2'


# In[9]:


import geopandas as gpd

# Load your agricultural shapefile
# Replace 'path_to_your_shp.shp' with your actual file path
districts_gdf = gpd.read_file(r"data\africa_agricultural_domain_2019\africa_agricultural_domain_2019.shp")

# Print columns to find the district name column (e.g., 'DISTRICT', 'ADM2_EN', etc.)
print("Shapefile columns:", districts_gdf.columns)
print("ERA5 columns:", era5_df.columns)


# In[10]:


# 1. Rename the column in era5_df to match the shapefile
# Replace 'old_name' with your current ERA5 district column name
# Replace 'district_name' with the column name from your shapefile
era5_df = era5_df.rename(columns={'feature_id': 'ADM_NAME'})

# 2. Merge the ERA5 data with the Shapefile
# This adds the 'geometry' (polygons) to your soil moisture data
merged_gdf = districts_gdf.merge(era5_df, on='ADM_NAME')

# Check the first few rows
print(merged_gdf.head())


# In[11]:


import matplotlib.pyplot as plt
import pandas as pd

# 1. Ensure the 'date' column is recognized as a date by Python
merged_gdf['date'] = pd.to_datetime(merged_gdf['date'])

# 2. Pick the first date in your dataset
first_date = merged_gdf['date'].min()
first_day_data = merged_gdf[merged_gdf['date'] == first_date]

# 3. Plot using the actual column names from your output
fig, ax = plt.subplots(figsize=(12, 10))

# We use 'volumetric_soil_water_layer_2' because that is what is in your print output
first_day_data.plot(
    column='volumetric_soil_water_layer_2', 
    ax=ax, 
    legend=True, 
    cmap='YlGnBu',
    legend_kwds={'label': "Volumetric Soil Water (m³/m³)"}
)

plt.title(f"ERA5 Soil Moisture by District - {first_date.strftime('%Y-%m-%d')}")
plt.axis('off')
plt.savefig('Volumetric Soil Water.png', dpi=300)
plt.show()


# In[ ]:




