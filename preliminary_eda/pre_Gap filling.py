#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install earthengine-api geemap


# In[2]:


import ee
import os

# Check the file path of the 'ee' module
print(f"EE Location: {ee.__file__}")

# Check if it has the right attributes
if hasattr(ee, 'FeatureCollection'):
    print("✅ SUCCESS: You have the official Google Earth Engine API!")
else:
    print("❌ FAILURE: You still have the wrong 'ee' package installed.")


# In[21]:


import geopandas as gpd
import pandas as pd
import fiona

# 1. Point to your GADM GeoPackage or Folder
gpkg_path = 'C:/Users/FlawiyaShirishMore/OneDrive - Africa Specialty Risks Ltd/ASR-Parametric_Research_Study/africa_risk/data/gadm_410-levels/gadm_410-levels.gpkg' # Update this to your filename

# Define Africa ISO3 list (Top-level)
africa_iso3 = [
    "DZA","AGO","BEN","BWA","BFA","BDI","CMR","CPV","CAF","TCD","COM","COG","COD",
    "DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","CIV","KEN",
    "LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA",
    "RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA",
    "ZMB","ZWE","ESH"
]

# 2. Load ADM2 and ADM1
print("🔍 Loading GADM Layers...")
adm2 = gpd.read_file(gpkg_path, layer='ADM_2')
adm1 = gpd.read_file(gpkg_path, layer='ADM_1')

# 3. Create the Smart Selection Logic
# We want ADM2 for everyone EXCEPT Libya (LBY) and Western Sahara (ESH) 
# or countries where ADM2 might be empty.

# Filter ADM2 for Africa (Excluding Libya)
adm2_africa = adm2[adm2['GID_0'].isin(africa_iso3) & (adm2['GID_0'] != 'LBY')].copy()

# Filter ADM1 for Libya only
libya_adm1 = adm1[adm1['GID_0'] == 'LBY'].copy()

# 4. Standardize Columns to Merge them
# We want a common schema: [ISO3, ADM_NAME, ADM_LEVEL, GEOMETRY]
adm2_africa['ADM_NAME'] = adm2_africa['NAME_2']
adm2_africa['ADM_LEVEL'] = 2

libya_adm1['ADM_NAME'] = libya_adm1['NAME_1']
libya_adm1['ADM_LEVEL'] = 1

# Combine them
skeleton = pd.concat([
    adm2_africa[['GID_0', 'ADM_NAME', 'ADM_LEVEL', 'geometry']], 
    libya_adm1[['GID_0', 'ADM_NAME', 'ADM_LEVEL', 'geometry']]
])

skeleton = gpd.GeoDataFrame(skeleton, crs=adm2.crs)
skeleton.to_file("Africa_Full_Coverage_Skeleton.shp")
print("✅ Created Africa_Full_Coverage_Skeleton.shp with 0 gaps.")


# In[22]:


import pandas as pd
import geopandas as gpd

# 1. Load your master rainfall CSV
df_current = pd.read_csv('master_rainfall_africa.csv')

# --- FIX CASE SENSITIVITY ---
# Print columns just to be sure what we have
print("Original CSV Columns:", df_current.columns.tolist())

# Force all CSV columns to lowercase
df_current.columns = df_current.columns.str.lower()

# Check for 'adm_name' again. If it's still missing, it might be 'name_2' or 'location'
if 'adm_name' not in df_current.columns:
    # Look for common GEE/GADM alternatives if 'adm_name' isn't there
    for alt in ['adm2_name', 'name_2', 'location', 'district']:
        if alt in df_current.columns:
            df_current = df_current.rename(columns={alt: 'adm_name'})
            break

# Standardize the data values (Upper case and No spaces)
df_current['adm_name'] = df_current['adm_name'].astype(str).str.upper().str.strip()
unique_districts_in_data = df_current['adm_name'].unique()

# --- PREPARE THE SKELETON ---
# Make sure the skeleton we built also uses 'adm_name' in uppercase for the match
# 'skeleton' comes from the previous step
skeleton.columns = skeleton.columns.str.upper() 
skeleton['ADM_NAME'] = skeleton['ADM_NAME'].astype(str).str.upper().str.strip()

# 2. IDENTIFY THE HOLES
# Find polygons in Skeleton that are NOT in our unique_districts_in_data list
missing_polygons = skeleton[~skeleton['ADM_NAME'].isin(unique_districts_in_data)]

# 3. VISUALIZE THE GAPS (To show your prof you found the errors)
print(f"📍 Total Administrative Units in Skeleton: {len(skeleton)}")
print(f"📍 Units with Rainfall Data: {len(unique_districts_in_data)}")
print(f"📍 WHITE HOLES (Gaps) Found: {len(missing_polygons)}")

if len(missing_polygons) > 0:
    # Save the gaps for GEE extraction
    missing_polygons.to_file("Missing_Gaps_For_GEE.shp")
    
    # Quick Plot to see where the holes are
    ax = skeleton.plot(color='lightgrey', edgecolor='white', figsize=(15, 12))
    missing_polygons.plot(ax=ax, color='red')
    plt.title("Red areas show 'White Holes' (Missing Rainfall Data)", fontsize=15)
    plt.show()
else:
    print("🎉 Success! No gaps found. Your map is complete.")


# In[23]:


# 1. Tell GeoPandas which column has the shapes
skeleton = skeleton.set_geometry('GEOMETRY')

# 2. (Optional but better) Rename it back to lowercase for standard use
skeleton = skeleton.rename(columns={'GEOMETRY': 'geometry'})
skeleton = skeleton.set_geometry('geometry')

# 3. Ensure the CRS is set (GADM is always EPSG:4326)
if skeleton.crs is None:
    skeleton.crs = "EPSG:4326"

print("✅ Geometry column fixed and CRS defined.")


# In[24]:


import geopandas as gpd
import pandas as pd

# Load all three levels from GADM
print("📥 Loading GADM levels for multi-tier fallback...")
adm0 = gpd.read_file(gpkg_path, layer='ADM_0')
adm1 = gpd.read_file(gpkg_path, layer='ADM_1')
adm2 = gpd.read_file(gpkg_path, layer='ADM_2')

africa_list = ["DZA","AGO","BEN","BWA","BFA","BDI","CMR","CPV","CAF","TCD","COM","COG","COD","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","CIV","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE","ESH"]

final_layers = []

for iso in africa_list:
    # 1. Try to get ADM2 (Districts)
    pick = adm2[adm2['GID_0'] == iso].copy()
    level = 2
    
    # 2. If no ADM2, try ADM1 (Provinces)
    if pick.empty:
        pick = adm1[adm1['GID_0'] == iso].copy()
        level = 1
        
    # 3. If still no ADM1, try ADM0 (Country Border)
    if pick.empty:
        pick = adm0[adm0['GID_0'] == iso].copy()
        level = 0
        
    if not pick.empty:
        # Standardize column names for the merge
        name_col = f'NAME_{level}' if level > 0 else 'COUNTRY'
        pick['ADM_NAME'] = pick[name_col]
        pick['ADM_LEVEL'] = level
        final_layers.append(pick[['GID_0', 'ADM_NAME', 'ADM_LEVEL', 'geometry']])

# Create the truly complete skeleton
skeleton = gpd.GeoDataFrame(pd.concat(final_layers), crs=adm0.crs)
skeleton.columns = skeleton.columns.str.upper() # Keep columns consistent
print(f"✅ Skeleton rebuilt. Countries included: {skeleton['GID_0'].nunique()}")


# In[25]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 1. FIX THE GEOMETRY LINK (Solve the AttributeError)
# Tell the computer: "The column GEOMETRY is where the shapes are"
skeleton = skeleton.set_geometry('GEOMETRY')

# Rename it back to lowercase 'geometry' so it works perfectly with all libraries
skeleton = skeleton.rename(columns={'GEOMETRY': 'geometry'})
skeleton = skeleton.set_geometry('geometry')

# Ensure CRS is correct
if skeleton.crs is None:
    skeleton.crs = "EPSG:4326"

# 2. UPDATE STATUS FOR THE NEW GEOMETRY
# (This ensures Lesotho and Eswatini get checked against your rainfall data)
data_present_list = df_current['adm_name'].unique()
skeleton['STATUS'] = 'Missing (GAP)'
skeleton.loc[skeleton['ADM_NAME'].str.upper().isin(data_present_list), 'STATUS'] = 'Data Present'

# 3. COUNT THE GAPS
gap_count = len(skeleton[skeleton['STATUS'] == 'Missing (GAP)'])
print(f"✅ Total African Entities: {len(skeleton)}")
print(f"✅ Gaps remaining: {gap_count}")

# 4. FINAL ROBUST PLOT
fig, ax = plt.subplots(1, 1, figsize=(15, 15))

# Plot the whole skeleton (base layer)
skeleton.plot(
    column='STATUS', 
    ax=ax, 
    cmap='RdYlGn_r', # Red for Missing, Green for Data Present
    edgecolor='black', 
    linewidth=0.1,
    aspect=1
)

# Professional Legend for the Dissertation
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Rainfall Data Present (2000-2025)', 
           markerfacecolor='#27ae60', markersize=12),
    Line2D([0], [0], marker='o', color='w', label=f'White Holes ({gap_count} Gaps Identified)', 
           markerfacecolor='#e74c3c', markersize=12)
]
ax.legend(handles=legend_elements, loc='lower left', frameon=True, fontsize=12, title="Audit Results")

plt.title("Spatio-Temporal Continuity Audit: Africa Drought Study", fontsize=18, fontweight='bold')
plt.suptitle(f"Verified Administrative Skeleton | Precision Level: ADM1-ADM2 Hybrid", y=0.88, fontsize=12)
ax.axis('off')

plt.tight_layout()
plt.savefig("Africa_Final_Verified_Audit.png", dpi=300)
plt.show()

# 5. Identify the names of the gaps (if any)
if gap_count > 0:
    print("\n📍 REMAINING GAPS FOUND:")
    print(skeleton[skeleton['STATUS'] == 'Missing (GAP)'][['GID_0', 'ADM_NAME']])


# In[26]:


# Isolate the 77 gaps into a dedicated GeoDataFrame
missing_77_gdf = skeleton[skeleton['STATUS'] == 'Missing (GAP)'].copy()

# Ensure we have the necessary columns for GEE identification
# We use GID_0 (ISO) and ADM_NAME (District Name)
missing_77_gdf = missing_77_gdf[['GID_0', 'ADM_NAME', 'ADM_LEVEL', 'geometry']]

print(f"✅ Isolated {len(missing_77_gdf)} districts for historical recovery.")


# In[1]:


import ee
import geemap
import pandas as pd

# 1. Initialize
PROJECT_ID = "vernal-parser-412016"
try:
    ee.Initialize(project=PROJECT_ID)
    print("✅ Earth Engine initialized successfully!")
except Exception:
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)


# In[ ]:




