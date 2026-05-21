#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path


# In[2]:


# Load CSV
csv_path = r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk\Drought\Output\final-GLAD-50-2019.csv"
df_results = pd.read_csv(csv_path)

# Remove duplicate
df_results = df_results.drop_duplicates(subset=['ADM_NAME', 'COUNTRY'])


# In[3]:


# Load Africa shp
original_shp_path = r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk\Drought\Output\africa_admx_shp\africa_admx.shp"
gdf_districts = gpd.read_file(original_shp_path)

print(f"Original shapefile has {len(gdf_districts)} rows.")

# Dissolve
print("Dissolving multi-part polygons to force a 1-to-1 match...")
gdf_districts = gdf_districts.dissolve(by=['COUNTRY', 'ADM_NAME']).reset_index()

print(f"Cleaned shapefile has {len(gdf_districts)} unique districts.")
print(f"Filtered CSV has {len(df_results)} unique districts.")


# In[4]:


# Perform the Merge (The 'Join')
final_gdf = gdf_districts.merge(
    df_results,
    on=['ADM_NAME', 'COUNTRY'],
    how='inner'
)


# In[5]:


import geopandas as gpd

# Extract ONLY Africa continent boundary (ADM_0 = country level)
gdf = gpd.read_file(r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk\Drought\data\gadm_410-levels\World_Continents_-8107292174417139505\World_Continents.shp")


# In[6]:


print("Your exact column names:")
print(gdf.columns.tolist())
print("\nFirst few rows:")
print(gdf[['CONTINENT', 'SQKM', 'SQMI']].head())


# In[7]:


# Cleanup & name
rename_dict = {
    'cropland_km2': 'crop_km2',
    'total_area_km2': 'total_km2',
    'cropland_pct': 'crop_pct',
    'total_country_crop': 'cntry_crop',
    'cum_pct': 'cum_pct'
}
final_gdf = final_gdf.rename(columns=rename_dict)

# Keep only the columns we want
desired_cols =['ISO3', 'ISO3_x', 'ISO3_y', 'COUNTRY', 'ADM_NAME', 'crop_km2', 'total_km2', 'crop_pct', 'cum_pct', 'geometry']
columns_to_keep =[col for col in desired_cols if col in final_gdf.columns]
final_gdf = final_gdf[columns_to_keep]

# Clean up ISO3 naming if it got altered during merge
if 'ISO3_x' in final_gdf.columns:
    final_gdf = final_gdf.rename(columns={'ISO3_x': 'ISO3'})
elif 'ISO3_y' in final_gdf.columns:
    final_gdf = final_gdf.rename(columns={'ISO3_y': 'ISO3'})


# In[8]:


# Calculate the master scale based on all 3,500+ districts in Africa
# This value stays the same for every map you generate from now on
GLOBAL_VMAX = final_gdf['crop_pct'].quantile(0.95)
GLOBAL_CMAP = 'YlGn' # Yellow-to-Green (Clean & Professional)


# In[12]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# --------------------------------------------------
# 1. LOAD AFRICA BACKGROUND BOUNDARY
# --------------------------------------------------
# Option A: use your own Africa boundary file if you have one
# africa_boundary = gpd.read_file(r"C:\path\to\africa_boundary.shp")

# Option B: use Natural Earth built into GeoPandas-compatible datasets
world = gpd.read_file("https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip")
africa_boundary = world[world["CONTINENT"] == "Africa"].copy()

# --------------------------------------------------
# 2. MAKE SURE final_gdf EXISTS
# --------------------------------------------------
# Replace this with your actual file if final_gdf is not already in memory
# final_gdf = gpd.read_file(r"C:\path\to\your_final_layer.shp")

print("africa_boundary columns:", africa_boundary.columns.tolist())
print("africa_boundary shape:", africa_boundary.shape)

# --------------------------------------------------
# 3. CRS ALIGNMENT
# --------------------------------------------------
if africa_boundary.crs != final_gdf.crs:
    africa_boundary = africa_boundary.to_crs(final_gdf.crs)

# --------------------------------------------------
# 4. DEFINE MAP SETTINGS
# --------------------------------------------------
GLOBAL_CMAP = "YlGn"
GLOBAL_VMAX = final_gdf["crop_pct"].max()

# --------------------------------------------------
# 5. CREATE FIGURE
# --------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(14, 12))
fig.patch.set_facecolor("white")

# --------------------------------------------------
# 6. LAYER 1: AFRICA BACKGROUND
# --------------------------------------------------
africa_boundary.plot(
    ax=ax,
    color="white",
    edgecolor="#cccccc",
    linewidth=0.8,
    zorder=1
)

# --------------------------------------------------
# 7. LAYER 2: DISTRICT POLYGONS
# --------------------------------------------------
final_gdf.plot(
    column="crop_pct",
    ax=ax,
    zorder=2,
    cmap=GLOBAL_CMAP,
    vmin=0,
    vmax=GLOBAL_VMAX,
    linewidth=0.1,
    edgecolor="#cccccc",
    legend=False
)

# --------------------------------------------------
# 8. OPTIONAL: ADD ETHIOPIA OUTLINE IF YOU HAVE IT
# --------------------------------------------------
# If you have ethiopia boundary already loaded:
# ethiopia_boundary.boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=3)

# --------------------------------------------------
# 9. COLORBAR
# --------------------------------------------------
norm = mpl.colors.Normalize(vmin=0, vmax=GLOBAL_VMAX)
sm = mpl.cm.ScalarMappable(cmap=GLOBAL_CMAP, norm=norm)
sm._A = []

cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Cropland Percentage (%)", fontsize=11)

# --------------------------------------------------
# 10. CLEAN MAP
# --------------------------------------------------
ax.set_title("Cropland Percentage Across Districts", fontsize=14)
ax.set_axis_off()

plt.tight_layout()
# 6. SAVE CROPPED RESULT
plt.savefig('Africa_Cropland_districts.png', bbox_inches='tight', dpi=300, facecolor='white')
plt.show()


# In[10]:


import geopandas as gpd
import matplotlib.pyplot as plt

# 1. LOAD ETHIOPIA BOUNDARY (clipper)
gadm_file = r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk\Drought\data\gadm_410-levels\gadm_410-levels.gpkg"
ethiopia_boundary = gpd.read_file(gadm_file, layer='ADM_0')
ethiopia_boundary = ethiopia_boundary[ethiopia_boundary['COUNTRY'] == 'Ethiopia'].copy()

# 2. **REPLACE THIS** with your ACTUAL cropland file path:
cropland_file = r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk\Drought\Output\ethiopia\output\ethiopia_filtered_data.shp"
cropland_gdf = gpd.read_file(cropland_file)

print(f"Original cropland: {len(cropland_gdf)} polygons")
print("Columns:", cropland_gdf.columns.tolist())

# 3. ALIGN CRS
if cropland_gdf.crs != ethiopia_boundary.crs:
    cropland_gdf = cropland_gdf.to_crs(ethiopia_boundary.crs)

# 4. **CLIP** - Keep ONLY Ethiopia cropland
ethiopia_cropland = gpd.clip(cropland_gdf, ethiopia_boundary)

print(f"Ethiopia-only cropland: {len(ethiopia_cropland)} polygons")
print(f"Removed: {len(cropland_gdf) - len(ethiopia_cropland)} polygons")

# 5. PLOT BEFORE/AFTER
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# BEFORE (full Africa)
cropland_gdf.plot(ax=ax1, column='crop_pct' if 'crop_pct' in cropland_gdf.columns else cropland_gdf.columns[0], 
                  cmap='YlOrBr', edgecolor='none', alpha=0.7)
ax1.set_title("Before Clipping")
ax1.axis('off')

# AFTER (Ethiopia only)
ethiopia_boundary.boundary.plot(ax=ax2, color='black', linewidth=2)
ethiopia_cropland.plot(ax=ax2, column='crop_pct' if 'crop_pct' in cropland_gdf.columns else cropland_gdf.columns[0], 
                       cmap='YlOrBr', edgecolor='white', linewidth=0.3, alpha=0.9)
ax2.set_title("After Clipping (Ethiopia Only)")
ax2.axis('off')

plt.tight_layout()
plt.savefig(r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk\Drought\Output\Cropland_Clipping_Before_After.png", dpi=300)
plt.show()

# 6. SAVE CROPPED RESULT
ethiopia_cropland.to_file(r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk\Drought\Output\ethiopia_cropland_clipped.shp")
print("✅ Saved ethiopia_cropland_clipped.shp")


# In[18]:


import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl

# 1. DEFINE OUTPUT PATH
output_base = Path(r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk\Drought\Output")

# 2. DEFINE MAP SETTINGS
GLOBAL_CMAP = 'YlOrBr'
GLOBAL_VMAX = final_gdf['crop_pct'].max()

# 3. FILTER ZAMBIA
zambia_gdf = final_gdf[final_gdf['COUNTRY'] == 'Zambia'].copy()
print(f"✅ Found {len(zambia_gdf)} Zambian districts")

# 4. CREATE MAP (NO DUPLICATE SCALEBARS)
fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
fig.patch.set_facecolor('white')

# ✅ FIXED: legend=False + manual colorbar only
zambia_gdf.plot(
    column='crop_pct', 
    ax=ax,
    cmap=GLOBAL_CMAP, 
    vmin=0, 
    vmax=GLOBAL_VMAX,
    edgecolor='white', 
    linewidth=0.5,
    alpha=0.9,
    legend=False,  # ← THIS FIXES DUPLICATE SCALEBAR
    missing_kwds=dict(color='lightgray')
)

# 5. SINGLE CLEAN COLORBAR
norm = mpl.colors.Normalize(vmin=0, vmax=GLOBAL_VMAX)
sm = mpl.cm.ScalarMappable(cmap=GLOBAL_CMAP, norm=norm)
sm.set_array([])  # ← Better than _A = []
cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, shrink=0.8)
cbar.set_label("Cropland Percentage (%)", fontsize=12, weight='bold')
cbar.ax.tick_params(labelsize=10)

# 6. MAP ELEMENTS
ax.set_title("Zambia Agricultural Districts", fontsize=18, fontweight='bold', color='#1a5c37')
ax.set_axis_off()

# 7. CAPTION
plt.figtext(0.5, 0.02, "Figure 2: Cropland Distribution Across Zambian Districts", 
            ha='center', fontsize=11, style='italic')

# 8. SAVE
plt.tight_layout()
plt.savefig(output_base / '2_Zambia_Scale.png', bbox_inches='tight', dpi=300, facecolor='white')
plt.show()

print(f"✅ Zambia map saved: {output_base / '2_Zambia_Scale.png'}")
print("✅ SINGLE scalebar only!")


# In[19]:


southern_districts = [
    "CHIKANKATA", "CHIRUNDU", "CHOMA", "GWEMBE", "ITEZHI-TEZHI",
    "KALOMO", "KAZUNGULA", "LIVINGSTONE", "MAZABUKA", "MONZE",
    "NAMWALA", "PEMBA", "SIAVONGA", "SINAZONGWE", "ZIMBA"
]

southern_prov_gdf = final_gdf[
    (final_gdf['COUNTRY'] == 'Zambia') & 
    (final_gdf['ADM_NAME'].str.upper().isin(southern_districts))
].copy()

fig, ax = plt.subplots(figsize=(14, 12), dpi=300)

# Plot with the GLOBAL scale
southern_prov_gdf.plot(
    ax=ax, column='crop_pct', 
    cmap=GLOBAL_CMAP, vmin=0, vmax=GLOBAL_VMAX, # <--- FIXED SCALE (Matches Africa)
    edgecolor='grey', linewidth=1.0
)

# Add Labels
for idx, row in southern_prov_gdf.iterrows():
    coords = row.geometry.representative_point().coords[0]
    ax.annotate(text=row['ADM_NAME'], xy=coords, ha='center', 
                fontsize=9, fontweight='bold', color='black',
                bbox=dict(facecolor='white', alpha=0.4, edgecolor='none'))

ax.set_title("🇿🇲 Southern Province: District Detail", fontsize=20, fontweight='bold', color='#1a5c37')
ax.set_axis_off()

# Save
plt.savefig(output_base / '3_Southern_Province_Scale.png', bbox_inches='tight', dpi=300)
plt.show()

