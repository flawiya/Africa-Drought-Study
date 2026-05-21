#!/usr/bin/env python
# coding: utf-8

# # Drought

# In[1]:


#Install the API
get_ipython().system('pip install ee')
get_ipython().system('pip install altair vega_datasets')


# In[2]:


# warning = FALSE
import geopandas as gpd
import pandas as pd
import pycountry
import os
import requests #request for API
import zipfile #load zip files
#import geemap #import gee map into jupyter
from rapidfuzz import process, utils #same words written differently can be connected
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#load csv for drought using pandas read csv
drought_df=pd.read_csv('C:/Users/FlawiyaShirishMore/OneDrive - Africa Specialty Risks Ltd/ASR-Parametric_Research_Study/africa_risk/data/drought.csv')


# In[4]:


drought_df.shape #check total number of columns, rows


# In[5]:


drought_df.head() #top 5 rows data


# In[6]:


drought_df.describe() #zonal stats of the data


# In[7]:


drought_df.isna().sum() #count of missing values in each column


# In[8]:


drought_df.columns[drought_df.isna().any()] #check which column contains missing data


# In[9]:


missing_cols =['External IDs', 'Origin', 'Associated Types', 'Latitude',
       'Longitude', 'River Basin','CPI'] #define missing_cols variable with the column names
drought_df=drought_df.drop(columns=missing_cols) #drop missing_cols


# In[10]:


drought_df.columns #verify updated dataframe


# In[11]:


tmp = drought_df[["No. Affected", "Total Affected"]].copy()

# Convert to numeric
tmp["No. Affected"] = pd.to_numeric(tmp["No. Affected"], errors="coerce")
tmp["Total Affected"] = pd.to_numeric(tmp["Total Affected"], errors="coerce")

# Drop rows where either column is NaN
tmp = tmp.dropna(subset=["No. Affected", "Total Affected"])

# Keep only positive values for log scale (optional but recommended)
tmp = tmp[(tmp["No. Affected"] > 0) & (tmp["Total Affected"] > 0)]


# In[12]:


from drought_plots import plot_year_month_heatmap

# If your month is text (e.g., "Jan"), convert to numbers first (optional):
# drought_df["Start Month"] = pd.to_datetime(drought_df["Start Month"], format="%b").dt.month
#Seasonal clustering by year
fig1 = plot_year_month_heatmap(
    drought_df,
    year_col="Start Year",
    month_col="Start Month",
    title="Drought Events by Year and Month"
)
plt.savefig('Drought Events by Year and Month.jpg', dpi = 300)
plt.show()


# *repeated cluster on drought during Jan, Feb, March:
# East africa experiences long rainin mar-may,  so drought begins before when rain season fails. In the south african areas the mainrainy season is during Nov-mar, if the rain fail early the drought occurs inthe area. As we can see, the africa is a big continent, with varied climatic, and geographies, the early year drought onsets are expected in multi-regions.
# 
# *Secondary peak of drought in during may-june:
# It is the end of rainy season in east africa(MAM), beginning of **Sahel pre-monsson dry period**, where stress trigger dorught declaration.
# 
# * in 2001 very high jan, mar onset count/spike. 2015-2017 cluster of drought starts across jan-jun, these years corresponds to major **el nino** conditions that has historically caused severe drought in southern africa, rainfall deficits in horn of africa.

# In[13]:


#Are drought onsets getting more/less frequent?
yearly = drought_df.groupby("Start Year").size()
ax = yearly.plot(marker="o", color="#d35400", title="Drought Starts per Year (Africa)", figsize=(9,4))
ax.set_xlabel("Year"); ax.set_ylabel("Starts"); ax.grid(alpha=0.3)
plt.savefig('Drought Starts per Year (Africa)', dpi = 300)


# In[14]:


#Which months most often see drought onsets?
monthly=drought_df.groupby('Start Month').size().reindex(range(1,13), fill_value=0) #range start from 0 ends at 12
monthly.plot(kind='bar', color='skyblue', figsize=(8,4), title='Seasonality of Drought Onsets')
plt.savefig('Seasonality of Drought Onsets', dpi = 300)


# In[15]:


#Whether consequences are rising even if frequency is not.
#Trend in impacts per year (e.g., Total Affected / Total Damage)
impact_year = (drought_df.groupby("Start Year")[["Total Affected","Total Damage (000 US$)"]]
               .sum().fillna(0))
ax = impact_year.plot(secondary_y="Total Damage (000 US$)", figsize=(10,4),
                      title="Yearly Impacts: Affected vs Damage")
plt.savefig('Yearly Impacts: Affected vs Damage', dpi = 300)


# In[16]:


# -----------------------------------------------------------
# PURPOSE:
#   Build an Africa boundary layer that is:
#     - ADM-2 for all countries (from GADM) EXCEPT Libya
#     - ADM-1 for Libya (since ADM-2 is not standardized/available)
#
# OUTPUTS:
#   - GPKG:  <your_gpkg_stem>_AFRICA_ADMx.gpkg  (layer: ADMx_AFRICA)
#   - ZIP Shapefile: africa_admx_shapefile.zip  (for Earth Engine UI upload)
#
# NOTES:
#   - The script harmonizes fields to:
#       ISO3, COUNTRY, ADM_NAME, ADM_LEVEL, geometry
#   - Shapefile field-length limit is 10 chars, so names are short & safe.
# -----------------------------------------------------------

# 0) Imports
import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import os
# ➊ NEW: you use pd.concat below
import pandas as pd

# 1) Point to your GeoPackage
gpkg_path = Path(r"C:/Users/FlawiyaShirishMore/OneDrive - Africa Specialty Risks Ltd/ASR-Parametric_Research_Study/africa_risk/data/gadm_410-levels/gadm_410-levels.gpkg")  # <-- change if needed

# ---- A. Inspect the GPKG ----
print("Layers found in GPKG:")
layers = fiona.listlayers(gpkg_path)
for i, lyr in enumerate(layers, 1):
    print(f"{i:2d}. {lyr}")

# Helper: pick candidate layers by name pattern
adm2_layer = None
adm1_layer = None
for lyr in layers:
    low = lyr.lower()
    if (adm2_layer is None) and (low.endswith("_2") or "adm2" in low or "admin2" in low):
        adm2_layer = lyr
    if (adm1_layer is None) and (low.endswith("_1") or "adm1" in low or "admin1" in low):
        adm1_layer = lyr

if adm2_layer is None or adm1_layer is None:
    raise ValueError(f"Could not auto-detect ADM_1/ADM_2 layers. Found: {layers}. Set adm1_layer/adm2_layer manually.")

print(f"\nSelected Admin-2 layer: {adm2_layer}")
print(f"Selected Admin-1 layer: {adm1_layer}")

# ---- B. Load the layers ----
adm2 = gpd.read_file(gpkg_path, layer=adm2_layer)
adm1 = gpd.read_file(gpkg_path, layer=adm1_layer)

print("\nAdmin-2 columns preview:", list(adm2.columns)[:20])
print("CRS (ADM-2):", adm2.crs)
print("Admin-1 columns preview:", list(adm1.columns)[:20])
print("CRS (ADM-1):", adm1.crs)

# Basic column pick helper
def pick_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

# Identify key columns (GADM typical)
cols2 = set(adm2.columns)
col2_iso  = pick_col(cols2, ["GID_0", "ISO3", "ISO", "COUNTRY_ISO"])
col2_ctry = pick_col(cols2, ["NAME_0", "COUNTRY", "CNTRY_NAME", "NAME_ENGL"])
col2_nm2  = pick_col(cols2, ["NAME_2", "ADM2_EN", "ADM2_NAME", "shapeName"])

cols1 = set(adm1.columns)
col1_iso  = pick_col(cols1, ["GID_0", "ISO3", "ISO", "COUNTRY_ISO"])
col1_ctry = pick_col(cols1, ["NAME_0", "COUNTRY", "CNTRY_NAME", "NAME_ENGL"])
col1_nm1  = pick_col(cols1, ["NAME_1", "ADM1_EN", "ADM1_NAME", "shapeName"])

print(f"\n[ADM-2] ISO: {col2_iso}, Country: {col2_ctry}, ADM2 name: {col2_nm2}")
print(f"[ADM-1] ISO: {col1_iso}, Country: {col1_ctry}, ADM1 name: {col1_nm1}")

if col2_iso is None and "GID_0" in cols2:
    col2_iso = "GID_0"
if col1_iso is None and "GID_0" in cols1:
    col1_iso = "GID_0"
if col2_iso is None or col1_iso is None:
    raise ValueError("Could not locate country ISO column (ADM-1/ADM-2). Adjust mapping above.")

# ---- C. Filter to AFRICA only ----
africa_iso3 = {
    "DZA","AGO","BEN","BWA","BFA","BDI","CMR","CPV","CAF","TCD","COM","COG","COD","CIV","DJI","EGY","GNQ",
    "ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS",
    "MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN",
    "UGA","ZMB","ZWE","ESH"
}

# Normalize ISO fields and create 'ISO3' in both frames
adm2["ISO3"] = adm2[col2_iso].astype(str).str.upper().str[:3]
adm1["ISO3"] = adm1[col1_iso].astype(str).str.upper().str[:3]

# Africa filters
adm2_africa = adm2[adm2["ISO3"].isin(africa_iso3)].copy()
adm1_africa = adm1[adm1["ISO3"].isin(africa_iso3)].copy()

# ---- D. Build ADMx (ADM-2 most countries, ADM-1 for Libya) ----
# 1) Exclude Libya from ADM-2 set (to ensure Libya will come only from ADM-1)
adm2_no_lby = adm2_africa[adm2_africa["ISO3"] != "LBY"].copy()

# 2) Keep only Libya from ADM-1
lby_adm1 = adm1_africa[adm1_africa["ISO3"] == "LBY"].copy()

# 3) Harmonize to a compact schema: ISO3, COUNTRY, ADM_NAME, ADM_LEVEL, geometry
def to_adm2_schema(df, iso_col, ctry_col, nm2_col):
    out = gpd.GeoDataFrame({
        "ISO3": df[iso_col].astype(str).str.upper().str[:3],
        "COUNTRY": df[ctry_col].astype(str) if ctry_col else "",
        "ADM_NAME": df[nm2_col].astype(str) if nm2_col else "",
        "ADM_LEVEL": "ADM2"
    }, geometry=df.geometry, crs=df.crs)
    return out

def to_adm1_schema(df, iso_col, ctry_col, nm1_col):
    out = gpd.GeoDataFrame({
        "ISO3": df[iso_col].astype(str).str.upper().str[:3],
        "COUNTRY": df[ctry_col].astype(str) if ctry_col else "",
        "ADM_NAME": df[nm1_col].astype(str) if nm1_col else "",
        "ADM_LEVEL": "ADM1"
    }, geometry=df.geometry, crs=df.crs)
    return out

adm2_std = to_adm2_schema(adm2_no_lby, col2_iso, col2_ctry, col2_nm2)
lby_std  = to_adm1_schema(lby_adm1,   col1_iso, col1_ctry, col1_nm1)

# 4) Merge (concatenate) and re-check CRS
admX_africa = gpd.GeoDataFrame(pd.concat([adm2_std, lby_std], ignore_index=True), crs=adm2_std.crs)

# ➋ NEW: 4b) Ensure Western Sahara (ESH) is present. If missing:
#   (a) Try to source from your ADM-1 layer (GADM);
#   (b) Fallback to Natural Earth (ships with GeoPandas).
if not (admX_africa["ISO3"] == "ESH").any():
    print("Western Sahara (ESH) not found in GADM layers. Attempting to add it...")
    try:
        # (a) From ADM-1 layer if present
        esh_from_adm1 = adm1[adm1[col1_iso].astype(str).str.upper().str[:3] == "ESH"].copy()
        if len(esh_from_adm1):
            esh_std = to_adm1_schema(esh_from_adm1, col1_iso, col1_ctry, col1_nm1)
            # Dissolve to a single placeholder feature
            esh_diss = esh_std.dissolve(by="ISO3", as_index=False, aggfunc="first")
            esh_diss["COUNTRY"] = "Western Sahara"
            esh_diss["ADM_NAME"] = "Western Sahara"
            esh_diss["ADM_LEVEL"] = "ADM1"
            admX_africa = gpd.GeoDataFrame(
                pd.concat([admX_africa, esh_diss], ignore_index=True),
                crs=admX_africa.crs
            )
            print("Added Western Sahara from GADM ADM-1.")
        else:
            # (b) Fallback: Natural Earth lowres (no internet needed)
            #    (comes with GeoPandas; columns include 'iso_a3' and 'name')
            ne_world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
            esh = ne_world[ne_world["iso_a3"] == "ESH"]
            if len(esh) == 0:
                esh = ne_world[ne_world["name"] == "Western Sahara"]
            if len(esh):
                esh = esh.to_crs(admX_africa.crs)
                esh_gdf = gpd.GeoDataFrame({
                    "ISO3": ["ESH"],
                    "COUNTRY": ["Western Sahara"],
                    "ADM_NAME": ["Western Sahara"],
                    "ADM_LEVEL": ["ADM1"],
                    "geometry": esh.geometry.values
                }, geometry="geometry", crs=admX_africa.crs)
                admX_africa = gpd.GeoDataFrame(
                    pd.concat([admX_africa, esh_gdf], ignore_index=True),
                    crs=admX_africa.crs
                )
                print("Added Western Sahara from Natural Earth (lowres).")
            else:
                print("WARNING: Could not locate Western Sahara in Natural Earth fallback.")
    except Exception as e:
        print(f"WARNING: Failed to add Western Sahara (ESH): {e}")

print(f"\nADM-2 (no Libya): {len(adm2_std):,}")
print(f"Libya (ADM-1):    {len(lby_std):,}")
print(f"TOTAL ADMx:       {len(admX_africa):,}")
print("ADM_LEVEL counts:\n", admX_africa["ADM_LEVEL"].value_counts(dropna=False))

# ---- E. Quick plot (Africa) ----
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
admX_africa.boundary.plot(ax=ax, linewidth=0.15, color="black")
ax.set_title("Africa – Admin Level X (ADM-2 except Libya=ADM-1 + Western Sahara=ADM-1)", fontsize=15)
ax.axis("off")
plt.tight_layout()
plt.savefig("Africa_ADMx_outline.png", dpi=300)
plt.show()

# ---- F. Save the ADMx Africa subset (GPKG) ----
out_gpkg = gpkg_path.with_name(gpkg_path.stem + "_AFRICA_ADMx.gpkg")
admX_africa.to_file(out_gpkg, layer="ADMx_AFRICA", driver="GPKG")
print(f"\nSaved ADMx Africa to: {out_gpkg.resolve()} (layer=ADMx_AFRICA)")

# ---- G. Optional: Save as a zipped Shapefile for Earth Engine UI upload ----
shp_dir = Path("africa_admx_shp")
shp_dir.mkdir(exist_ok=True)

# Keep only the essential columns to avoid Shapefile truncation/dup issues
admX_for_shp = admX_africa[["ISO3", "COUNTRY", "ADM_NAME", "ADM_LEVEL", "geometry"]].copy()
shp_path = shp_dir / "africa_admx.shp"
# Write Shapefile (creates .shp/.shx/.dbf/.prj)
admX_for_shp.to_file(shp_path)

zip_path = Path("africa_admx_shapefile.zip")
if zip_path.exists():
    zip_path.unlink()
shutil.make_archive(base_name=zip_path.stem, format="zip", root_dir=shp_dir)
print(f"Zipped shapefile ready for EE UI upload: {zip_path.resolve()}")


# In[17]:


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


# In[18]:


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


# In[19]:


# 1. Tell GeoPandas which column has the shapes
skeleton = skeleton.set_geometry('GEOMETRY')

# 2. (Optional but better) Rename it back to lowercase for standard use
skeleton = skeleton.rename(columns={'GEOMETRY': 'geometry'})
skeleton = skeleton.set_geometry('geometry')

# 3. Ensure the CRS is set (GADM is always EPSG:4326)
if skeleton.crs is None:
    skeleton.crs = "EPSG:4326"

print("✅ Geometry column fixed and CRS defined.")


# In[20]:


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


# In[21]:


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


# In[22]:


# Isolate the 77 gaps into a dedicated GeoDataFrame
missing_77_gdf = skeleton[skeleton['STATUS'] == 'Missing (GAP)'].copy()

# Ensure we have the necessary columns for GEE identification
# We use GID_0 (ISO) and ADM_NAME (District Name)
missing_77_gdf = missing_77_gdf[['GID_0', 'ADM_NAME', 'ADM_LEVEL', 'geometry']]

print(f"✅ Isolated {len(missing_77_gdf)} districts for historical recovery.")


# In[23]:


import os
import zipfile
import shutil

# 1. Clean up columns for Shapefile compatibility (Max 10 characters)
# 'geometry' remains as is
missing_77_export = missing_77_gdf.rename(columns={
    'ADM_NAME': 'name',
    'ADM_LEVEL': 'level',
    'GID_0': 'iso3'
})

# 2. Create a temporary folder for the files
temp_folder = "gee_upload"
if os.path.exists(temp_folder):
    shutil.rmtree(temp_folder)
os.makedirs(temp_folder)

# 3. Save the Shapefile components
shp_base_name = "Africa_77_Gaps"
missing_77_export.to_file(os.path.join(temp_folder, f"{shp_base_name}.shp"))

# 4. Create the ZIP file
zip_name = f"{shp_base_name}.zip"
with zipfile.ZipFile(zip_name, 'w') as zipf:
    for file in os.listdir(temp_folder):
        zipf.write(os.path.join(temp_folder, file), file)

print(f"✅ Success! Created '{zip_name}'.")
print("👉 ACTION: Go to GEE Assets -> New -> Shapefile and upload this ZIP file.")


# In[ ]:


# If running in a new environment, install the packages:
# (Uncomment if needed)

get_ipython().run_line_magic('pip', 'install earthengine-api geemap --quiet')


# In[ ]:


pip install ee


# In[ ]:


pip install fcntl


# In[ ]:


import ee, geemap

try:
    ee.Initialize(project='vernal-parser-412016')   # your GCP project ID
    print("✅ EE initialized.")
except Exception:
    print("🔑 Authenticating...")
    ee.Authenticate()                               # follow the browser link, paste token
    ee.Initialize(project='vernal-parser-412016')
    print("✅ EE initialized after auth.")


# In[ ]:


import ee

# 1. Initialize
PROJECT_ID = "vernal-parser-412016"
ee.Initialize(project=PROJECT_ID)

# 2. TARGET THE 77 GAPS: Use the specific asset you just verified
# Replace 'Africa_77_Gaps' with the exact name in your GEE Assets tab
asset_id = f"projects/{PROJECT_ID}/assets/Africa_77_Gaps"
gap_districts = ee.FeatureCollection(asset_id)

# 3. Load Datasets (Rainfall + Temperature for Water Balance)
chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select('precipitation')
era5_temp = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").select('temperature_2m')

def export_year_stats(year):
    """Calculates monthly Rainfall and Temperature for the 77 gaps."""
    
    months = ee.List.sequence(1, 12)
    
    def calculate_stats(m):
        # A. Monthly Rainfall Sum
        monthly_rain = chirps.filter(ee.Filter.calendarRange(year, year, 'year')) \
                              .filter(ee.Filter.calendarRange(m, m, 'month')) \
                              .sum()
        
        # B. Monthly Temperature Mean (The "Fabulous" Factor)
        monthly_temp = era5_temp.filter(ee.Filter.calendarRange(year, year, 'year')) \
                                .filter(ee.Filter.calendarRange(m, m, 'month')) \
                                .mean()
        
        # Combine bands into one image
        combined_img = monthly_rain.addBands(monthly_temp)
        
        # 2. Run Zonal Statistics for the 77 districts
        stats = combined_img.reduceRegions(
            collection=gap_districts,
            reducer=ee.Reducer.mean(),
            scale=5566 
        )
        
        # 3. Add metadata
        return stats.map(lambda f: f.set({'month': m, 'year': year}))

    # Flatten the 12 months
    yearly_stats = ee.FeatureCollection(months.map(calculate_stats)).flatten()
    
    # 4. EXPORT TO DRIVE
    # Using 'selectors' ensures the CSV headers match your Master CSV for an easy merge
    task = ee.batch.Export.table.toDrive(
        collection=yearly_stats,
        description=f"Gap_Fill_Multivariate_{year}",
        folder="ASR_Dissertation_Final_Gaps",
        fileNamePrefix=f"gap_results_{year}",
        fileFormat='CSV',
        selectors=['iso3', 'name', 'year', 'month', 'precipitation', 'temperature_2m']
    )
    task.start()
    print(f"🚀 Started GEE Task: Multivariate Gap-Fill for {year}")

# --- EXECUTION ---
# Reconstruct the full 1981 - 2025 timeline for the missing gaps
years_to_process = list(range(1981, 2026))

for yr in years_to_process:
    export_year_stats(yr)

print("\n📊 ALL RECONSTRUCTION TASKS SUBMITTED!")
print("Check the 'Tasks' tab in GEE. Once they turn blue, download the CSVs.")


# In[ ]:





# In[ ]:




import ee

# 1. Initialize
PROJECT_ID = "vernal-parser-412016"
ee.Initialize(project=PROJECT_ID)

# 2. Load your successfully uploaded asset
asset_id = f"projects/{PROJECT_ID}/assets/africa_admx_shapefile"
districts = ee.FeatureCollection(asset_id)

# 3. Load CHIRPS Rainfall Data (1981 - 2024)
chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select('precipitation')

def export_year_stats(year):
    """Calculates monthly rainfall sums for all 6,498 districts and exports to Drive."""
    
    # We create a list of months to process
    months = ee.List.sequence(1, 12)
    
    def calculate_stats(m):
        # 1. Get total rainfall for the specific month
        monthly_total = chirps.filter(ee.Filter.calendarRange(year, year, 'year')) \
                              .filter(ee.Filter.calendarRange(m, m, 'month')) \
                              .sum()
        
        # 2. Run Zonal Statistics: Mean rainfall per district polygon
        # We use .reduceRegions to calculate values for all 6,498 districts at once
        stats = monthly_total.reduceRegions(
            collection=districts,
            reducer=ee.Reducer.mean(),
            scale=5566 # CHIRPS resolution
        )
        
        # 3. Add month and year metadata to every row
        return stats.map(lambda f: f.set({'month': m, 'year': year}))

    # Flatten the 12 months into one single collection for the year
    yearly_stats = ee.FeatureCollection(months.map(calculate_stats)).flatten()
    
    # 4. EXPORT TO DRIVE
    # This runs on Google's cloud, not your laptop.
    task = ee.batch.Export.table.toDrive(
        collection=yearly_stats,
        description=f"Rainfall_Stats_Africa_{year}",
        folder="ASR_Dissertation_Data",
        fileNamePrefix=f"rainfall_{year}",
        fileFormat='CSV',
        selectors=['ISO3', 'ADM_NAME', 'ADM_LEVEL', 'year', 'month', 'mean'] # Keep file size small
    )
    task.start()
    print(f"✅ Started export task for year: {year}")

# --- EXECUTION ---
# For your dissertation, you need 2000 to 2024. 
years_to_process = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

for yr in years_to_process:
    export_year_stats(yr)

print("\n🚀 TASKS SUBMITTED!")
print("Go to https://code.earthengine.google.com/ and check the 'Tasks' tab to monitor progress.")# 1. Create a unique ID for each district
df_2022['district_id'] = df_2022['ISO3'] + "_" + df_2022['ADM_NAME']

# 2. Logic for Aim 2: Deviation from Mean
# Since we only pulled 2022, we compare each month to the annual average of that district
# (In your final dissertation, you will compare 2022 rainfall to the 1981-2023 mean)
df_2022['dist_mean_2022'] = df_2022.groupby('district_id')['mean'].transform('mean')
df_2022['rain_deficit'] = df_2022['dist_mean_2022'] - df_2022['mean']

# 3. Proxy Loss Ratio (Scale 0 to 1)
# High deficit = High Loss Ratio (Risk)
df_2022['loss_ratio'] = df_2022['rain_deficit'].clip(lower=0) 
df_2022['loss_ratio'] = df_2022['loss_ratio'] / df_2022['loss_ratio'].max()

print("\n--- Parametric Risk Assessment Preview ---")
print(df_2022.sort_values(by='loss_ratio', ascending=False).head(10))# merge csv to master csv
import pandas as pd
import glob
import os

# 1. Define the path to your folder containing the downloaded CSVs
input_path = r'C:/Users/FlawiyaShirishMore/OneDrive - Africa Specialty Risks Ltd/ASR-Parametric_Research_Study/africa_risk/data/ASR_Dissertation_Data' # <-- CHANGE THIS
output_file = 'master_rainfall_africa.csv'

# 2. Get a list of all CSV filenames in that folder
all_files = glob.glob(os.path.join(input_path, "*.csv"))

print(f"📂 Found {len(all_files)} files to merge.")

# 3. Use a list comprehension to read all CSVs into a list of DataFrames
# We only keep the columns we need to save memory
keep_cols = ['ISO3', 'ADM_NAME', 'ADM_LEVEL', 'year', 'month', 'mean']

df_list = [pd.read_csv(f)[keep_cols] for f in all_files]

# 4. Concatenate (Stack) them all together
master_df = pd.concat(df_list, ignore_index=True)

# 5. Data Cleaning
# Ensure types are correct
master_df['year'] = master_df['year'].astype(int)
master_df['month'] = master_df['month'].astype(int)
master_df['mean'] = pd.to_numeric(master_df['mean'], errors='coerce')

# 6. Save the Master Database
master_df.to_csv(output_file, index=False)

print(f"✅ Successfully created {output_file}")
print(f"📊 Total Rows in Database: {len(master_df):,}")# Historical Anomaly Analysis
# 1. Calculate the Long-Term Mean (LTM) for every district-month combination
# This gives you the "normal" rainfall for January in District X, February in District X, etc.
ltm = master_df.groupby(['ISO3', 'ADM_NAME', 'month'])['mean'].mean().reset_index()
ltm.rename(columns={'mean': 'historical_monthly_avg'}, inplace=True)

# 2. Merge this baseline back into your master database
df_final = pd.merge(master_df, ltm, on=['ISO3', 'ADM_NAME', 'month'])

# 3. Calculate the Anomaly (Actual Rain minus Normal Rain)
df_final['anomaly'] = df_final['mean'] - df_final['historical_monthly_avg']

# 4. DEFINE THE DROUGHT (Aim 1 Goal)
# A common index: If rainfall is less than 50% of the historical average for that month
df_final['is_drought'] = df_final['mean'] < (df_final['historical_monthly_avg'] * 0.5)

print(f"🚨 Drought events identified: {df_final['is_drought'].sum():,}")
# In[ ]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load your Master Database (from Step 6)
df = pd.read_csv('C:/Users/FlawiyaShirishMore/OneDrive - Africa Specialty Risks Ltd/ASR-Parametric_Research_Study/africa_risk/master_rainfall_africa.csv')

# 2. Load your Spatial Layer
# This is the 'skeleton' we created earlier
districts_shp = gpd.read_file('africa_admx_shp/africa_admx.shp')

# 3. CALCULATE DROUGHT INDEX (Reference: Percent of Normal Index)
# First, get the long-term mean (2000-2025) for each district
ltm = df.groupby(['ISO3', 'ADM_NAME'])['mean'].mean().reset_index()
ltm.rename(columns={'mean': 'long_term_avg'}, inplace=True)

# Merge back to the main dataframe
df = pd.merge(df, ltm, on=['ISO3', 'ADM_NAME'])

# Calculate PNI for a specific year (e.g., 2022)
df_2015 = df[df['year'] == 2015].copy()
df_2015['pni'] = (df_2015['mean'] / df_2015['long_term_avg']) * 100

# 4. CLASSIFICATION (WMO Scheme)
def classify_drought(pni):
    if pni < 55: return 3 # Extreme
    if pni < 70: return 2 # Severe
    if pni < 80: return 1 # Moderate
    return 0 # Normal

df_2015['drought_severity'] = df_2015['pni'].apply(classify_drought)

# 5. MERGE DATA WITH GEOMETRY
# Ensure the columns match (ADM_NAME and ISO3 are the keys)
map_data = districts_shp.merge(df_2015, on=['ISO3', 'ADM_NAME'])

# 6. CREATE THE CHOROPLETH
fig, ax = plt.subplots(1, 1, figsize=(12, 12))

# Plot the background (all of Africa in light grey)
districts_shp.plot(ax=ax, color='#eeeeee', edgecolor='#bcbcbc', linewidth=0.3)

# Plot the Drought Severity for 2022
map_data.plot(column='drought_severity', 
              ax=ax, 
              cmap='YlOrRd', # Yellow to Red
              legend=True,
              legend_kwds={'label': "Drought Severity (0:Normal, 3:Extreme)",
                           'orientation': "horizontal"})

ax.set_title("Africa Drought Risk Assessment (2015)", fontsize=16)
ax.axis('off')

plt.savefig("Africa_Drought_Choropleth.png", dpi=300)
plt.show()


# In[ ]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Calculate PNI for a specific year (e.g., 2022)
df_2020 = df[df['year'] == 2020].copy()
df_2020['pni'] = (df_2020['mean'] / df_2020['long_term_avg']) * 100

# 4. CLASSIFICATION (WMO Scheme)
def classify_drought(pni):
    if pni < 55: return 3 # Extreme
    if pni < 70: return 2 # Severe
    if pni < 80: return 1 # Moderate
    return 0 # Normal

df_2020['drought_severity'] = df_2020['pni'].apply(classify_drought)

# 5. MERGE DATA WITH GEOMETRY
# Ensure the columns match (ADM_NAME and ISO3 are the keys)
map_data = districts_shp.merge(df_2020, on=['ISO3', 'ADM_NAME'])

# 6. CREATE THE CHOROPLETH
fig, ax = plt.subplots(1, 1, figsize=(12, 12))

# Plot the background (all of Africa in light grey)
districts_shp.plot(ax=ax, color='#eeeeee', edgecolor='#bcbcbc', linewidth=0.3)

# Plot the Drought Severity for 2022
map_data.plot(column='drought_severity', 
              ax=ax, 
              cmap='YlOrRd', # Yellow to Red
              legend=True,
              legend_kwds={'label': "Drought Severity (0:Normal, 3:Extreme)",
                           'orientation': "horizontal"})

ax.set_title("Africa Drought Risk Assessment (2020)", fontsize=16)
ax.axis('off')

plt.savefig("Africa_Drought_Choropleth.png", dpi=300)
plt.show()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

# 1. Define your desired sample
sample_countries = ['EGY', 'ETH', 'KEN', 'NGA', 'ZAF', 'ZMB', 'SEN', 'MAR']

# 2. Filter the list to ONLY include countries that actually exist in your matrix
# This prevents the KeyError
available_countries = [c for c in sample_countries if c in corr_matrix.index]

print(f"Requested: {len(sample_countries)} countries")
print(f"Found in data: {len(available_countries)} countries")

# 3. Plot only the available ones
if len(available_countries) > 1:
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix.loc[available_countries, available_countries], 
                annot=True, 
                cmap='coolwarm', 
                center=0) # Center at 0 to clearly show positive/negative correlation
    plt.title("Aim 3: Inter-Regional Rainfall Correlation")
    plt.savefig("Presentation_Correlation_Heatmap.png")
    plt.show()
else:
    print("Error: Not enough matching countries found to create a heatmap.")


# In[ ]:


# 1. Print columns to see what we actually have
print("Current columns:", df.columns.tolist())

# 2. Standardize all column names to lowercase
df.columns = df.columns.str.lower()

# 3. Check if 'year' or 'date' exists (CHIRPS usually comes with dates)
# If your column is 'date', create the 'year' column
if 'year' not in df.columns and 'date' in df.columns:
    df['year'] = pd.to_datetime(df['date']).dt.year

print("Standardized columns:", df.columns.tolist())

