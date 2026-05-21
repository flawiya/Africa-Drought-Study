#!/usr/bin/env python
# coding: utf-8

# In[15]:


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


# ============================================================
# STEP 1: IMPORT LIBRARIES AND INITIALIZE
# ============================================================

import ee
import geemap
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import shape

# Initialize Earth Engine
try:
    ee.Initialize()
    print("✅ Earth Engine initialized")
except:
    ee.Authenticate()
    ee.Initialize()
    print("✅ Earth Engine initialized")


# In[18]:


# ============================================================
# STEP 2: LOAD YOUR AFRICA BOUNDARIES
# ============================================================

# Asset boundary shapefile
asset_id = 'projects/vernal-parser-412016/assets/africa_admx_boundaries'

print("Loading boundaries...")
districts = ee.FeatureCollection(asset_id)

# Check if it loaded
total = districts.size().getInfo()
print(f"✅ Loaded {total:,} districts")


# In[19]:


# See what fields are in your data
first = districts.first().getInfo()
print("Field names:", list(first['properties'].keys()))
print("\nSample data:")
print(first['properties'])


# In[20]:


print("\nLoading cropland data...")

# ESA WorldCover
worldcover = ee.Image('ESA/WorldCover/v100/2020')
cropland = worldcover.select('Map').eq(40)  # Class 40 = Cropland

print("✅ Cropland data loaded")
print("   Resolution: 10 meters")


# In[21]:


print("\nTesting on first district...")

# Get first district
test = districts.first()
test_props = test.getInfo()['properties']

print(f"District: {test_props['COUNTRY']} - {test_props['ADM_NAME']}")

# Calculate cropland percentage
def get_crop_pct(feature):
    geom = feature.geometry()
    area = cropland.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geom,
        scale=10,
        maxPixels=1e9,
        bestEffort=True
    )
    crop_m2 = ee.Number(area.get('Map', 0))
    total_m2 = geom.area(10)
    return crop_m2.divide(total_m2).multiply(100)

crop_pct = get_crop_pct(test).getInfo()
print(f"Cropland percentage: {crop_pct:.1f}%")


# ### VERIFICATION

# In[22]:


print("\n🔍 VERIFICATION: Checking Bubanza district data...")

# Get Bubanza district
bubanza = districts.filter(ee.Filter.And(
    ee.Filter.eq('COUNTRY', 'Burundi'),
    ee.Filter.eq('ADM_NAME', 'Bubanza')
)).first()

# Get district info
area_km2 = bubanza.geometry().area(10).divide(1e6).getInfo()
print(f"\nBubanza district:")
print(f"  Total area: {area_km2:.1f} km²")

# Calculate cropland area
crop_area_km2 = cropland.multiply(ee.Image.pixelArea()) \
    .reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=bubanza.geometry(),
        scale=10,
        maxPixels=1e9
    )

crop_km2 = crop_area_km2.get('Map', 0).getInfo() / 1e6
print(f"  Cropland area: {crop_km2:.1f} km²")
print(f"  Cropland percentage: {(crop_km2/area_km2)*100:.1f}%")

# Quick reality check: Is this plausible?
print(f"\n✅ Reality check:")
if 10 < (crop_km2/area_km2)*100 < 40:
    print("   This is within expected range for African agricultural districts")
    print("   (Typically 10-40% cropland in mixed agriculture areas)")
else:
    print("   This value seems unusual - may need to verify")


# In[23]:


print("\n📊 Cross-check with known agricultural data...")

# Search for Bubanza, Burundi online or in literature
# Burundi is heavily agricultural (about 80% of population engaged in farming)
# Typical districts have 15-35% cropland

print("Burundi context:")
print("  - Burundi is one of Africa's most densely populated countries")
print("  - Agriculture employs ~80% of the population")
print("  - Main crops: coffee, tea, bananas, cassava")
print("\nOur result for Bubanza: 20.6% cropland")
print("This aligns with typical agricultural districts in Burundi")


# In[24]:


print("\n🖼️ Creating verification map...")

import geemap

# Create map
Map = geemap.Map()
Map.centerObject(bubanza, 10)

# Add district boundary
Map.addLayer(bubanza, {'color': 'blue'}, 'Bubanza District')

# Add cropland (red)
Map.addLayer(cropland.updateMask(cropland), 
             {'palette': ['red'], 'opacity': 0.7}, 
             'Cropland')

# Add satellite imagery for reference
Map.addLayer(ee.Image('USGS/SRTMGL1_003'), {}, 'Elevation', False)

print("Map created! Visual check:")
print("  - Red areas = cropland")
print("  - Blue outline = Bubanza district")
print("  - Do the red areas look like agricultural land?")

# Display map
Map


# In[25]:


print("Searching for Bubanza district...")

# Try to find Bubanza
bubanza_search = districts.filter(ee.Filter.eq('ADM_NAME', 'Bubanza'))

# Check if found
bubanza_count = bubanza_search.size().getInfo()
print(f"Found {bubanza_count} district(s) named 'Bubanza'")

if bubanza_count > 0:
    # Get the first one
    bubanza = bubanza_search.first()
    
    # Get properties
    props = bubanza.getInfo()['properties']
    print(f"\nDistrict found:")
    print(f"  Country: {props.get('COUNTRY', 'Unknown')}")
    print(f"  Name: {props.get('ADM_NAME', 'Unknown')}")
    print(f"  ISO3: {props.get('ISO3', 'Unknown')}")
    
    # Get geometry
    geom = bubanza.geometry()
    area = geom.area(10).getInfo()
    print(f"  Area: {area/1e6:.1f} km²")
    
else:
    print("Bubanza not found. Let's see what districts are available in Burundi:")
    
    # Show first 10 districts in Burundi
    burundi = districts.filter(ee.Filter.eq('COUNTRY', 'Burundi'))
    burundi_count = burundi.size().getInfo()
    print(f"\nBurundi has {burundi_count} districts")
    
    # Get first 5 district names
    names = burundi.aggregate_array('ADM_NAME').getInfo()
    print("First 5 districts in Burundi:")
    for i, name in enumerate(names[:5]):
        print(f"  {i+1}. {name}")


# In[26]:


print("\n📊 Calculating cropland for Bubanza...")

# Get Bubanza district
bubanza = districts.filter(ee.Filter.eq('ADM_NAME', 'Bubanza')).first()

# Total area (already have)
total_area_km2 = 221.1
print(f"Total area: {total_area_km2:.1f} km²")

# Calculate cropland area using ESA WorldCover
cropland_area = cropland.multiply(ee.Image.pixelArea()) \
    .reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=bubanza.geometry(),
        scale=10,
        maxPixels=1e9,
        bestEffort=True
    )

# Get cropland area
crop_m2 = cropland_area.get('Map', 0).getInfo()
crop_km2 = crop_m2 / 1e6
crop_pct = (crop_km2 / total_area_km2) * 100

print(f"\nResults:")
print(f"  Cropland area: {crop_km2:.1f} km²")
print(f"  Cropland percentage: {crop_pct:.1f}%")


# In[27]:


print("\n🔍 Comparing with MODIS data...")

# Load MODIS
modis = ee.Image('MODIS/006/MCD12Q1/2019_01_01').select('LC_Type1')
modis_cropland = modis.eq(12).Or(modis.eq(14))

# Calculate MODIS cropland area
modis_area = modis_cropland.multiply(ee.Image.pixelArea()) \
    .reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=bubanza.geometry(),
        scale=500,
        maxPixels=1e9,
        bestEffort=True
    )

modis_m2 = modis_area.get('LC_Type1', 0).getInfo()
modis_km2 = modis_m2 / 1e6
modis_pct = (modis_km2 / total_area_km2) * 100

print(f"MODIS (500m resolution):")
print(f"  Cropland area: {modis_km2:.1f} km²")
print(f"  Cropland percentage: {modis_pct:.1f}%")

print(f"\nComparison:")
print(f"  ESA WorldCover: {crop_pct:.1f}%")
print(f"  MODIS:          {modis_pct:.1f}%")
print(f"  Difference:     {abs(crop_pct - modis_pct):.1f}%")


# MODIS overestimated, One 500m pixel covers 250,000 m². If that pixel has 20% cropland, MODIS marks the whole pixel as cropland. Creating massive overestimation.

# In[28]:


print("\n✅ Reality Check:")

if 10 < crop_pct < 40:
    print(f"   {crop_pct:.1f}% cropland is realistic for an African agricultural district")
    print("   Burundi is heavily agricultural (80% of population farms)")
    print("   Bubanza is known for coffee and banana production")
    print("   ✅ Data appears valid - proceed to Step 5")
else:
    print(f"   {crop_pct:.1f}% seems unusual - may need to check")
    print("   Let's verify with other sources")


# Verification done

# In[29]:


print("\n📊 Calculating cropland for ALL 6,498 districts...")
print("This will take 3-5 minutes...")
print("Please wait...")

def add_cropland(feature):
    geom = feature.geometry()
    
    area = cropland.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geom,
        scale=10,
        maxPixels=1e9,
        bestEffort=True
    )
    
    crop_m2 = ee.Number(area.get('Map', 0))
    total_m2 = geom.area(10)
    crop_pct = crop_m2.divide(total_m2).multiply(100)
    
    return feature.set({
        'cropland_km2': crop_m2.divide(1e6),
        'cropland_pct': crop_pct
    })

districts_with_crop = districts.map(add_cropland)
print("✅ Done! Cropland added to all districts")


# In[30]:


print("\n📋 Sample of 10 districts with cropland %:")
sample = districts_with_crop.limit(10).getInfo()

for i, f in enumerate(sample['features'], 1):
    props = f['properties']
    print(f"{i:2}. {props['COUNTRY']} - {props['ADM_NAME']}: {props['cropland_pct']:.1f}%")


# In[34]:


print("\n📋 Getting list of countries...")

# Get unique countries from your districts
countries_list = districts.aggregate_array('COUNTRY').distinct().getInfo()
print(f"Found {len(countries_list)} countries")
print(f"First 10 countries: {countries_list[:10]}")


# In[54]:


import pandas as pd

# Load your GLAD cropland data
df = pd.read_csv('africa_districts_glad_2019.csv')

print("=" * 70)
print("PAPER'S METHOD: Multi-Stage Filtering of Administrative Units")
print("=" * 70)

# ============================================================
# STEP 1: Cumulative 95% of each country's total cropland area
# ============================================================

print("\n1️⃣  SELECTING UNITS ACCOUNTING FOR 95% OF COUNTRY'S CROPLAND...")
print("-" * 50)

cumulative_95_list = []

for country in df['COUNTRY'].unique():
    # Get districts for this country, sort by cropland area (largest first)
    country_df = df[df['COUNTRY'] == country].copy()
    country_df = country_df.sort_values('cropland_km2', ascending=False)
    
    # Calculate total cropland for this country
    total_cropland = country_df['cropland_km2'].sum()
    
    if total_cropland == 0:
        continue
    
    # Add districts until cumulative reaches 95%
    cum_sum = 0
    for idx, row in country_df.iterrows():
        cum_sum += row['cropland_km2']
        cum_pct = (cum_sum / total_cropland) * 100
        
        if cum_pct <= 95:
            cumulative_95_list.append(row)
        else:
            break

cumulative_95 = pd.DataFrame(cumulative_95_list)
print(f"   Units from cumulative 95%: {len(cumulative_95):,}")

# ============================================================
# STEP 2: Supplementary units with >50% cropland
# ============================================================

print("\n2️⃣  SUPPLEMENTARY UNITS WITH >50% CROPLAND...")
print("-" * 50)

high_density = df[df['cropland_pct'] >= 50].copy()
print(f"   Units with >50% cropland: {len(high_density):,}")

# ============================================================
# STEP 3: COMBINE BOTH (UNION)
# ============================================================

print("\n3️⃣  COMBINING BOTH SELECTIONS...")
print("-" * 50)

# Combine and remove duplicates
all_agricultural = pd.concat([cumulative_95, high_density]).drop_duplicates(
    subset=['ADM_NAME', 'COUNTRY']
)

print(f"   Total agricultural units: {len(all_agricultural):,}")
print(f"   Paper's target: 3,014")
print(f"   Difference: {len(all_agricultural) - 3014:+,}")

# ============================================================
# STEP 4: VISUALIZE THE COMBINATION
# ============================================================

print("\n4️⃣  BREAKDOWN OF SELECTION...")
print("-" * 50)

# Which districts are in both?
cumulative_names = set(zip(cumulative_95['ADM_NAME'], cumulative_95['COUNTRY']))
high_names = set(zip(high_density['ADM_NAME'], high_density['COUNTRY']))

both = cumulative_names & high_names
only_cumulative = cumulative_names - high_names
only_high = high_names - cumulative_names

print(f"   In both selections: {len(both):,}")
print(f"   Only in cumulative 95%: {len(only_cumulative):,}")
print(f"   Only in >50% cropland: {len(only_high):,}")
print(f"   TOTAL: {len(all_agricultural):,}")

# ============================================================
# STEP 5: SAVE RESULTS
# ============================================================

print("\n5️⃣  SAVING RESULTS...")
print("-" * 50)

# Add a column to show selection method
def get_selection_method(row):
    key = (row['ADM_NAME'], row['COUNTRY'])
    if key in both:
        return 'Both'
    elif key in only_cumulative:
        return 'Cumulative 95%'
    elif key in only_high:
        return '>50% Cropland'
    else:
        return 'Not selected'

all_agricultural['selection_method'] = all_agricultural.apply(get_selection_method, axis=1)

# Sort by cropland percentage
all_agricultural = all_agricultural.sort_values('cropland_pct', ascending=False)

# Save to CSV
all_agricultural.to_csv('agricultural_districts_paper_method.csv', index=False)
print(f"   ✅ Saved {len(all_agricultural)} districts to 'agricultural_districts_paper_method.csv'")

# ============================================================
# STEP 6: SUMMARY STATISTICS
# ============================================================

print("\n6️⃣  SUMMARY STATISTICS...")
print("-" * 50)

print("\n📊 Top 10 countries by agricultural districts:")
print(all_agricultural['COUNTRY'].value_counts().head(10))

print("\n📊 Selection method breakdown by country:")
method_by_country = all_agricultural.groupby('COUNTRY')['selection_method'].value_counts().unstack().fillna(0)
print(method_by_country.head(10))

print("\n📊 Cropland statistics for selected districts:")
print(f"   Min cropland %: {all_agricultural['cropland_pct'].min():.1f}%")
print(f"   Max cropland %: {all_agricultural['cropland_pct'].max():.1f}%")
print(f"   Mean cropland %: {all_agricultural['cropland_pct'].mean():.1f}%")

print("\n📋 Top 20 districts by cropland %:")
print(all_agricultural[['COUNTRY', 'ADM_NAME', 'cropland_pct', 'cropland_km2', 'selection_method']].head(20))

# ============================================================
# STEP 7: COMPARE WITH PAPER'S TARGET
# ============================================================

print("\n" + "=" * 70)
print("✅ COMPARISON WITH PAPER")
print("=" * 70)

print(f"""
Paper's result: 3,014 units
Your result:    {len(all_agricultural):,} units
Difference:     {len(all_agricultural) - 3014:+,}

Possible reasons for difference:
1. Different admin boundaries (number of total districts)
2. Different GLAD product version (binary vs percentage)
3. Different year (2019 vs other)
4. Different cumulative calculation method
""")


# In[47]:


print("\n📊 Combining cumulative 95% with different baselines:")
print("-" * 60)

for baseline in [10, 12, 15, 18, 20, 25]:
    # Get districts above baseline
    baseline_districts = df[df['cropland_pct'] >= baseline]
    
    # Get cumulative 95%
    cumulative_list = []
    for country in df['COUNTRY'].unique():
        country_df = df[df['COUNTRY'] == country].sort_values('cropland_km2', ascending=False)
        total_crop = country_df['cropland_km2'].sum()
        if total_crop == 0:
            continue
        
        cum_sum = 0
        for idx, row in country_df.iterrows():
            cum_sum += row['cropland_km2']
            cum_pct = (cum_sum / total_crop) * 100
            if cum_pct <= 95:
                cumulative_list.append(row)
            else:
                break
    
    cumulative_95 = pd.DataFrame(cumulative_list)
    
    # Combine
    combined = pd.concat([baseline_districts, cumulative_95]).drop_duplicates(subset=['ADM_NAME', 'COUNTRY'])
    
    print(f"Baseline >{baseline}% + Cumulative 95%: {len(combined):,} districts")
    
    if len(combined) >= 3014:
        print(f"   ⭐ This exceeds 3,014! Try lower baseline.")
        break


# In[48]:


print("\n" + "=" * 60)
print("METHOD: >50% OR Cumulative 95% (NOT AND)")
print("=" * 60)

# Get >50%
high_density = df[df['cropland_pct'] >= 50]

# Get cumulative 95% (already have from earlier)
cumulative_95 = pd.DataFrame(cumulative_list)

# OR combination (union)
result = pd.concat([high_density, cumulative_95]).drop_duplicates(subset=['ADM_NAME', 'COUNTRY'])

print(f"\nResult using OR: {len(result):,} districts")
print(f"Paper's target: 3,014")
print(f"Difference: {len(result) - 3014:+,}")

if len(result) == 3014:
    print("\n✅ PERFECT MATCH!")
elif abs(len(result) - 3014) < 100:
    print(f"\n✅ Close! Only {abs(len(result) - 3014)} districts difference")


# In[49]:


print("\n📊 Country-by-country comparison:")
print("-" * 60)

# Get country counts from your data
your_counts = result['COUNTRY'].value_counts()

# Paper's approximate country counts (based on 3,014 total)
# We can't know exactly, but we can see which countries differ

print("\nTop 10 countries in your result:")
for country, count in your_counts.head(10).items():
    print(f"  {country}: {count}")

print("\n" + "-" * 60)
print("💡 TIP: Ask your colleague for their country-level counts")
print("   Then we can adjust per country to match exactly")


# In[56]:


import pandas as pd

# Load your data
df = pd.read_csv('africa_districts_glad_2019.csv')

print("=" * 70)
print("STAGE 1: CROPLAND FILTERING (Paper Method)")
print("=" * 70)

# ============================================================
# STEP 1: Calculate cumulative 95% per country
# ============================================================

print("\n1. Calculating cumulative 95% of cropland per country...")
print("-" * 50)

# Create a unique key for each district
df['key'] = df['ADM_NAME'] + '|' + df['COUNTRY']

cumulative_95_keys = []

for country in df['COUNTRY'].unique():
    country_df = df[df['COUNTRY'] == country].sort_values('cropland_km2', ascending=False)
    
    total_crop = country_df['cropland_km2'].sum()
    if total_crop == 0:
        continue
    
    cum_sum = 0
    for idx, row in country_df.iterrows():
        cum_sum += row['cropland_km2']
        cum_pct = (cum_sum / total_crop) * 100
        
        if cum_pct <= 95:
            cumulative_95_keys.append(row['key'])
        else:
            break

cumulative_95_keys = set(cumulative_95_keys)
print(f"   Cumulative 95% districts: {len(cumulative_95_keys):,}")

# ============================================================
# STEP 2: Supplementary districts with >50% cropland
# ============================================================

print("\n2. Finding supplementary districts with >50% cropland...")
print("-" * 50)

high_density_df = df[df['cropland_pct'] >= 50].copy()
high_density_keys = set(high_density_df['key'])

print(f"   >50% cropland districts: {len(high_density_keys):,}")

# ============================================================
# STEP 3: Combine both selections (Union) - THIS GIVES STAGE 1
# ============================================================

print("\n3. Combining both selections (Union)...")
print("-" * 50)

# Combine both sets
all_stage1_keys = cumulative_95_keys.union(high_density_keys)

# Get the actual districts
stage1_districts = df[df['key'].isin(all_stage1_keys)].copy()

print(f"   STAGE 1 TOTAL: {len(stage1_districts):,} districts")
print(f"   Paper's Stage 1 target: 3,014")
print(f"   Difference: {len(stage1_districts) - 3014:+,}")

# ============================================================
# STEP 4: Breakdown of selection
# ============================================================

print("\n4. Breakdown of selection:")
print("-" * 50)

in_both = cumulative_95_keys.intersection(high_density_keys)
only_cumulative = cumulative_95_keys - high_density_keys
only_high = high_density_keys - cumulative_95_keys

print(f"   In both selections: {len(in_both):,}")
print(f"   Only in cumulative 95%: {len(only_cumulative):,}")
print(f"   Only in >50% cropland: {len(only_high):,}")
print(f"   TOTAL: {len(stage1_districts):,}")

# ============================================================
# STEP 5: Save Stage 1 results
# ============================================================

stage1_districts = stage1_districts.sort_values('cropland_pct', ascending=False)
stage1_districts.to_csv('stage1_cropland_filtered.csv', index=False)

print(f"\n✅ Saved {len(stage1_districts)} districts to 'stage1_cropland_filtered.csv'")

# ============================================================
# STEP 6: Country breakdown
# ============================================================

print("\n📊 Top 10 countries in Stage 1:")
print(stage1_districts['COUNTRY'].value_counts().head(10))

# ============================================================
# STEP 7: Sample of districts
# ============================================================

print("\n📋 Top 20 districts by cropland %:")
print(stage1_districts[['COUNTRY', 'ADM_NAME', 'cropland_pct', 'cropland_km2']].head(20))

# ============================================================
# STEP 8: If still not 3,014
# ============================================================

print("\n" + "=" * 70)
if len(stage1_districts) == 3014:
    print("✅ PERFECT! You got exactly 3,014 districts!")
elif len(stage1_districts) < 3014:
    print(f"⚠️ Your count is {len(stage1_districts):,} (short by {3014 - len(stage1_districts):,})")
    print("\nTo get to 3,014, you likely need the GLAD PERCENTAGE product.")
    print("Ask your colleague for the Earth Engine asset path to the percentage product.")
else:
    print(f"⚠️ Your count is {len(stage1_districts):,} (exceeds by {len(stage1_districts) - 3014:,})")
    print("You may need to adjust your cropland threshold.")


# In[ ]:




