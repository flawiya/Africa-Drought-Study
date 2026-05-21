#!/usr/bin/env python
# coding: utf-8

# In[2]:


#COMPLETE GOOGLE EARTH ENGINE + JUPYTER FIX
# =========================================
# NUCLEAR CLEANSE for `ee` import errors & `fcntl` issues on Windows

# WHAT IT DOES:
# 1. Finds your Python packages folder
# 2. ⚠️  DELETES corrupted `ee` & `blessings` folders completely  
# 3. Reinstalls CLEAN Earth Engine API + Jupyter tools
# 4. Forces kernel restart reminder

# WHEN TO RUN: When you see these errors:
# - ModuleNotFoundError: No module named 'ee'  
# - No module named 'fcntl'
# - ImportError: cannot import name 'ee'

import os
import shutil
import sys
import ee

print("=" * 60)
print("🚀 EARTH ENGINE JUPYTER NUCLEAR FIX STARTING...")
print("=" * 60)

# =============================================================================
# STEP 1: LOCATE PYTHON PACKAGES FOLDER
# =============================================================================
print("\n📍 1. FINDING PYTHON SITE-PACKAGES...")
import site
package_path = site.getsitepackages()[0]  
print(f"   ✅ Found packages at: {package_path}")

# Verify we're in right environment
print(f"   ✅ Python executable: {sys.executable}")

# =============================================================================
# STEP 2: DELETE CORRUPTED FOLDERS (THE NUCLEAR PART ⚠️)
# =============================================================================
print("\n💣 2. DELETING PROBLEMATIC LIBRARIES...")
bad_folders = [
    'ee',           # Corrupted Earth Engine remnants
    'blessings',    # Windows-incompatible Unix terminal lib
    'earthengine_api'  # Backup name for old installs
]

deleted_count = 0
for folder in bad_folders:
    full_path = os.path.join(package_path, folder)
    if os.path.exists(full_path):
        print(f"   💥 DELETING: {folder}")
        shutil.rmtree(full_path)
        deleted_count += 1
    else:
        print(f"   ⏭️  Already gone: {folder}")

print(f"   ✅ Deleted {deleted_count} problematic folders")

# =============================================================================
# STEP 3: FORCE FRESH INSTALL
# =============================================================================
print("\n🔄 3. INSTALLING CLEAN EARTH ENGINE API...")
print("   (This takes 1-2 minutes...)")

# Clean uninstall first, then fresh install
uninstall_cmd = f"{sys.executable} -m pip uninstall earthengine-api ee blessings -y"
install_cmd = f"{sys.executable} -m pip install --upgrade --no-cache-dir earthengine-api geemap pandas folium ipyleaflet"

print(f"   🗑️  Uninstalling old versions...")
os.system(uninstall_cmd)

print(f"   📦 Installing fresh...")
os.system(install_cmd)

print("   ✅ Installation complete!")

# =============================================================================
# STEP 4: TEST IMPORT (OPTIONAL)
# =============================================================================
print("\n🧪 4. QUICK TEST...")
try:
    import ee
    print("   ✅ SUCCESS: `import ee` works!")
except ImportError as e:
    print(f"   ⚠️  Import failed (normal before restart): {e}")

# =============================================================================
# FINAL INSTRUCTIONS
# =============================================================================
print("\n" + "="*60)
print("✅ FIX COMPLETE! DO THIS NEXT:")
print("="*60)
print("""
1. TOP MENU → KERNEL → RESTART & CLEAR OUTPUT
2. New cell → RUN THESE LINES:

import ee
ee.Authenticate()     # Google login (one-time)
ee.Initialize()

import geemap
Map = geemap.Map()    # Test map
Map                  # Displays interactive map
""")
print("="*60)
print("🎉 You now have CLEAN Earth Engine in Jupyter!")
print("="*60)


# In[1]:


import ee
# Force re-authrentication to get fresh code token with the right scopes
ee.Authenticate(force=True)

#Re-intialize with GCP ID
ee.Initialize(project="vernal-parser-412016")


# In[3]:


import ee
# Initialize
PROJECT_ID = "vernal-parser-412016"
try:
    ee.Initialize(project=PROJECT_ID)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)


# In[5]:


# Load your Final Agricultural Districts Asset
asset_id = f"projects/{PROJECT_ID}/assets/africa-glad-50-2019"
agri_districts = ee.FeatureCollection(asset_id)

# Load CHIRPS Daily Rainfall Data
chirps_daily = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select('precipitation')

def export_year_stats(year):
    """Calculates monthly rainfall for all districts and exports a CSV to Drive."""

    # Create a list of months (1 to 12)
    months = ee.List.sequence(1, 12)

    def calculate_monthly_stats(m):
        # A. TEMPORAL AGGREGATION
        # Filter CHIRPS for the specific year and month, then SUM the daily images.
        # This gives us ONE image where each pixel represents the total monthly rainfall.
        monthly_total_image = chirps_daily \
            .filter(ee.Filter.calendarRange(year, year, 'year')) \
            .filter(ee.Filter.calendarRange(m, m, 'month')) \
            .sum()

        # B. SPATIAL AGGREGATION
        # Run Zonal Statistics: Get the AVERAGE of that monthly total across the district
        stats = monthly_total_image.reduceRegions(
            collection=agri_districts,
            reducer=ee.Reducer.mean(),
            scale=5566 # CHIRPS native resolution (~5.5km)
        )

        # C. FORMATTING
        # Add the month and year to the table, and rename 'mean' to 'precip_mm' for clarity
        return stats.map(lambda f: f.set({
            'month': m,
            'year': year,
            'precip_mm': f.get('mean')
        }))

    # Flatten the 12 monthly collections into one single collection for the year
    yearly_stats = ee.FeatureCollection(months.map(calculate_monthly_stats)).flatten()

    # Export to Google Drive
    task = ee.batch.Export.table.toDrive(
        collection=yearly_stats,
        description=f"CHIRPS_Monthly_Rainfall_{year}",
        folder="Agri_Drought_Study_Data", # The folder that will be created in your Google Drive
        fileNamePrefix=f"rainfall_{year}",
        fileFormat='CSV',
        # These are the exact columns that will be saved in your CSV
        selectors=['ISO3', 'COUNTRY', 'ADM_NAME', 'year', 'month', 'precip_mm']
    )
    task.start()
    # Submit tasks from 2000 to 2025
years_to_process = list(range(2000, 2025))

print(f"Submitting {len(years_to_process)} extraction tasks to Google Earth Engine...")

for yr in years_to_process:
    export_year_stats(yr)

print("\n🚀 ALL TASKS SUBMITTED SUCCESSFULLY!")
print("Go to https://code.earthengine.google.com/ and check the 'Tasks' tab to monitor progress.")
print("The CSV files will automatically appear in your Google Drive folder 'Agri_Drought_Study_Data'.")


# In[ ]:




