#!/usr/bin/env python
# coding: utf-8

import ee
import pandas as pd
import numpy as np
import os

# ==========================================
# 1. CONFIGURATION & INITIALIZATION
# ==========================================
PROJECT_ID = "vernal-parser-412016"
ASSET_ID = f"projects/{PROJECT_ID}/assets/africa-glad-50-2019"
FOLDER_NAME = "Agri_Drought_Study_Data"

def initialize_ee(project_id):
    """Initializes Google Earth Engine with the specified project ID."""
    try:
        ee.Initialize(project=project_id)
        print(f"✅ Earth Engine Initialized for project: {project_id}")
    except Exception:
        print("🔑 Authentication required...")
        ee.Authenticate()
        ee.Initialize(project=project_id)

# ==========================================
# 2. PROCESSING FUNCTIONS
# ==========================================
def export_ndvi_year(year, agri_districts, modis_ndvi):
    """
    Calculates monthly NDVI averages for a given year and submits an export task.
    """
    months = ee.List.sequence(1, 12)

    def calculate_monthly_ndvi(m):
        m = ee.Number(m)
        start_date = ee.Date.fromYMD(ee.Number(year), m, 1)
        end_date = start_date.advance(1, 'month')

        month_col = modis_ndvi.filterDate(start_date, end_date)
        img_present = month_col.mean().multiply(0.0001).rename('ndvi')
        
        stats_present = img_present.reduceRegions(
            collection=agri_districts,
            reducer=ee.Reducer.mean(),
            scale=5566
        ).map(lambda f: f.set({
            'ndvi': f.get('mean'), 
            'data_status': 'original',
            'month': m,
            'year': year
        }))

        stats_missing = agri_districts.map(lambda f: f.set({
            'ndvi': None, 
            'data_status': 'null_injected',
            'month': m,
            'year': year
        }))

        return ee.FeatureCollection(
            ee.Algorithms.If(month_col.size().gt(0), stats_present, stats_missing)
        )

    yearly_stats = ee.FeatureCollection(months.map(calculate_monthly_ndvi)).flatten()

    task = ee.batch.Export.table.toDrive(
        collection=yearly_stats,
        description=f"MODIS_NDVI_Africa_{year}",
        folder=FOLDER_NAME,
        fileNamePrefix=f"ndvi_{year}",
        fileFormat='CSV',
        selectors=['ISO3', 'COUNTRY', 'ADM_NAME', 'year', 'month', 'ndvi', 'data_status']
    )
    
    task.start()
    print(f"🚀 Export task submitted for NDVI year: {year}")

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Initialize connection
    initialize_ee(PROJECT_ID)

    # Load static datasets
    try:
        agri_districts = ee.FeatureCollection(ASSET_ID)
        modis_collection = ee.ImageCollection("MODIS/061/MOD13A2").select('NDVI')
        
        years_to_process = list(range(2000, 2026))
        
        print(f"📊 Dataset Loaded: {ASSET_ID}")
        print(f"📅 Submitting {len(years_to_process)} NDVI tasks to GEE...")

        for yr in years_to_process:
            export_ndvi_year(yr, agri_districts, modis_collection)

        print("\n✅ All tasks successfully submitted.")

    except Exception as e:
        print(f"❌ Error loading assets or submitting tasks: {e}")