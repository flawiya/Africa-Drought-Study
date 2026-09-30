#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ee
import geemap
import pandas as pd
import numpy as np
import os


# In[2]:


# Initialize
PROJECT_ID = "vernal-parser-412016"
try:
    ee.Initialize(project=PROJECT_ID)
    print("Earth Engine Initialized")
except Exception:
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)


# In[4]:


# Load Datasets
asset_id = f"projects/{PROJECT_ID}/assets/africa-glad-50-2019"
agri_districts = ee.FeatureCollection(asset_id)


# In[ ]:


# ERA5-Land Monthly Aggregated Data
era5_col = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")

def export_era5_year(year):
    months = ee.List.sequence(1, 12)

    def calculate_monthly_era5(m):
        start_date = ee.Date.fromYMD(year, m, 1)
        end_date = start_date.advance(1, 'month')

        month_col = era5_col.filterDate(start_date, end_date)

        def get_actual_stats():
            img = month_col.first()

            temp = img.select('temperature_2m').subtract(273.15).rename('temp_c')
            pet = img.select('potential_evaporation_sum').multiply(-1000).rename('pet_mm')
            soil1 = img.select('volumetric_soil_water_layer_1').rename('soil_0_7cm')
            soil2 = img.select('volumetric_soil_water_layer_2').rename('soil_7_28cm')
            combined = ee.Image([temp, pet, soil1, soil2])

            stats = combined.reduceRegions(
                    collection=agri_districts,
                    reducer=ee.Reducer.mean(),
                    scale=5566
            )

            return stats.map(lambda f: f.set({
                'temp_c': f.get('temp_c'),
                'pet_mm': f.get('pet_mm'),
                'soil_0_7cm': f.get('soil_0_7cm'),
                'soil_7_28cm': f.get('soil_7_28cm'),
                'data_status': 'original'
            }))

        def get_null_stats():
            return agri_districts.map(lambda f: f.set({
                'temp_c': None,
                'pet_mm': None,
                'soil_0_7cm': None,
                'soil_7_28cm': None,
                'data_status': 'null_injected'
            }))

        monthly_stats = ee.FeatureCollection(
                ee.Algorithms.If(month_col.size().gt(0), get_actual_stats(), get_null_stats()))
        
        return monthly_stats.map(lambda f: f.set({'month': m, 'year': year}))
    
    yearly_stats = ee.FeatureCollection(months.map(calculate_monthly_era5)).flatten()
    task = ee.batch.Export.table.toDrive(
            collection=yearly_stats,
            description=f"ERA5_Land_Africa_{year}",
            folder="Agri_Drought_Study_Data",
            fileNamePrefix=f"era5_land_{year}",
            fileFormat='CSV',
            selectors=['ISO3', 'COUNTRY', 'ADM_NAME', 'year', 'month',
                       'temp_c', 'pet_mm', 'soil_0_7cm', 'soil_7_28cm', 'data_status']
    )
    task.start()
    print(f"Submitted ERA5 Export Task for year: {year}")

years_to_process = list(range(2000, 2026))
print(f"Submitting {len(years_to_process)} ERA5-Land tasks to Google Earth Engine...")

for yr in years_to_process:
    export_era5_year(yr)

print("\n All tasks submitted!")

