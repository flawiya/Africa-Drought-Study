#!/usr/bin/env python
# coding: utf-8

# In[2]:


import ee
import geemap
import pandas as pd
import numpy as np
import os


# In[ ]:


# Initialize
PROJECT_ID = "vernal-parser-412016"
try:
    ee.Initialize(project=PROJECT_ID)
    print("Earth Engine Initialized")
except Exception:
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)


# In[ ]:


# Load Datasets
asset_id = f"projects/{PROJECT_ID}/assets/africa-glad-50-2019"
agri_districts = ee.FeatureCollection(asset_id)


# In[ ]:


# MODIS Terra Land Surface Temperature 8-Day (1km Resolution)
modis_lst = ee.ImageCollection("MODIS/061/MOD11A2").select('LST_Day_1km')

def export_lst_year(year):
    months = ee.List.sequence(1, 12)

    def calculate_monthly_lst(m):
        start_date = ee.Date.fromYMD(year, m, 1)
        end_date = start_date.advance(1, 'month')
        month_col = modis_lst.filterDate(start_date, end_date)

        def get_actual_stats():
            img = month_col.mean().multiply(0.02).subtract(273.15).rename('lst_c')
            stats = img.reduceRegions(
                    collection=agri_districts,
                    reducer=ee.Reducer.mean(),
                    scale=5566
            )

            return stats.map(lambda f: f.set({
                'lst_c': f.get('mean'),
                'data_status': 'original'
            }))

        def get_null_stats():
            return agri_districts.map(lambda f: f.set({
                'lst_c': None,
                'data_status': 'null_injected'
            }))

        monthly_stats = ee.FeatureCollection(
                ee.Algorithms.If(month_col.size().gt(0), get_actual_stats(), get_null_stats())
        )

        return monthly_stats.map(lambda f: f.set({'month': m, 'year': year}))

    yearly_stats = ee.FeatureCollection(months.map(calculate_monthly_lst)).flatten()
    task = ee.batch.Export.table.toDrive(
        collection=yearly_stats,
        description=f"MODIS_LST_Africa_{year}",
        folder="Agri_Drought_Study_Data",
        fileNamePrefix=f"lst_{year}",
        fileFormat='CSV',
        selectors=['ISO3', 'COUNTRY', 'ADM_NAME', 'year', 'month', 'lst_c', 'data_status'])
    task.start()
    print(f"Submitted LST Export Task for year: {year}")

years_to_process = list(range(2000, 2026))
print(f"Submitting {len(years_to_process)} MODIS LST tasks to Google Earth Engine...")

for yr in years_to_process:
    export_lst_year(yr)

print("\n All Tasks Submitted")
print("Check your progress bar:")

