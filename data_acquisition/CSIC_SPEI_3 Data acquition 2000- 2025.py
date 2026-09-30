#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ee

# Initialize
PROJECT_ID = "vernal-parser-412016"
ee.Initialize(project=PROJECT_ID)


# In[2]:


# Load your 3,333 Agricultural Districts
asset_id = f"projects/{PROJECT_ID}/assets/africa-glad-50-2019"
agri_districts = ee.FeatureCollection(asset_id)


# In[ ]:


# Load CSIC SPEI v2.10 - Select the 3-Month Scale
# This matches your monthly CHIRPS and ERA5 data
csic_spei = ee.ImageCollection("CSIC/SPEI/2_10").select('SPEI_03_month')

def export_csic_spei(year):
    months = ee.List.sequence(1, 12)

    def calculate_monthly(m):
        start_date = ee.Date.fromYMD(year, m, 1)
        end_date = start_date.advance(1, 'month')
        
        # Filter the collection for the month
        month_img = csic_spei.filterDate(start_date, end_date)
        
        # BRANCH A: DATA EXISTS
        def get_actual():
            stats = month_img.first().reduceRegions(
                collection=agri_districts,
                reducer=ee.Reducer.mean(),
                scale=55000 # CSIC native resolution (~50km)
            )
            return stats.map(lambda f: f.set({
                'spei_03': f.get('mean'),
                'data_status': 'original'
            }))
            
        # BRANCH B: DATA MISSING
        def get_null():
            return agri_districts.map(lambda f: f.set({
                'spei_03': None,
                'data_status': 'null_injected'
            }))

        # Conditional logic
        result = ee.FeatureCollection(
            ee.Algorithms.If(month_img.size().gt(0), get_actual(), get_null())
        )
        
        return result.map(lambda f: f.set({'month': m, 'year': year}))

    yearly_stats = ee.FeatureCollection(months.map(calculate_monthly)).flatten()

    # Export to Drive
    task = ee.batch.Export.table.toDrive(
        collection=yearly_stats,
        description=f"CSIC_SPEI03_Africa_{year}",
        folder="Agri_Drought_Study_Data", 
        fileNamePrefix=f"spei03_{year}",
        fileFormat='CSV',
        selectors=['ISO3', 'COUNTRY', 'ADM_NAME', 'year', 'month', 'spei_03', 'data_status']
    )
    task.start()
    print(f"Submitted CSIC SPEI-03 task for year: {year}")

# Run for 2000 to 2023 (CSIC v2.10 limit)
years = list(range(2000, 2023))
for yr in years:
    export_csic_spei(yr)

