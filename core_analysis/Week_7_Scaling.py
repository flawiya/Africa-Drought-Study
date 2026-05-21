#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import geopandas as gpd
import numpy as np


# While traditional drought monitoring often relies on precipitation (e.g., Standardized Precipitation Index - SPI), precipitation alone does not account for evapotranspiration, soil water retention, or previous moisture states. The Standardized Soil Moisture Index (SSI) is a more direct indicator of Agricultural Drought because it measures the water actually available to the crop's root zone. Soil moisture anomalies are more strongly correlated with crop yield fluctuations than rainfall anomalies (Carrão et al., 2016). In parametric insurance, using soil moisture reduces Basis Risk—the discrepancy between the index trigger and actual yield loss. Standard monthly climatologies (comparing a month to its historical average) can mask short-term, high-intensity dry spells known as "Flash Droughts." By using a 25-year daily baseline for each specific Julian Day (e.g., comparing Jan 1st, 2024, only to previous Jan 1sts), we capture the rapid onset of moisture stress.Recent studies emphasize that monthly-resolution indices are too coarse to capture the phenological sensitivity of crops like maize during critical stages like silking or grain filling (Wang et al., 2024; Stagge et al., 2025). 
# The decision to filter the data for January 1st through April 30th is based on the primary growing season for much of Sub-Saharan Africa. This period covers the vegetative, flowering, and early maturity stages. Water stress during this specific window is the primary driver of yield failure in rain-fed agriculture.
# We have calculated daily SSI values. However, for a dissertation analysis or an insurance contract, we need an Annual Metric per district. We will calculate two key insurance indicators:
# Drought Frequency (Count of Extreme Days): The number of days in the risk window where SSI fell below -1.5 (Severe/Extreme Drought).
# Drought Intensity (Sum of SSI): The cumulative moisture deficit over the period.
# Aggregating daily stress into a seasonal total allows for Spatial Clustering. By having one value per year per district, we can apply Machine Learning (like K-Means or PCA) to identify regions with synchronized risk, which is vital for Portfolio Diversification in insurance (Shruthi et al., 2025).

# In[ ]:


# =========================================================
# 1. LOAD TEMPORAL DATA
# =========================================================
# This file contains your raw 'volumetric_soil_water_layer_2'
path = r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk\Drought\data\Africa_Agri_districts_ERA5_LAND_DAILY_AGGR_2000_2026_timeseries.csv"
df = pd.read_csv(path)

# =========================================================
# 2. CALCULATE DAILY SSI (CLIMATOLOGY BY DAY OF YEAR)
# =========================================================
print("Calculating Daily Climatology...")

# Step A: Group by district and Day of Year (doy) to get the 25-year baseline for each specific day
climatology = df.groupby(['feature_id', 'doy'])['volumetric_soil_water_layer_2'].agg(
    mean_sm='mean', 
    std_sm='std'
).reset_index()

# Step B: Merge the baseline back into the main data
df = df.merge(climatology, on=['feature_id', 'doy'], how='left')

# Step C: Calculate the SSI (Standardized Soil Moisture Index)
# We add a tiny epsilon (1e-6) to the denominator to avoid division by zero
df['SSI'] = (df['volumetric_soil_water_layer_2'] - df['mean_sm']) / (df['std_sm'] + 1e-6)

# =========================================================
# 3. FILTER FOR JANUARY TO APRIL (THE RISK PERIOD)
# =========================================================
print("Filtering for Risk Period (Jan 1 - Apr 30)...")
df_risk = df[(df['month'] >= 1) & (df['month'] <= 4)].copy()

# =========================================================
# 4. LOAD AND OPTIMIZE CONTINENTAL SHAPEFILE
# =========================================================
print("Loading and Simplifying Shapefile...")
shp_path = r"C:\Users\FlawiyaShirishMore\OneDrive - Africa Specialty Risks Ltd\ASR-Parametric_Research_Study\africa_risk\Drought\Output\africa_agricultural_domain_2019\africa_agricultural_domain_2019.shp"
gdf_africa = gpd.read_file(shp_path)

# Simplify geometry to save memory (CRITICAL for 3,000+ districts)
gdf_africa['geometry'] = gdf_africa['geometry'].simplify(tolerance=0.02, preserve_topology=True)

# =========================================================
# 5. STANDARDIZE AND MERGE
# =========================================================
print("Merging Data...")
gdf_africa['ADM_NAME'] = gdf_africa['ADM_NAME'].astype(str).str.strip().str.upper()
df_risk['feature_id'] = df_risk['feature_id'].astype(str).str.strip().str.upper()

# Merge
africa_merged_risk = gdf_africa.merge(
    df_risk, 
    left_on='ADM_NAME', 
    right_on='feature_id', 
    how='inner'
)

print("Done! Data is merged, SSI is calculated, and period is filtered.")
print(africa_merged_risk[['ADM_NAME', 'date', 'SSI']].head())


# In[ ]:


# =========================================================
# 5. CALCULATE ANNUAL AGGREGATED METRICS (PER YEAR, PER DISTRICT)
# =========================================================
print("Aggregating daily data into annual risk metrics...")

# Step A: Define the Extreme Threshold
# Standard drought literature defines SSI < -1.5 as "Severe/Extreme Drought"
threshold = -1.5

# Step B: Create a binary marker for extreme days
africa_merged_risk['is_extreme'] = (africa_merged_risk['SSI'] <= threshold).astype(int)

# Step C: Group by District and Year to get Annual Totals
# We calculate both the Count (Frequency) and the Sum (Magnitude/Intensity)
df_annual = africa_merged_risk.groupby(['ADM_NAME', 'year']).agg(
    Extreme_Days_Count=('is_extreme', 'sum'),
    Seasonal_SSI_Sum=('SSI', 'sum'),
    Mean_Soil_Moisture=('volumetric_soil_water_layer_2', 'mean')
).reset_index()

# =========================================================
# 6. RE-ATTACH GEOMETRY
# =========================================================
# Since the groupby turned our data back into a regular DataFrame, 
# we re-attach the geometry from our original gdf_africa for mapping.
final_risk_map = gdf_africa[['ADM_NAME', 'geometry']].merge(
    df_annual, on='ADM_NAME', how='inner'
)

# =========================================================
# 7. EXPORT DATA FOR DISSERTATION ANALYSIS
# =========================================================
# Exporting as CSV for statistical analysis (Stata/R/Excel)
df_annual.to_csv("Africa_Annual_Drought_Risk_2000_2026.csv", index=False)

# Exporting as Shapefile/GeoJSON for GIS software (QGIS/ArcGIS)
final_risk_map.to_file("Africa_Drought_Risk_Spatial.shp")

print("Success! Annual metrics calculated.")
print(f"Total Administrative Units: {df_annual['ADM_NAME'].nunique()}")
print(df_annual.head())


# In the previous step, we condensed millions of daily data points into a single "Annual Risk Profile" per district. Now, we must visualize the geographic distribution of these failures.
# The 2024 Benchmark: Southern Africa experienced one of its worst droughts in decades in 2024 due to a powerful El Niño. By mapping this specific year, we can "ground-truth" our model. If our index shows high counts of extreme days in the Gwembe Valley (Zambia) or Matabeleland (Zimbabwe), we validate that the SSI accurately reflects physical disaster (World Food Programme, 2024).
# 
# Static maps are limited. For a continent as large as Africa, a researcher needs to hover over specific districts to see exact values.
# Justification: Interactive maps allow for "Details-on-Demand" (Shneiderman, 1996). In a parametric insurance context, underwriters use these visualizations to determine Accumulation Risk—ensuring they don't sell too many policies in a single geographic cluster that might fail simultaneously.
# 
# Rendering 3,333 complex polygons in an interactive browser format (HTML/Plotly) is computationally expensive.
# Methodology: We apply a Douglas-Peucker algorithm (via simplify) to the geometries. This reduces the vertex count while preserving the essential shape of the administrative boundaries. This is standard practice in GIS for optimizing web-based dashboards (Roth, 2013).

# In[ ]:


import plotly.express as px
import json

# =========================================================
# 8. PREPARE 2024 MAP DATA
# =========================================================
print("Preparing the 2024 Risk Map...")

# Step A: Filter for the target year (2024)
# We choose 2024 because it is the most recent extreme event for validation
map_2024 = final_risk_map[final_risk_map['year'] == 2024].copy()

# Step B: Convert GeoDataFrame to GeoJSON for Plotly
# This allows Plotly to draw the polygons
map_2024 = map_2024.to_crs(epsg=4326) # Ensure standard coordinate system
geojson = json.loads(map_2024.to_json())

# =========================================================
# 9. GENERATE INTERACTIVE CHOROPLETH (Plotly)
# =========================================================
fig = px.choropleth(
    map_2024,
    geojson=geojson,
    locations='ADM_NAME',      # The column in df_annual
    featureidkey='properties.ADM_NAME', # The key in the GeoJSON
    color='Extreme_Days_Count',
    color_continuous_scale="Reds",
    range_color=[0, 120],       # Max possible days in Jan-Apr is ~120
    scope="africa",             # Zoom into Africa
    labels={'Extreme_Days_Count': 'Days with SSI < -1.5'},
    title='2024 Africa Agricultural Drought: Count of Extreme Dry Days (Jan-Apr)'
)

fig.update_geos(
    visible=False, 
    resolution=50,
    showcountries=True, 
    countrycolor="Black"
)

fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})

# =========================================================
# 10. SAVE AS STANDALONE INTERACTIVE HTML
# =========================================================
fig.write_html("Africa_Drought_Risk_Map_2024.html")

print("Success! Interactive map saved as 'Africa_Drought_Risk_Map_2024.html'")
fig.show()


# By examining the 2024 map, your dissertation can discuss Systemic vs. Idiosyncratic Risk. If the map shows a solid block of red across entire countries (e.g., Zambia, Malawi, Zimbabwe), it proves that drought is a systemic risk. Systemic risk is difficult for local insurers to manage alone and usually requires International Reinsurance (Barnett & Mahul, 2007). 
# 
# While maps show where a drought happens, time-series analysis shows how often and how severe these events are. In parametric insurance design, this is known as Historical Burn Analysis. By calculating the number of extreme drought days every year for 25 years, we can estimate the Return Period (e.g., is a 60-day drought a 1-in-10-year event or a 1-in-50-year event?).
# 
# A critical concept in agricultural economics is the distinction between Systemic Risk (affects everyone) and Idiosyncratic Risk (affects only one farm or district). By plotting the National/Continental Average alongside specific districts, we can visually identify "Co-variance." If a district's line peaks exactly when the national average peaks (as in 2024), it indicates a systemic climate shock that requires large-scale disaster response (Barnett & Mahul, 2007).

# In[ ]:


import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# =========================================================
# 1. PREPARE DATA FOR TIMESERIES
# =========================================================
print("Calculating averages and preparing plotting dataframe...")

# We use the df_annual created in the previous step
# (Columns: ADM_NAME, year, Extreme_Days_Count)

# Step A: Calculate the Continental Average per year
df_continent_avg = df_annual.groupby('year')['Extreme_Days_Count'].mean().reset_index()
df_continent_avg['ADM_NAME'] = 'Continental Average'

# Step B: Combine District data with the Average data
# Note: For the plot to be readable, we concatenate them
df_plot = pd.concat([df_annual, df_continent_avg], ignore_index=True)

# =========================================================
# 2. CREATE INTERACTIVE LINE PLOT
# =========================================================
fig2 = px.line(
    df_plot, 
    x='year', 
    y='Extreme_Days_Count', 
    color='ADM_NAME',
    title='Historical Drought Frequency (2000-2025): Days with SSI ≤ -1.5',
    labels={'Extreme_Days_Count': 'Number of Extreme Days', 'year': 'Year', 'ADM_NAME': 'District'}
)

# =========================================================
# 3. CUSTOM STYLING (The "Boss's Layout")
# =========================================================
# Iterate through all lines to set the Continental Average to stand out
for trace in fig2.data:
    if trace.name == 'Continental Average':
        trace.line.color = 'darkblue'
        trace.line.width = 5      # Make it very thick
        trace.opacity = 1.0
    else:
        # All other 3,000+ districts become background context
        trace.line.color = 'lightblue'
        trace.line.width = 0.5    # Make them very thin
        trace.opacity = 0.3       # Make them faint

# Add 2024 Drought reference line
fig2.add_vline(
    x=2024, 
    line_width=2, 
    line_dash="dash", 
    line_color="black", 
    annotation_text="2024 Regional Crisis "
)

# Layout improvements
fig2.update_layout(
    template="plotly_white",
    hovermode="closest", # Shows the specific district you are hovering over
    showlegend=False,    # Hide legend because 3,000 items is too many
    yaxis_range=[0, 125] # Jan-Apr is 120 days total
)

# Save as HTML for the report
fig2.write_html("Africa_Drought_Frequency_Trend.html")

print("Success! Interactive trend chart saved as 'Africa_Drought_Frequency_Trend.html'")
fig2.show()


# The chart will likely show a significant spike in 2024 across most districts in Southern and Eastern Africa. This aligns with the Standardized 
# Precipitation Evapotranspiration Index (SPEI) reports which indicated that February 2024 was the driest on record for the region (Wang et al., 2024).
# Thresholding for Parametric Payouts
# The Y-axis (Count of Days) is the most common "Trigger" in index insurance. If a contract is written with a "Strike" of 40 days, any year where the line crosses the 40-day mark would automatically result in a payment to farmers.

# This dashboard implements Cross-Filtering, a technique where an action in one view (selecting a geographic coordinate) triggers a query and update in another view (temporal line chart). This allows to explore the Spatial Heterogeneity of drought. While the map shows that 2024 was a systemic crisis, the line chart reveals if a specific district is "structurally" prone to drought (frequent spikes) or if 2024 was a rare "Black Swan" event (Barnett & Mahul, 2007).
# The "Continental Average" as a Risk Benchmark
# In the line chart, we plot the Continental Average. This benchmark is essential to distinguish between Idiosyncratic Risk (localized issues like poor soil in one district) and Covariant Risk (regional climate shocks like El Niño). If the district line moves in perfect synchronization with the Continental line, it confirms the district is highly sensitive to the broader African climate cycle.

# In[ ]:


from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json

# ---------------------------------------------------------
# 1. DATA PREPARATION (Using optimized Africa datasets)
# ---------------------------------------------------------
metric = 'Extreme_Days_Count'

# Ensure names are standardized
gdf_africa['ADM_NAME'] = gdf_africa['ADM_NAME'].astype(str).str.strip().str.upper()
df_annual['ADM_NAME'] = df_annual['ADM_NAME'].astype(str).str.strip().str.upper()

# Create the 2024 Baseline for the Map
df_2024 = df_annual[df_annual['year'] == 2024].copy()
map_2024 = gdf_africa.merge(df_2024, on='ADM_NAME', how='inner')

# Optimization: Create a lightweight GeoJSON for Plotly
map_2024 = map_2024.to_crs(epsg=4326)
geojson_payload = json.loads(map_2024.to_json())

# Calculate Continental Average for the Line Chart Benchmark
df_continent_avg = df_annual.groupby('year')[metric].mean().reset_index()

# ---------------------------------------------------------
# 2. INITIALIZE DASH APP & LAYOUT
# ---------------------------------------------------------
app = Dash(__name__)

app.layout = html.Div(style={'font-family': 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif', 'padding': '20px'}, children=[
    html.H1("Africa Agricultural Drought Risk Dashboard", style={'textAlign': 'center', 'color': '#2c3e50'}),
    html.Div(style={'backgroundColor': '#ecf0f1', 'padding': '10px', 'borderRadius': '5px', 'marginBottom': '20px'}, children=[
        html.P([
            html.B("Objective: "), 
            "Quantifying spatiotemporal drought hotspots across 3,000+ districts. ",
            html.I("Map displays 2024 'Extreme Dry Days' (SSI ≤ -1.5).")
        ]),
        html.P("🖱️ Click a district on the map to analyze its 25-year historical trend vs. the African average.")
    ]),

    html.Div(style={'display': 'flex', 'gap': '20px'}, children=[
        # LEFT: CONTINENTAL MAP
        html.Div(style={'flex': '1'}, children=[
            dcc.Graph(
                id='africa-map',
                figure=px.choropleth(
                    map_2024,
                    geojson=geojson_payload,
                    locations='ADM_NAME',
                    featureidkey="properties.ADM_NAME",
                    color=metric,
                    color_continuous_scale='Reds',
                    range_color=[0, 100],
                    title="Spatial Risk: 2024 Extreme Drought Frequency"
                ).update_geos(fitbounds="locations", visible=False)
                 .update_layout(margin={"r":0,"t":40,"l":0,"b":0}, clickmode='event+select')
            )
        ]),

        # RIGHT: HISTORICAL TREND
        html.Div(style={'flex': '1'}, children=[
            dcc.Graph(id='district-trend-chart')
        ])
    ])
])

# ---------------------------------------------------------
# 3. INTERACTIVE CALLBACK
# ---------------------------------------------------------
@app.callback(
    Output('district-trend-chart', 'figure'),
    Input('africa-map', 'clickData')
)
def update_graph(clickData):
    # Default selection on load
    selected_district = "GWEMBE" 
    
    if clickData:
        selected_district = clickData['points'][0]['location']
    
    # Filter historical data for selected district
    d_history = df_annual[df_annual['ADM_NAME'] == selected_district].sort_values('year')
    
    fig = go.Figure()

    # District Line
    fig.add_trace(go.Scatter(
        x=d_history['year'], y=d_history[metric],
        mode='lines+markers', name=f'District: {selected_district}',
        line=dict(color='#e74c3c', width=4)
    ))

    # Continental Benchmark
    fig.add_trace(go.Scatter(
        x=df_continent_avg['year'], y=df_continent_avg[metric],
        mode='lines', name='Continental Avg',
        line=dict(color='#7f8c8d', width=2, dash='dot')
    ))

    fig.update_layout(
        title=f"Historical Burn: {selected_district} vs Continental Average",
        xaxis_title="Year",
        yaxis_title="Count of Extreme Days",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin={"r":10,"t":80,"l":10,"b":10}
    )
    
    # Highlight the 2024 crisis point
    fig.add_vline(x=2024, line_width=1, line_dash="dash", line_color="black")

    return fig

# ---------------------------------------------------------
# 4. RUN
# ---------------------------------------------------------
if __name__ == '__main__':
    app.run(jupyter_mode="inline", port=8060)


# In[ ]:





# In[ ]:


import geopandas as gpd
import pandas as pd
import plotly.express as px
import json

# ---------------------------------------------------------
# 1. PREPARE DATA (Linking your gdf_africa to df_annual)
# ---------------------------------------------------------
print("Preparing data for animation...")

# Use the column calculated in your previous step
metric = 'Extreme_Days_Count'

# Standardize names for merging (already done in previous steps, but good for safety)
gdf_africa['ADM_NAME'] = gdf_africa['ADM_NAME'].astype(str).str.strip().str.upper()
df_annual['ADM_NAME'] = df_annual['ADM_NAME'].astype(str).str.strip().str.upper()

# Sort data by year so the slider moves correctly from 2000 to 2026
df_animated = df_annual.sort_values(by=['year', 'ADM_NAME']).reset_index(drop=True)

# ---------------------------------------------------------
# 2. CREATE GEOMETRY PAYLOAD (GeoJSON)
# ---------------------------------------------------------
# Plotly works best when we provide the geometry as a GeoJSON object
# We use gdf_africa which you already simplified to 0.02
gdf_africa.set_index('ADM_NAME', inplace=True, drop=False)
geojson_data = json.loads(gdf_africa.geometry.to_json())

# ---------------------------------------------------------
# 3. GENERATE THE ANIMATED MAP (Year Slider)
# ---------------------------------------------------------
print("Generating Animation...")

# Set the color range based on the maximum drought days found in the dataset
max_val = df_animated[metric].max()

fig = px.choropleth(
    df_animated,
    geojson=geojson_data,
    locations='ADM_NAME',      # Column in df_annual
    featureidkey='id',         # Match GeoJSON 'id' which we set to ADM_NAME
    color=metric,              # The "Count of Extreme Days"
    animation_frame='year',    # THE SLIDER: Creates the time animation
    hover_name='ADM_NAME',
    color_continuous_scale='YlOrRd',
    range_color=[0, max_val],  # Keep scale constant so years are comparable
    title='<b>African Agricultural Drought Frequency (2000-2026)</b><br>Annual Count of Days where SSI ≤ -1.5',
    labels={metric: 'Extreme Drought Days'}
)

# Optimization: Zoom the map to the data
fig.update_geos(fitbounds="locations", visible=False)

# Clean up layout
fig.update_layout(
    margin={"r":0,"t":80,"l":0,"b":0},
    sliders=[{"currentvalue": {"prefix": "Year: "}}]
)

# ---------------------------------------------------------
# 4. SAVE AS INTERACTIVE HTML
# ---------------------------------------------------------
# This is the file you share in the chat/email
output_html = "Africa_Drought_Frequency_Animation.html"
fig.write_html(output_html, include_plotlyjs="cdn")

print(f"Success! Map saved as {output_html}")
fig.show()


# * Carrão, H., Russo, S., Sepulcre-Canto, G., & Barbosa, P. (2016). An empirical standardized soil moisture index for agricultural drought assessment. International Journal of Applied Earth Observation and Geoinformation. (Justifies the use of SSI over SPI).
# * Wang, Q., et al. (2024). The first global multi-timescale daily SPEI dataset. Scientific Data. (Justifies the move from monthly to daily temporal resolution).
# * Stagge, J. H., et al. (2025). Expected annual minima from an idealized moving-average drought index. Hydrology and Earth System Sciences. (Justifies the statistical approach to extreme values in drought).
# * Shruthi, S., et al. (2025). Satellite-based data for agricultural index insurance: a systematic quantitative literature review. Natural Hazards and Earth System Sciences. (Justifies the application of this data for parametric insurance design).
# * McKee, T. B., Doesken, N. J., & Kleist, J. (1993). The relationship of drought frequency and duration to time scales. (The seminal paper for the "Standardization" math we used for the SSI).
# * World Food Programme (2024). El Niño and Drought in Southern Africa. (Context for the 2024 validation).
# * Shneiderman, B. (1996). The Eyes Have It: A Task by Data Type Taxonomy for Information Visualizations. (The foundational theory for interactive dashboards).
# * Roth, R. E. (2013). Interactive maps: What we know and what we need to know. Journal of Spatial Information Science. (Justifies the UI/UX choices).
# * Barnett, B. J., & Mahul, O. (2007). Weather Index Insurance for Agriculture and Disaster Risk Management. World Bank Publications. (Justifies the use of geographic risk clusters for financial modeling).
# * Wang, Q., et al. (2024). The first global multi-timescale daily SPEI dataset from 1982 to 2021. Scientific Data. (Context for the 2024 drought event).
# * Shruthi, S., et al. (2025). Satellite-based data for agricultural index insurance. (Justification for using historical trends to set insurance strikes).
