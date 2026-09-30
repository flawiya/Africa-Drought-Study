"""
ENSO Correlation Analysis for AZP-Style Clusters
=================================================
Loads drought data, performs clustering, and correlates 
results with the ENSO signal.
"""
import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import silhouette_score
from scipy.stats import pearsonr, spearmanr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# =============================================================
# CONFIG
# =============================================================
K = 3                   # Set to 3 to match your presentation
SPATIAL_NEIGHBORS = 10
LAMBDA = 0.1            
OUTPUT_DIR = ("outputs", "ENSO_Correlation_Analysis")

# Colors matched exactly to your presentation slides
COLOR_MAP = {
    "G1": "#08306b",  # Dark Blue (Northern/Sahel)
    "G2": "#8c510a",  # Brown (Southern Africa)
    "G3": "#00bfff"   # Deep Sky Blue (East Africa)
}

# =============================================================
# PATH SETUP
# =============================================================
BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(BASE, ".."))
OUT = os.path.join(PROJ, *OUTPUT_DIR)
os.makedirs(OUT, exist_ok=True)

# Path to the CSV from your first script
csv_path = os.path.join(PROJ, "outputs", "1 Africa_agricultural_districts_correlation_crop_calendar", "drought_annual.csv")
# Path to the Shapefile
shp_path = os.path.join(PROJ, "data", "africa-agricultural-domain-2019", "africa_agricultural_domain_2019.shp")

# Fallback check for Shapefile
if not os.path.exists(shp_path):
    shp_path = os.path.join(PROJ, "data", "africa_agricultural_domain_2019.shp")

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CRITICAL: Could not find CSV at {csv_path}. Please check the folder name.")

df_annual = pd.read_csv(csv_path)
gdf = gpd.read_file(shp_path)

# Standardize names
pivot = df_annual.pivot_table(index="feature_id", columns="year", values="Drought_Days").fillna(0)
pivot.index = pivot.index.astype(str).str.strip().str.upper()
gdf["ADM_NAME"] = gdf["ADM_NAME"].astype(str).str.strip().str.upper()
gdf = gdf.drop_duplicates(subset="ADM_NAME")
gdf_c = gdf.set_index("ADM_NAME").to_crs(epsg=3857)

common = sorted(list(set(pivot.index) & set(gdf_c.index)))

# Prepare Features
weather = StandardScaler().fit_transform(pivot.loc[common])
coords = np.column_stack([gdf_c.loc[common].centroid.x, gdf_c.loc[common].centroid.y])
geo = StandardScaler().fit_transform(coords)

X = np.column_stack([weather * (1 - LAMBDA), geo * LAMBDA])
spatial_graph = kneighbors_graph(coords, n_neighbors=SPATIAL_NEIGHBORS, mode="connectivity")

# ---------------------------------------------------------------------------
# 2. CLUSTERING
# ---------------------------------------------------------------------------
model = AgglomerativeClustering(n_clusters=K, connectivity=spatial_graph, linkage="ward")
labels = model.fit_predict(X)
label_dict = dict(zip(common, labels))

# Calculate group means
df_plot = df_annual.copy()
df_plot["feature_id"] = df_plot["feature_id"].astype(str).str.strip().str.upper()
df_plot = df_plot[df_plot["feature_id"].isin(common)]
df_plot["cluster"] = df_plot["feature_id"].map(label_dict)
pivot_g = df_plot.groupby(["cluster", "year"])["Drought_Days"].mean().reset_index().pivot(index="year", columns="cluster", values="Drought_Days")
years = pivot_g.index.values

# ---------------------------------------------------------------------------
# 3. ENSO DATA (ONI)
# ---------------------------------------------------------------------------
oni_map = {
    2000: -0.7, 2001: -0.3, 2002: 0.8, 2003: 0.5, 2004: 0.3, 2005: 0.4, 
    2006: 0.3, 2007: -0.6, 2008: -0.7, 2009: 0.7, 2010: -0.7, 2011: -0.8,
    2012: 0.2, 2013: -0.2, 2014: 0.3, 2015: 1.5, 2016: 0.8, 2017: 0.2, 
    2018: 0.3, 2019: 0.5, 2020: -0.4, 2021: -0.7, 2022: -0.5, 2023: 1.2, 2024: 1.1
}
enso_vals = [oni_map.get(y, 0) for y in years]

# ---------------------------------------------------------------------------
# 4. PLOT (MATCHING SLIDE 9)
# ---------------------------------------------------------------------------
fig_dual = make_subplots(specs=[[{"secondary_y": True}]])

for cid in range(K):
    grp_name = f"G{cid+1}"
    fig_dual.add_trace(
        go.Scatter(x=years, y=pivot_g[cid], name=grp_name,
                   line=dict(color=COLOR_MAP[grp_name], width=3)),
        secondary_y=False
    )

fig_dual.add_trace(
    go.Scatter(x=years, y=enso_vals, name="ENSO (ONI)",
               line=dict(color="black", width=2, dash="dot")),
    secondary_y=True
)

fig_dual.update_layout(title=f"Drought Trajectories vs ENSO (K={K})", template="plotly_white")
fig_dual.update_yaxes(title_text="Mean Drought Days", secondary_y=False)
fig_dual.update_yaxes(title_text="ONI Index (°C)", secondary_y=True)
fig_dual.write_html(os.path.join(OUT, "enso_trajectories.html"))

# ---------------------------------------------------------------------------
# 5. MAP (MATCHING SLIDE 8)
# ---------------------------------------------------------------------------
gdf_map = gdf_c.loc[common].copy()
gdf_map["cluster"] = [f"G{l+1}" for l in labels]
gdf_map = gdf_map.to_crs(epsg=4326).reset_index()

# Ensure map colors are in order G1, G2, G3
map_colors = [COLOR_MAP["G1"], COLOR_MAP["G2"], COLOR_MAP["G3"]]

m = gdf_map.explore(column="cluster", cmap=map_colors, legend=True, tiles="CartoDB positron")
m.save(os.path.join(OUT, "enso_cluster_map.html"))

print(f"DONE! Files saved in: {OUT}")