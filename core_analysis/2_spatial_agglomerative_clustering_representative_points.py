"""
2_spatial_agglomerative_clustering_representative_points.py
============================================================
Method: AgglomerativeClustering with k-NN spatial connectivity constraint
using GIS-gold-standard `representative_point()` instead of `.centroid`.

Why representative_point()?
---------------------------
Standard mathematical centroids (.centroid) can lie OUTSIDE district polygon boundaries
for concave/horseshoe-shaped districts or multi-island geometries. `.representative_point()`
is 100% guaranteed to land strictly INSIDE the interior of the district polygon.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import silhouette_score
import plotly.express as px

# =============================================================
# CONFIG — edit these if needed
# =============================================================
K = 0                          # 0 = auto-detect via silhouette, or set to fixed number
K_MAX = 7                      # max K to evaluate when auto-detecting
SPATIAL_NEIGHBORS = 10         # k for k-NN spatial contiguity graph
LAMBDA = 0.1                   # 0=pure weather, 1=pure geography
OUTPUT_DIR = ("outputs", "spatial_agglomerative_clustering_rep_pts")

# =============================================================
# PATH SETUP
# =============================================================
BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(BASE, ".."))

OUT = os.path.join(PROJ, *OUTPUT_DIR)
os.makedirs(OUT, exist_ok=True)

csv_path = os.path.join(PROJ, "outputs", "1 Africa_agricultural_districts_correlation_crop_calendar", "drought_annual.csv")
shp_path = os.path.join(PROJ, "data", "africa-agricultural-domain-2019", "africa_agricultural_domain_2019.shp")

if not os.path.exists(shp_path):
    shp_path = os.path.join(PROJ, "data", "africa_agricultural_domain_2019.shp")

print(f"Project Root: {PROJ}")
print(f"Looking for CSV: {csv_path}")
print(f"Looking for SHP: {shp_path}")

# --- Load data ---
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CRITICAL: Could not find drought_annual.csv at {csv_path}")
if not os.path.exists(shp_path):
    raise FileNotFoundError(f"CRITICAL: Could not find Shapefile at {shp_path}.")

df_annual = pd.read_csv(csv_path)
gdf = gpd.read_file(shp_path)

# --- Preprocessing ---
pivot = df_annual.pivot_table(index="feature_id", columns="year", values="Drought_Days").fillna(0)
pivot.index = pivot.index.astype(str).str.strip().str.upper()

gdf["ADM_NAME"] = gdf["ADM_NAME"].astype(str).str.strip().str.upper()
gdf = gdf.drop_duplicates(subset="ADM_NAME")
gdf_c = gdf.set_index("ADM_NAME").to_crs(epsg=3857)  # Project to meters

# GIS GOLD STANDARD: Use representative_point() to guarantee 100% inside-boundary compliance
rep_points = gdf_c.geometry.representative_point()
common = list(set(pivot.index) & set(gdf_c.index))

if len(common) == 0:
    print(f"Sample CSV IDs: {list(pivot.index)[:5]}")
    print(f"Sample SHP IDs: {list(gdf_c.index)[:5]}")
    raise ValueError("ERROR: No matching ADM_NAMEs found between CSV and Shapefile.")

print(f"Found {len(common)} matching districts.")

# Prepare Data Arrays
weather = StandardScaler().fit_transform(pivot.loc[common])
pts = rep_points.loc[common]
coords = np.column_stack([pts.x, pts.y])
geo = StandardScaler().fit_transform(coords)

# --- Build Features & Spatial Constraint ---
X = np.column_stack([weather * (1 - LAMBDA), geo * LAMBDA])
spatial_graph = kneighbors_graph(coords, n_neighbors=SPATIAL_NEIGHBORS, mode="connectivity")

# --- Auto-detect optimal K using Silhouette Score ---
if K == 0:
    print(f"\nAuto-detecting optimal K (2 to {K_MAX}) ...")
    scores = []
    for k in range(2, K_MAX + 1):
        model = AgglomerativeClustering(n_clusters=k, connectivity=spatial_graph, linkage="ward")
        labels = model.fit_predict(X)
        sil = silhouette_score(X, labels)
        scores.append((k, sil))
        print(f"  K={k:2d}: silhouette={sil:.4f}")
    
    best_k, best_score = max(scores, key=lambda x: x[1])
    print(f"\n-> Optimal K = {best_k} (silhouette = {best_score:.4f})")
    K = best_k

# --- Final Clustering Run ---
model = AgglomerativeClustering(n_clusters=K, connectivity=spatial_graph, linkage="ward")
gdf_c.loc[common, "Cluster"] = model.fit_predict(X)

# --- Save Map & Results ---
gdf_out = gdf_c.loc[common].to_crs(epsg=4326).reset_index()
fig = px.choropleth_mapbox(
    gdf_out,
    geojson=gdf_out.__geo_interface__,
    locations="ADM_NAME",
    featureidkey="properties.ADM_NAME",
    color="Cluster",
    mapbox_style="carto-positron",
    zoom=3,
    center={"lat": 2.5, "lon": 17.5},
    title=f"Drought Risk Regionalization (Representative Points, K={K})"
)

html_path = os.path.join(OUT, f"drought_clustering_rep_points_K{K}.html")
fig.write_html(html_path)
print(f"\n[OK] Saved Interactive Map: {html_path}")
