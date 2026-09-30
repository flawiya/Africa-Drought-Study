"""
Spatially-Constrained Agglomerative Clustering (AZP-style)
==========================================================
Method: AgglomerativeClustering with k-NN spatial connectivity constraint.
Auto-determines optimal K using Silhouette Score.
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
OUTPUT_DIR = ("outputs", "spatial_agglomerative_clustering")

# =============================================================
# PATH SETUP
# =============================================================
# Get the absolute path of the current script
BASE = os.path.dirname(os.path.abspath(__file__))
# Get the Project Root (one level up from core_analysis)
PROJ = os.path.abspath(os.path.join(BASE, ".."))

# FIX: Use * to unpack the tuple into individual directory strings
OUT = os.path.join(PROJ, *OUTPUT_DIR)
os.makedirs(OUT, exist_ok=True)

# Define file paths
csv_path = os.path.join(PROJ, "outputs", "1 Africa_agricultural_districts_correlation_crop_calendar", "drought_annual.csv")
shp_path = os.path.join(PROJ, "data", "africa-agricultural-domain-2019", "africa_agricultural_domain_2019.shp")

# Fallback: Check if file exists, if not, try looking in the parent 'data' folder
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
# Standardize names in CSV index
pivot = df_annual.pivot_table(index="feature_id", columns="year", values="Drought_Days").fillna(0)
pivot.index = pivot.index.astype(str).str.strip().str.upper()

# Standardize names in Shapefile
gdf["ADM_NAME"] = gdf["ADM_NAME"].astype(str).str.strip().str.upper()
gdf = gdf.drop_duplicates(subset="ADM_NAME")
gdf_c = gdf.set_index("ADM_NAME").to_crs(epsg=3857) # Project for accurate centroids

centroids = gdf_c.centroid
common = list(set(pivot.index) & set(gdf_c.index))

if len(common) == 0:
    print(f"Sample CSV IDs: {list(pivot.index)[:5]}")
    print(f"Sample SHP IDs: {list(gdf_c.index)[:5]}")
    raise ValueError("ERROR: No matching ADM_NAMEs found between CSV and Shapefile.")

print(f"Found {len(common)} matching districts.")

# Prepare Data Arrays
weather = StandardScaler().fit_transform(pivot.loc[common])
c = centroids.loc[common]
coords = np.column_stack([c.x, c.y])
geo = StandardScaler().fit_transform(coords)

# --- Build Features & Spatial Constraint ---
X = np.column_stack([weather * (1 - LAMBDA), geo * LAMBDA])
# k-NN graph ensures districts can only merge with nearby districts
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
    
    K = max(scores, key=lambda x: x[1])[0]
    print(f"Selected Optimal K = {K}")

# --- Final clustering ---
model = AgglomerativeClustering(n_clusters=K, connectivity=spatial_graph, linkage="ward")
labels = model.fit_predict(X)
label_dict = dict(zip(common, labels))

# --- Export Map ---
gdf_map = gdf_c.loc[common].copy()
gdf_map["cluster"] = [f"G{l+1}" for l in labels]
gdf_map = gdf_map.to_crs(epsg=4326)
gdf_map["geometry"] = gdf_map["geometry"].simplify(tolerance=0.01)
gdf_map = gdf_map.reset_index()[["ADM_NAME", "cluster", "geometry"]]

map_out = os.path.join(OUT, f"spatial_map_k{K}.html")
# Use a simple color list based on K
m = gdf_map.explore(column="cluster", legend=True, cmap=["#08306b", "#8c510a", "#00bfff"], tiles="CartoDB positron")
m.save(map_out)

# --- Export Trajectories ---
# Normalize feature_id in original df to match common list
df_plot = df_annual.copy()
df_plot["feature_id"] = df_plot["feature_id"].astype(str).str.strip().str.upper()
df_plot = df_plot[df_plot["feature_id"].isin(common)]
df_plot["cluster"] = df_plot["feature_id"].map(label_dict)

means = df_plot.groupby(["cluster", "year"])["Drought_Days"].mean().reset_index()
means["cluster"] = means["cluster"].apply(lambda x: f"G{int(x)+1}")

fig = px.line(means, x="year", y="Drought_Days", color="cluster",
              title=f"Drought Trends by Spatial Cluster (K={K})")
fig.write_html(os.path.join(OUT, f"trajectories_k{K}.html"))

# --- Print Summary ---
print(f"\n{'='*60}\nCLUSTER SUMMARY\n{'='*60}")
for cid in range(K):
    members = df_plot[df_plot["cluster"] == cid]
    districts = members["feature_id"].unique()
    avg = members["Drought_Days"].mean()
    print(f"G{cid+1}: {len(districts)} districts | Avg Annual Drought Days: {avg:.1f}")

print(f"\nSUCCESS! Files saved in: {OUT}")