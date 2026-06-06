"""
Spatially-Constrained Agglomerative Clustering (AZP-style)
==========================================================
Method: AgglomerativeClustering with k-NN spatial connectivity constraint.
Auto-determines optimal K using Silhouette Score.
Reference: Duque, Anselin & Rey (2012) — max-p regions model.
Implements spatial contiguity constraint via connectivity matrix,
achieving the same contiguity guarantee as REDCAP (Guo, 2008).
"""
import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr, pearsonr
import plotly.express as px
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore")
# =============================================================
# CONFIG 
# =============================================================
K = 0                          # 0 = auto-detect via silhouette, or set to fixed number
K_MAX = 7                     # max K to evaluate when auto-detecting
SPATIAL_NEIGHBORS = 10          # k for k-NN spatial contiguity graph
LAMBDA = 0.1                   # 0=pure weather, 1=pure geography
OUTPUT_DIR = "outputs/spatial_agglomerative_clustering"
# =============================================================
BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(BASE)
OUT = os.path.join(PROJ, OUTPUT_DIR)
os.makedirs(OUT, exist_ok=True)
# --- Load data ---
df_annual = pd.read_csv(os.path.join(PROJ, "outputs", "Week_11_correlation_map", "drought_annual.csv"))
gdf = gpd.read_file(os.path.join(PROJ, "data", "africa_agricultural_domain_2019", "africa_agricultural_domain_2019.shp"))
pivot = df_annual.pivot_table(index="feature_id", columns="year", values="Drought_Days").fillna(0)
gdf["ADM_NAME"] = gdf["ADM_NAME"].astype(str).str.strip().str.upper()
gdf = gdf.drop_duplicates(subset="ADM_NAME")
gdf_c = gdf.set_index("ADM_NAME").to_crs(epsg=3857)
centroids = gdf_c.centroid
common = list(set(pivot.index) & set(gdf_c.index))
weather = StandardScaler().fit_transform(pivot.loc[common])
c = centroids.loc[common]
coords = np.column_stack([c.x, c.y])
geo = StandardScaler().fit_transform(coords)
# --- Build spatial contiguity constraint ---
# k-NN graph ensures districts can only merge with nearby districts
# This enforces geographic compactness (same principle as REDCAP/AZP)
X = np.column_stack([weather * (1 - LAMBDA), geo * LAMBDA])
spatial_graph = kneighbors_graph(coords, n_neighbors=SPATIAL_NEIGHBORS, mode="connectivity")
# --- Auto-detect optimal K using Silhouette Score ---
if K == 0:
    print(f"Auto-detecting optimal K (1 to {K_MAX}) ...")
    scores = []
    for k in range(2, K_MAX + 1):
        model = AgglomerativeClustering(n_clusters=k, connectivity=spatial_graph, linkage="ward")
        labels = model.fit_predict(X)
        sil = silhouette_score(X, labels)
        scores.append((k, sil))
        print(f"  K={k:2d}: silhouette={sil:.4f}")
    # Pick K with highest silhouette
    K = max(scores, key=lambda x: x[1])[0]
    print(f"\nOptimal K = {K} (highest silhouette)")
# --- Final clustering with optimal K ---
model = AgglomerativeClustering(n_clusters=K, connectivity=spatial_graph, linkage="ward")
labels = model.fit_predict(X)
label_dict = dict(zip(common, labels))
# --- Build cluster map with consistent colors ---
gdf_map = gdf_c.loc[common].copy()
gdf_map["cluster"] = [f"G{l+1}" for l in labels]

# Create color mapping for geopandas
cluster_ids = gdf_map["cluster"].unique()
color_dict = {f"G{cid+1}": CLUSTER_COLORS[cid] for cid in range(K)}
gdf_map["color"] = gdf_map["cluster"].map(color_dict)

gdf_map = gdf_map.to_crs(epsg=4326)
gdf_map["geometry"] = gdf_map["geometry"].simplify(tolerance=0.05, preserve_topology=True)
gdf_map = gdf_map.reset_index()[["ADM_NAME", "cluster", "color", "geometry"]]

# Create interactive map with consistent colors
m = gdf_map.explore(column="cluster", legend=True, tiles="CartoDB positron",
                    cmap=lambda x: color_dict.get(x, "#cccccc"))
m.save(os.path.join(OUT, f"spatial_azp_k{K}_l{str(LAMBDA).replace('.','')}.html"))
# --- Group trajectories with consistent colors ---
df = df_annual[df_annual["feature_id"].isin(common)].copy()
df["cluster"] = df["feature_id"].map(label_dict)
means = df.groupby(["cluster", "year"])["Drought_Days"].mean().reset_index()

# Define consistent cluster color palette
CLUSTER_COLORS = px.colors.qualitative.Light24 if K <= 10 else px.colors.qualitative.Plotly
if K > len(CLUSTER_COLORS):
    CLUSTER_COLORS = list(CLUSTER_COLORS) * (K // len(CLUSTER_COLORS) + 1)
CLUSTER_COLORS = CLUSTER_COLORS[:K]

# Create color map by cluster ID
cluster_color_map = {cid: CLUSTER_COLORS[cid] for cid in range(K)}
means["color_id"] = means["cluster"].map(cluster_color_map)

fig = px.line(means, x="year", y="Drought_Days", color="cluster",
              title=f"AZP-Style Clusters (K={K}, auto-detected, λ={LAMBDA})")
# Apply explicit colors
for i, trace in enumerate(fig.data):
    if i < K:
        trace.line.color = CLUSTER_COLORS[i]
fig.write_html(os.path.join(OUT, f"spatial_trajectories_k{K}_l{str(LAMBDA).replace('.','')}.html"))
# --- Cluster profiles ---
print(f"\n{'='*60}")
print(f"CLUSTER PROFILES (K={K}, λ={LAMBDA})")
print(f"{'='*60}")
for cid in range(K):
    members = df[df["cluster"] == cid]
    districts = members["feature_id"].unique()
    mean_dd = members["Drought_Days"].mean()
    countries = gdf[gdf["ADM_NAME"].isin(districts)]["COUNTRY"].value_counts().head(3)
    print(f"\nG{cid+1}: {len(districts)} districts | {mean_dd:.1f} mean days/yr")
    for ctry, n in countries.items():
        print(f"  {ctry}: {n}")
# --- Inter-group Spearman correlation ---
group_means = df.groupby(["cluster", "year"])["Drought_Days"].mean().reset_index()
pivot_g = group_means.pivot_table(index="year", columns="cluster", values="Drought_Days")
print(f"\n{'='*60}")
print("INTER-GROUP CORRELATION (Spearman ρ)")
print(f"{'='*60}")
for i in range(K):
    for j in range(i + 1, K):
        rho, p_sp = spearmanr(pivot_g[i], pivot_g[j])
        r_pearson, p_pe = pearsonr(pivot_g[i], pivot_g[j])
        sig_s = "***" if p_sp < 0.001 else ("**" if p_sp < 0.01 else ("*" if p_sp < 0.05 else "ns"))
        sig_p = "***" if p_pe < 0.001 else ("**" if p_pe < 0.01 else ("*" if p_pe < 0.05 else "ns"))
        note = "[SYSTEMIC]" if abs(rho) > 0.7 else "[DIVERSIFY]" if abs(rho) < 0.3 else "[MODERATE]"
        print(f"  G{i+1} vs G{j+1}: Spearman ρ={rho:+.3f} {sig_s} | Pearson r ={r_pearson:.3f} {sig_p} {note}")

# --- Pearson correlation HTML table ---
import plotly.graph_objects as go
corr_data = []
for i in range(K):
    row = []
    for j in range(K):
        if i == j:
            row.append(1.0)
        else:
            a, b = (pivot_g[i], pivot_g[j]) if i < j else (pivot_g[j], pivot_g[i])
            r, p = pearsonr(a, b)
            row.append(r)
    corr_data.append(row)
fig_corr = go.Figure(data=go.Heatmap(
    z=corr_data,
    x=[f"G{c+1}" for c in range(K)],
    y=[f"G{c+1}" for c in range(K)],
    text=[[f"{v:+.3f}" for v in row] for row in corr_data],
    texttemplate="%{text}",
    colorscale="RdBu_r",
    zmin=-1, zmax=1,
))
fig_corr.update_layout(title=f"Inter-Cluster Pearson Correlation (λ={LAMBDA})")
fig_corr.write_html(os.path.join(OUT, f"pearson_matrix_k{K}_l{str(LAMBDA).replace('.','')}.html"))

print(f"\nDone — outputs in: {OUT}")

