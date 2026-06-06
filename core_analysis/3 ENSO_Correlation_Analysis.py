"""
ENSO Correlation Analysis for AZP-Style Clusters
=================================================
Loads the same drought data + clustering parameters as
spatial_agglomerative_clustering.py, then correlates each
cluster's mean drought time series with the ENSO signal.

ENSO data sources (checked in order):
  1. Local climate_merged.csv (with NINO34 column)  — user-supplied path
  2. NOAA ONI (Oceanic Nino Index) download fallback
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
import plotly.express as px

warnings.filterwarnings("ignore")

# =============================================================
# CONFIG
# =============================================================
CLIMATE_CSV_PATH = "data/climate_merged.csv"  # set to path of climate_merged.csv if available

K = 0                   # 0 = auto-detect via silhouette
K_MAX = 7               # max K when auto-detecting
SPATIAL_NEIGHBORS = 10
LAMBDA = 0.1            # must match the clustering run
OUTPUT_DIR = os.path.join("outputs", "ENSO_Correlation_Analysis")
# =============================================================

BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(BASE)
OUT = os.path.join(PROJ, OUTPUT_DIR)
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. LOAD DROUGHT DATA & SHAPEFILE  (same as clustering script)
# ---------------------------------------------------------------------------
df_annual = pd.read_csv(
    os.path.join(PROJ, "outputs", "Week_11_correlation_map", "drought_annual.csv")
)

shp_paths = [
    os.path.join(PROJ, "data", "africa_agricultural_domain_2019",
                 "africa_agricultural_domain_2019.shp"),
    os.path.join(PROJ, "data", "africa-agricultural-domain-2019",
                 "africa_agricultural_domain_2019.shp"),
]
for shp_path in shp_paths:
    if os.path.isfile(shp_path):
        gdf = gpd.read_file(shp_path)
        break
else:
    raise FileNotFoundError(
        "Could not find africa_agricultural_domain_2019.shp in data/; "
        "expected one of: {}".format(shp_paths)
    )

pivot = df_annual.pivot_table(
    index="feature_id", columns="year", values="Drought_Days"
).fillna(0)

gdf["ADM_NAME"] = gdf["ADM_NAME"].astype(str).str.strip().str.upper()
gdf = gdf.drop_duplicates(subset="ADM_NAME")
gdf_c = gdf.set_index("ADM_NAME").to_crs(epsg=3857)
centroids = gdf_c.centroid

common = sorted(set(pivot.index) & set(gdf_c.index))
weather = StandardScaler().fit_transform(pivot.loc[common])

c = centroids.loc[common]
coords = np.column_stack([c.x, c.y])
geo = StandardScaler().fit_transform(coords)

X = np.column_stack([weather * (1 - LAMBDA), geo * LAMBDA])
spatial_graph = kneighbors_graph(coords, n_neighbors=SPATIAL_NEIGHBORS,
                                 mode="connectivity")

# Auto-detect optimal K
if K == 0:
    scores = []
    for k in range(2, K_MAX + 1):
        model = AgglomerativeClustering(
            n_clusters=k, connectivity=spatial_graph, linkage="ward"
        )
        labels = model.fit_predict(X)
        sil = silhouette_score(X, labels)
        scores.append((k, sil))
    K = max(scores, key=lambda x: x[1])[0]
    print(f"Optimal K = {K}")

model = AgglomerativeClustering(
    n_clusters=K, connectivity=spatial_graph, linkage="ward"
)
labels = model.fit_predict(X)
label_dict = dict(zip(common, labels))

# Cluster mean drought time series (pivot_g)
df = df_annual[df_annual["feature_id"].isin(common)].copy()
df["cluster"] = df["feature_id"].map(label_dict)
group_means = df.groupby(["cluster", "year"])["Drought_Days"].mean().reset_index()
pivot_g = group_means.pivot_table(
    index="year", columns="cluster", values="Drought_Days"
)
years = pivot_g.index.values

print(f"Clusters: {K}, Years: {years[0]}-{years[-1]}, "
      f"Districts: {len(common)}")

# Save location-level drought index time series for every aligned district
location_timeseries = pivot.loc[common].reset_index().melt(
    id_vars="feature_id",
    var_name="year",
    value_name="Drought_Days"
)
location_timeseries["cluster"] = location_timeseries["feature_id"].map(label_dict)
location_timeseries["ADM_NAME"] = location_timeseries["feature_id"]
location_timeseries = location_timeseries[["ADM_NAME", "cluster", "year", "Drought_Days"]]
location_timeseries_path = os.path.join(
    OUT, f"enso_location_timeseries_k{K}_l{str(LAMBDA).replace('.', '')}.csv"
)
location_timeseries.to_csv(location_timeseries_path, index=False)
print(f"Location timeseries CSV -> {location_timeseries_path}")

# ---------------------------------------------------------------------------
# 2. LOAD ENSO DATA
# ---------------------------------------------------------------------------
enso_annual = None

# Option A: local climate CSV with NINO34
if CLIMATE_CSV_PATH and os.path.isfile(CLIMATE_CSV_PATH):
    print(f"\nLoading NINO34 from: {CLIMATE_CSV_PATH}")
    clim = pd.read_csv(CLIMATE_CSV_PATH)
    clim["date"] = pd.to_datetime(clim["date"])
    clim["year"] = clim["date"].dt.year
    # Check NINO34 column name variants
    nino_col = [c for c in clim.columns
                if "NINO" in c.upper() or "NINO34" in c.upper()]
    if nino_col:
        nino_col = nino_col[0]
        enso_annual = (
            clim.groupby("year")[nino_col].mean().reset_index()
        )
        enso_annual.columns = ["year", "enso"]
        print(f"  Found column '{nino_col}', {len(enso_annual)} years")

# Option B: NOAA ONI download fallback
if enso_annual is None:
    print("\nDownloading NOAA ONI (Oceanic Nino Index) ...")
    oni_url = (
        "https://psl.noaa.gov/data/correlation/oni.data"
    )
    try:
        oni_raw = pd.read_csv(
            oni_url, skiprows=1, delim_whitespace=True,
            names=["year"] + [f"m{i}" for i in range(1, 13)],
            na_values="-99.9"
        )
        oni_raw = oni_raw.dropna(subset=["year"])
        oni_raw["year"] = oni_raw["year"].astype(int)
        # Annual mean of monthly ONI
        month_cols = [f"m{i}" for i in range(1, 13)]
        oni_raw["enso"] = oni_raw[month_cols].mean(axis=1, skipna=True)
        enso_annual = oni_raw[["year", "enso"]].copy()
        print(f"  Downloaded ONI, {len(enso_annual)} years")
    except Exception as e:
        print(f"  ONI download failed: {e}")
        print("  Falling back to hardcoded ONI approximation ...")
        # Hardcoded annual ONI for key years (2000–2025)
        oni_map = {
            2000: -0.7, 2001: -0.3, 2002: 0.8, 2003: 0.5,
            2004: 0.3, 2005: 0.4, 2006: 0.3, 2007: -0.6,
            2008: -0.7, 2009: 0.7, 2010: -0.7, 2011: -0.8,
            2012: 0.2, 2013: -0.2, 2014: 0.3, 2015: 1.5,
            2016: 0.8, 2017: 0.2, 2018: 0.3, 2019: 0.5,
            2020: -0.4, 2021: -0.7, 2022: -0.5, 2023: 1.2,
            2024: 1.8, 2025: 0.6,
        }
        enso_annual = pd.DataFrame(
            list(oni_map.items()), columns=["year", "enso"]
        )
        print(f"  Hardcoded ONI, {len(enso_annual)} years")

# Align with clustering years
enso_annual = enso_annual[enso_annual["year"].isin(years)].sort_values("year")
common_years = np.intersect1d(years, enso_annual["year"].values)
pivot_g = pivot_g.loc[common_years]
enso_vals = enso_annual.set_index("year").loc[common_years, "enso"].values

# Save district vs ENSO correlation table
location_vs_enso = []
for dist_id in common:
    dist_series = pivot.loc[dist_id, common_years].values
    r_val, p_val = pearsonr(dist_series, enso_vals)
    location_vs_enso.append({
        "ADM_NAME": dist_id,
        "cluster": label_dict[dist_id] + 1,
        "pearson_r": r_val,
        "pearson_p": p_val,
    })
location_vs_enso_df = pd.DataFrame(location_vs_enso)
location_vs_enso_path = os.path.join(
    OUT,
    f"enso_location_vs_enso_correlations_k{K}_l{str(LAMBDA).replace('.', '')}.csv"
)
location_vs_enso_df.to_csv(location_vs_enso_path, index=False)
print(f"Location-vs-ENSO correlation CSV -> {location_vs_enso_path}")

print(f"\nAligned years: {common_years[0]}-{common_years[-1]} "
      f"({len(common_years)} years)")

# ---------------------------------------------------------------------------
# 3. CORRELATION: each cluster vs ENSO
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"CLUSTER vs ENSO CORRELATION (lambda={LAMBDA}, K={K})")
print(f"{'='*60}")
results = []
for cid in range(K):
    r_pe, p_pe = pearsonr(pivot_g[cid].values, enso_vals)
    r_sp, p_sp = spearmanr(pivot_g[cid].values, enso_vals)
    sig_pe = "***" if p_pe < 0.001 else ("**" if p_pe < 0.01
                                          else ("*" if p_pe < 0.05 else "ns"))
    sig_sp = "***" if p_sp < 0.001 else ("**" if p_sp < 0.01
                                          else ("*" if p_sp < 0.05 else "ns"))
    results.append({
        "cluster": cid,
        "pearson_r": r_pe,
        "pearson_p": p_pe,
        "pearson_sig": sig_pe,
        "spearman_rho": r_sp,
        "spearman_p": p_sp,
        "spearman_sig": sig_sp,
    })
    print(f"  G{cid+1}: Pearson r={r_pe:+.3f} {sig_pe}"
          f"  |  Spearman rho={r_sp:+.3f} {sig_sp}")

# ---------------------------------------------------------------------------
# 4. VISUALIZATIONS
# ---------------------------------------------------------------------------
# Define consistent cluster color palette (matches spatial clustering)
CLUSTER_COLORS = px.colors.qualitative.Light24 if K <= 10 else px.colors.qualitative.Plotly
if K > len(CLUSTER_COLORS):
    # Extend palette if more clusters than available colors
    CLUSTER_COLORS = list(CLUSTER_COLORS) * (K // len(CLUSTER_COLORS) + 1)
CLUSTER_COLORS = CLUSTER_COLORS[:K]

cluster_names = [f"G{c+1}" for c in range(K)]
pearson_rs = [r["pearson_r"] for r in results]
spearman_rhos = [r["spearman_rho"] for r in results]
# Create color map by cluster ID (not by correlation sign)
cluster_color_map = {cid: CLUSTER_COLORS[cid] for cid in range(K)}
colors = [cluster_color_map[cid] for cid in range(K)]

# 4a. Bar chart: cluster vs ENSO correlation
fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(
    x=cluster_names,
    y=pearson_rs,
    marker_color=colors,
    text=[f"{v:+.3f}" for v in pearson_rs],
    textposition="outside",
    name="Pearson r",
))
fig_bar.add_trace(go.Bar(
    x=cluster_names,
    y=spearman_rhos,
    marker_color=colors,
    text=[f"{v:+.3f}" for v in spearman_rhos],
    textposition="outside",
    name="Spearman ρ",
    opacity=0.5,
))
fig_bar.update_layout(
    title=f"Cluster vs ENSO Correlation (λ={LAMBDA}, K={K})",
    yaxis_title="Correlation",
    yaxis_range=[-1, 1],
    template="plotly_white",
    barmode="group",
)
fig_bar.write_html(os.path.join(OUT, f"enso_correlation_k{K}_l{str(LAMBDA).replace('.','')}.html"))
print(f"\nBar chart -> {OUT}/enso_correlation_k{K}_l{str(LAMBDA).replace('.','')}.html")

# 4b. Dual-axis time series: cluster trajectories + ENSO
fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
for cid in range(K):
    fig_dual.add_trace(
        go.Scatter(
            x=common_years, y=pivot_g[cid].values,
            mode="lines+markers",
            name=f"G{cid+1}",
            line=dict(color=cluster_color_map[cid], width=2),
        ),
        secondary_y=False,
    )
fig_dual.add_trace(
    go.Scatter(
        x=common_years, y=enso_vals,
        mode="lines+markers",
        name="ENSO (ONI)",
        line=dict(color="black", width=3, dash="dot"),
    ),
    secondary_y=True,
)
fig_dual.update_layout(
    title=f"Cluster Drought Trajectories vs ENSO (λ={LAMBDA}, K={K})",
    xaxis_title="Year",
    template="plotly_white",
)
fig_dual.update_yaxes(title_text="Mean Drought Days / Year", secondary_y=False)
fig_dual.update_yaxes(title_text="ONI (°C)", secondary_y=True)
fig_dual.write_html(os.path.join(OUT, f"enso_trajectories_k{K}_l{str(LAMBDA).replace('.','')}.html"))
print(f"Dual-axis plot -> {OUT}/enso_trajectories_k{K}_l{str(LAMBDA).replace('.','')}.html")

# ---------------------------------------------------------------------------
# Save cluster mean correlation matrix and HTML heatmap
# ---------------------------------------------------------------------------
cluster_corr = pivot_g.corr(method="pearson")
cluster_corr_path = os.path.join(OUT, f"enso_cluster_corr_matrix_k{K}_l{str(LAMBDA).replace('.','')}.csv")
cluster_corr.to_csv(cluster_corr_path)
print(f"Cluster correlation matrix CSV -> {cluster_corr_path}")

fig_corr_matrix = go.Figure(
    go.Heatmap(
        z=cluster_corr.values,
        x=cluster_corr.columns.astype(str),
        y=cluster_corr.index.astype(str),
        colorscale="RdBu_r",
        zmin=-1,
        zmax=1,
        colorbar=dict(title="Pearson r")
    )
)
fig_corr_matrix.update_layout(
    title=f"Cluster Mean Drought Correlation Matrix (λ={LAMBDA}, K={K})",
    xaxis_title="Cluster",
    yaxis_title="Cluster",
    template="plotly_white"
)
fig_corr_matrix_path = os.path.join(OUT, f"enso_cluster_corr_matrix_k{K}_l{str(LAMBDA).replace('.','')}.html")
fig_corr_matrix.write_html(fig_corr_matrix_path)
print(f"Cluster correlation heatmap -> {fig_corr_matrix_path}")

# ---------------------------------------------------------------------------
# 5. SUMMARY
# ---------------------------------------------------------------------------
summary_path = os.path.join(OUT, f"enso_summary_k{K}_l{str(LAMBDA).replace('.','')}.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(f"ENSO Correlation Summary (λ={LAMBDA}, K={K})\n")
    f.write(f"{'='*50}\n")
    f.write(f"Years: {common_years[0]}–{common_years[-1]}\n")
    f.write(f"Districts: {len(common)}\n")
    f.write(f"ENSO source: {'climate_merged.csv' if CLIMATE_CSV_PATH else 'NOAA ONI'}\n\n")
    f.write(f"{'Cluster':<10} {'Pearson r':<12} {'p':<10} {'Spearman ρ':<12} {'p':<10}\n")
    f.write(f"{'-'*55}\n")
    for r in results:
        f.write(f"G{r['cluster']+1:<9} {r['pearson_r']:+.3f}      "
                f"{r['pearson_p']:.4f}    "
                f"{r['spearman_rho']:+.3f}      "
                f"{r['spearman_p']:.4f}\n")
print(f"Summary -> {summary_path}")
print(f"\nDone - outputs in: {OUT}")

# ---------------------------------------------------------------------------
# 5. NEW: DISTRICT-LEVEL ENSO CORRELATION MAP
# ---------------------------------------------------------------------------
print("\nGenerating District-level ENSO Correlation Map...")

# Calculate correlation for each district individually
district_corrs = []
for dist_id in common:
    # Get drought days for this specific district across the aligned years
    dist_series = pivot.loc[dist_id, common_years].values
    
    # Calculate Pearson correlation with ENSO
    r_val, p_val = pearsonr(dist_series, enso_vals)
    
    district_corrs.append({
        "ADM_NAME": dist_id,
        "ENSO_Correlation": r_val,
        "P_Value": p_val,
        "Sensitivity": "High" if p_val < 0.05 else "Low"
    })

# Convert to DataFrame and merge with the GeoDataFrame
df_dist_corr = pd.DataFrame(district_corrs)
gdf_enso_map = gdf_c.loc[common].reset_index().merge(df_dist_corr, on="ADM_NAME")

# Create the interactive map
# We use a Diverging color scale (RdBu_r): 
# Red = Positive Correlation (El Niño = More Drought)
# Blue = Negative Correlation (El Niño = Less Drought)
m_enso = gdf_enso_map.explore(
    column="ENSO_Correlation",
    cmap="RdBu_r",
    vmin=-1, vmax=1,
    legend=True,
    tooltip=["ADM_NAME", "ENSO_Correlation", "Sensitivity"],
    popup=True,
    tiles="CartoDB positron"
)

# Save the map
enso_map_path = os.path.join(OUT, f"enso_district_map_l{str(LAMBDA).replace('.','')}.html")
m_enso.save(enso_map_path)
print(f"Interactive ENSO Map saved to: {enso_map_path}")

# Save cluster membership map
cluster_geo = gdf_c.loc[common].reset_index()
cluster_geo["cluster"] = cluster_geo["ADM_NAME"].map(lambda x: label_dict.get(x, -1) + 1)
cluster_map = cluster_geo.explore(
    column="cluster",
    cmap="tab20",
    legend=True,
    tooltip=["ADM_NAME", "cluster"],
    popup=True,
    tiles="CartoDB positron"
)
cluster_map_path = os.path.join(OUT, f"enso_cluster_map_k{K}_l{str(LAMBDA).replace('.','')}.html")
cluster_map.save(cluster_map_path)
print(f"Cluster membership map saved to: {cluster_map_path}")