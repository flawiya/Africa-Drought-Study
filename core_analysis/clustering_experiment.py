import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import plotly.express as px
# =============================================================
# CONFIG 
# =============================================================
K = 4                    # number of clusters (try 4, 6, 8, 10)
LAMBDA = 0.5             # spatial weight: 0=pure weather, 1=pure geography
OUTPUT_DIR = "experiments"
# =============================================================

BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(BASE)
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", os.path.join(PROJ, OUTPUT_DIR))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(PROJ, "data"))
OUT = os.path.join(OUTPUT_ROOT)
os.makedirs(OUT, exist_ok=True)


def require_input_path(path, description):
    if os.path.exists(path):
        return path

    basename = os.path.basename(path)
    print(f"  {description} not found at expected path; searching {DATA_DIR} for {basename}...")
    for root, _, files in os.walk(DATA_DIR):
        if basename in files:
            found_path = os.path.join(root, basename)
            print(f"  Found {description} at: {found_path}")
            return found_path

    raise FileNotFoundError(
        f"{description} not found:\n  {path}\n"
        "If the file lives elsewhere, set OUTPUT_ROOT or DATA_DIR to the correct path and rerun."
    )


def find_drought_annual_csv():
    candidates = [
        # common output used by Week_11 script
        os.path.join(OUTPUT_ROOT, "Week_11_correlation_map", "drought_annual.csv"),
        os.path.join(OUTPUT_ROOT, "Week 11 correlation map", "drought_annual.csv"),
        # direct outputs folder (when script was run with OUTPUT_ROOT=outputs)
        os.path.join(PROJ, "outputs", "Week_11_correlation_map", "drought_annual.csv"),
        os.path.join(PROJ, "outputs", "Week 11 correlation map", "drought_annual.csv"),
        # pipeline cache fallback
        os.path.join(OUTPUT_ROOT, "pipeline_cache", "df_annual.csv"),
        os.path.join(PROJ, "outputs", "pipeline_cache", "df_annual.csv"),
        # explicit outputs path (user-provided)
        os.path.join(PROJ, "outputs", "Week_11_correlation_map", "drought_annual.csv"),
    ]

    for path in candidates:
        if os.path.exists(path):
            print(f"Found drought CSV at: {path}")
            return path

    raise FileNotFoundError(
        "drought_annual.csv not found in the expected output folders.\n"
        "Run core_analysis/Week_11_correlation_map.py first to generate the file, "
        f"or place it in one of these locations:\n  {chr(10) + '  '.join(candidates)}"
    )


drought_csv_path = find_drought_annual_csv()
gdf_path = os.path.join(DATA_DIR, "africa-agricultural-domain-2019", "africa_agricultural_domain_2019.shp")
resolved_gdf = require_input_path(gdf_path, "GADM agricultural districts shapefile")


df_annual = pd.read_csv(drought_csv_path)
gdf = gpd.read_file(resolved_gdf)

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

X = np.column_stack([weather * (1 - LAMBDA), geo * LAMBDA])
labels = KMeans(n_clusters=K, random_state=42).fit_predict(X)
label_dict = dict(zip(common, labels))

gdf_map = gdf_c.loc[common].copy()
gdf_map["cluster"] = [f"G{l+1}" for l in labels]
gdf_map = gdf_map.to_crs(epsg=4326)
gdf_map["geometry"] = gdf_map["geometry"].simplify(tolerance=0.05, preserve_topology=True)
gdf_map = gdf_map.reset_index()[["ADM_NAME", "cluster", "geometry"]]

m = gdf_map.explore(column="cluster", legend=True, tiles="CartoDB positron")
m.save(os.path.join(OUT, f"cluster_k{K}_l{str(LAMBDA).replace('.','')}.html"))

df = df_annual[df_annual["feature_id"].isin(common)].copy()
df["cluster"] = df["feature_id"].map(label_dict)
means = df.groupby(["cluster", "year"])["Drought_Days"].mean().reset_index()
fig2 = px.line(means, x="year", y="Drought_Days", color="cluster", title=f"Group Drought Trajectories (k={K}, λ={LAMBDA})")
fig2.write_html(os.path.join(OUT, f"trajectories_k{K}_l{str(LAMBDA).replace('.','')}.html"))

print(f"Done — k={K}, λ={LAMBDA}, {len(common)} districts, {len(set(labels))} clusters")
print(f"   Outputs in: {OUT}")
