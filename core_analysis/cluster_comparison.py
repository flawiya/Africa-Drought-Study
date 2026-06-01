"""cluster_comparison.py
Utility to run and compare multiple clustering algorithms on annual
drought-day time series produced by the Gamma-SSI pipeline.

Usage (example):
  python core_analysis/cluster_comparison.py \
    --df-annual path/to/df_annual.csv \
    --gdf-maize path/to/valid_maize.geojson

If you don't supply inputs, the script will try to import and run
the main pipeline in `core_analysis/Week 11 correlation map.py` to
produce `df_annual` and `valid_maize`. That is optional but handy.

Outputs (written to outputs/clustering_comparison):
  - clusters_<method>_k<k>.csv  (ADM_NAME -> cluster label)
  - metrics_summary.csv          (rows per run with evaluation metrics)

Implemented methods: KMeans, Agglomerative, DBSCAN, GaussianMixture.
Optional: HDBSCAN if installed.

This is intended as a quick comparison harness — after you inspect
the outputs you can pick one approach to integrate into the mapping
pipeline in `Week 11 correlation map.py`.
"""

import os
import argparse
import json
import warnings
import importlib.util
from itertools import combinations

import numpy as np
import pandas as pd
import geopandas as gpd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.stats import spearmanr

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except Exception:
    HDBSCAN_AVAILABLE = False


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "clustering_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PIPELINE_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "pipeline_cache")
os.makedirs(PIPELINE_CACHE_DIR, exist_ok=True)
DF_ANNUAL_CACHE = os.path.join(PIPELINE_CACHE_DIR, "df_annual.csv")
GDF_MAIZE_CACHE = os.path.join(PIPELINE_CACHE_DIR, "valid_maize.geojson")


def load_week11_pipeline_module():
    module_path = os.path.join(os.path.dirname(__file__), "Week 11 correlation map.py")
    if not os.path.exists(module_path):
        raise FileNotFoundError(
            "Expected Week 11 pipeline script at 'Week 11 correlation map.py' in core_analysis"
        )
    spec = importlib.util.spec_from_file_location("week_11_correlation_map", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_inputs(df_annual_path=None, gdf_maize_path=None):
    """Load or compute `df_annual` (feature_id, year, Drought_Days)
    and `valid_maize` (GeoDataFrame with ADM_NAME and geometry).
    """
    df_annual = None
    gdf_maize = None

    if df_annual_path and os.path.exists(df_annual_path):
        df_annual = pd.read_csv(df_annual_path, dtype={"feature_id": str})
    elif os.path.exists(DF_ANNUAL_CACHE):
        print(f"Loading cached df_annual from {DF_ANNUAL_CACHE}")
        df_annual = pd.read_csv(DF_ANNUAL_CACHE, dtype={"feature_id": str})

    if gdf_maize_path and os.path.exists(gdf_maize_path):
        gdf_maize = gpd.read_file(gdf_maize_path)
    elif os.path.exists(GDF_MAIZE_CACHE):
        print(f"Loading cached valid_maize from {GDF_MAIZE_CACHE}")
        gdf_maize = gpd.read_file(GDF_MAIZE_CACHE)

    if df_annual is None or gdf_maize is None:
        try:
            week11 = load_week11_pipeline_module()
        except Exception as exc:
            missing = []
            if df_annual is None:
                missing.append("df_annual")
            if gdf_maize is None:
                missing.append("valid_maize")
            raise FileNotFoundError(
                f"{', '.join(missing)} not provided and Week 11 pipeline import failed. Provide --df-annual and/or --gdf-maize."
            ) from exc

        if df_annual is None:
            print("Running Week 11 pipeline to generate df_annual (this may take time)...")
            pipeline_main = getattr(week11, "main")
            df_annual = pipeline_main()
            df_annual.to_csv(DF_ANNUAL_CACHE, index=False)
            print(f"Saved cached df_annual to {DF_ANNUAL_CACHE}")

        if gdf_maize is None:
            print("Loading valid_maize from Week 11 pipeline helpers...")
            load_shapefiles = getattr(week11, "load_shapefiles")
            filter_maize1 = getattr(week11, "filter_maize1")
            join_districts_to_maize1 = getattr(week11, "join_districts_to_maize1")
            gdf_districts, gdf_geoglam = load_shapefiles()
            gdf_maize1 = filter_maize1(gdf_geoglam)
            gdf_maize = join_districts_to_maize1(gdf_districts, gdf_maize1)
            gdf_maize.to_file(GDF_MAIZE_CACHE, driver="GeoJSON")
            print(f"Saved cached valid_maize to {GDF_MAIZE_CACHE}")

    # Normalise names
    df_annual["feature_id"] = df_annual["feature_id"].astype(str).str.strip().str.upper()
    gdf_maize["ADM_NAME"] = gdf_maize["ADM_NAME"].astype(str).str.strip().str.upper()

    return df_annual, gdf_maize


def prepare_features(df_annual, gdf_maize, spatial_weight=2.0, n_pcs=5, scale_temporal=True):
    """Prepare spatio-temporal feature matrix for clustering.

    Steps:
      - Pivot df_annual to district × year matrix
      - Standardise per-district time series (z-score across years)
      - Optional PCA reduction on temporal features
      - Append centroid lat/lon scaled by `spatial_weight`
    Returns: DataFrame indexed by ADM_NAME with feature columns
    """
    # Pivot
    piv = df_annual.pivot(index="feature_id", columns="year", values="Drought_Days").fillna(0)

    # Keep only districts present in gdf_maize
    common = sorted(set(piv.index).intersection(set(gdf_maize["ADM_NAME"])))
    piv = piv.loc[common].copy()

    # Standardise per-row (district) across years
    if scale_temporal:
        piv_z = piv.sub(piv.mean(axis=1), axis=0).div(piv.std(axis=1).replace(0, 1), axis=0).fillna(0)
    else:
        piv_z = piv.copy()

    # PCA
    pca = PCA(n_components=min(n_pcs, piv_z.shape[1], piv_z.shape[0]))
    temporal_pc = pca.fit_transform(piv_z.values)
    temporal_cols = [f"pc_{i+1}" for i in range(temporal_pc.shape[1])]
    df_temporal = pd.DataFrame(temporal_pc, index=piv_z.index, columns=temporal_cols)

    # Centroids
    g = gdf_maize.drop_duplicates("ADM_NAME").set_index("ADM_NAME")
    coords = g.loc[common, "geometry"]
    if hasattr(coords, "crs") and coords.crs is not None:
        coords = coords.to_crs(epsg=3857).centroid.to_crs(epsg=4326)
    else:
        coords = coords.centroid
    lat = coords.y.values
    lon = coords.x.values
    df_spatial = pd.DataFrame({"lat": lat, "lon": lon}, index=piv_z.index)

    # Scale spatial features to unit variance and apply spatial_weight
    scaler = StandardScaler()
    spatial_scaled = scaler.fit_transform(df_spatial)
    df_spatial_scaled = pd.DataFrame(spatial_scaled * spatial_weight, index=df_spatial.index, columns=["lat_s", "lon_s"])

    features = pd.concat([df_temporal, df_spatial_scaled], axis=1)
    return features, piv


def run_clusterers(features, pivot_ts, methods_cfg, output_prefix="run"):
    """Run multiple clustering algorithms and evaluate them.

    methods_cfg: list of dicts like {"name":"kmeans","params":{...}}
    Returns: DataFrame with metrics and writes cluster CSVs.
    """
    metrics = []

    for cfg in methods_cfg:
        name = cfg.get("name")
        params = cfg.get("params", {})
        label = cfg.get("label", name)

        try:
            if name == "kmeans":
                n_clusters = params.get("n_clusters", 6)
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = model.fit_predict(features.values)
                cluster_labels = [f"K_{c+1}" for c in clusters]

            elif name == "agglomerative":
                n_clusters = params.get("n_clusters", 6)
                model = AgglomerativeClustering(n_clusters=n_clusters)
                clusters = model.fit_predict(features.values)
                cluster_labels = [f"A_{c+1}" for c in clusters]

            elif name == "gmm":
                n_components = params.get("n_components", 6)
                model = GaussianMixture(n_components=n_components, random_state=42)
                clusters = model.fit_predict(features.values)
                cluster_labels = [f"G_{c+1}" for c in clusters]

            elif name == "dbscan":
                eps = params.get("eps", 0.5)
                min_samples = params.get("min_samples", 5)
                model = DBSCAN(eps=eps, min_samples=min_samples)
                clusters = model.fit_predict(features.values)
                cluster_labels = [f"DB_{c}" for c in clusters]

            elif name == "hdbscan" and HDBSCAN_AVAILABLE:
                min_cluster_size = params.get("min_cluster_size", 10)
                model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
                clusters = model.fit_predict(features.values)
                cluster_labels = [f"H_{c}" for c in clusters]

            else:
                print(f"Unknown or unavailable method: {name} — skipping")
                continue

        except Exception as e:
            print(f"Clustering {label} failed: {e}")
            continue

        # Map clusters to ADM_NAME
        df_map = pd.DataFrame({"ADM_NAME": features.index, "cluster": cluster_labels})
        out_csv = os.path.join(OUTPUT_DIR, f"clusters_{label}.csv")
        df_map.to_csv(out_csv, index=False)

        # Evaluation metrics
        n_unique = len(set(clusters))
        sil = None
        db = None
        ch = None
        try:
            if n_unique > 1 and -1 not in set(clusters):
                sil = silhouette_score(features.values, clusters)
                db = davies_bouldin_score(features.values, clusters)
                ch = calinski_harabasz_score(features.values, clusters)
        except Exception:
            pass

        # Temporal coherence: mean within-cluster Spearman rho
        mean_within_rho = compute_within_cluster_spearman(pivot_ts, df_map)

        # Spatial compactness: mean pairwise Haversine distance within clusters (km)
        mean_within_km = compute_within_cluster_mean_distance(df_map)

        metrics.append({
            "run": label,
            "method": name,
            "params": json.dumps(params),
            "n_clusters": n_unique,
            "silhouette": sil,
            "davies_bouldin": db,
            "calinski_harabasz": ch,
            "mean_within_spearman": mean_within_rho,
            "mean_within_km": mean_within_km,
            "output_csv": out_csv,
        })

        print(f"Saved clusters: {out_csv} — clusters: {n_unique} — silhouette: {sil}")

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "metrics_summary.csv"), index=False)
    return metrics_df


def compute_within_cluster_spearman(pivot_ts, df_map):
    """Compute mean pairwise Spearman correlation within clusters.
    pivot_ts: district × year matrix (index=ADM_NAME)
    df_map: ADM_NAME -> cluster label
    Returns mean of all within-cluster pair correlations (averaged across clusters)
    """
    merged = pivot_ts.copy()
    merged = merged.loc[merged.index.intersection(df_map["ADM_NAME"])]
    label_map = dict(zip(df_map["ADM_NAME"], df_map["cluster"]))
    clusters = {}
    for adm, lab in label_map.items():
        clusters.setdefault(lab, []).append(adm)

    rho_list = []
    for lab, members in clusters.items():
        if len(members) < 2:
            continue
        sub = merged.loc[members]
        # If necessary, fill NaNs with column mean
        sub = sub.fillna(sub.mean(axis=1))
        # pairwise spearman
        for a, b in combinations(members, 2):
            x = sub.loc[a].values
            y = sub.loc[b].values
            if np.allclose(x, x[0]) or np.allclose(y, y[0]):
                continue
            try:
                r, _ = spearmanr(x, y)
                if not np.isnan(r):
                    rho_list.append(r)
            except Exception:
                continue

    return float(np.mean(rho_list)) if rho_list else None


def haversine_km(lon1, lat1, lon2, lat2):
    # Haversine formula (km)
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def compute_within_cluster_mean_distance(df_map):
    """Compute mean pairwise centroid distance (km) within clusters.
    Requires reading district centroids from the original Week 11 script's
    shapefile paths via import; if unavailable returns None.
    """
    try:
        week11 = load_week11_pipeline_module()
        gadm_path = getattr(week11, "GADM_PATH")
        gdf = gpd.read_file(gadm_path)
        gdf["ADM_NAME"] = gdf["ADM_NAME"].astype(str).str.strip().str.upper()
        gdf = gdf.drop_duplicates(subset=["ADM_NAME"], keep="first").set_index("ADM_NAME")
        centroids = gdf.geometry
        if centroids.crs is not None:
            centroids = centroids.to_crs(epsg=3857).centroid.to_crs(epsg=4326)
        else:
            centroids = centroids.centroid
    except Exception:
        return None

    label_map = dict(zip(df_map["ADM_NAME"], df_map["cluster"]))
    clusters = {}
    for adm, lab in label_map.items():
        clusters.setdefault(lab, []).append(adm)

    dist_list = []
    for lab, members in clusters.items():
        if len(members) < 2:
            continue
        coords = centroids.reindex(members).dropna()
        if len(coords) < 2:
            continue
        lons = coords.x.values
        lats = coords.y.values
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                km = haversine_km(lons[i], lats[i], lons[j], lats[j])
                dist_list.append(km)

    return float(np.mean(dist_list)) if dist_list else None


def build_default_methods(k_list=(4,6,8,10)):
    methods = []
    for k in k_list:
        methods.append({"name": "kmeans", "label": f"kmeans_k{k}", "params": {"n_clusters": k}})
        methods.append({"name": "agglomerative", "label": f"agg_k{k}", "params": {"n_clusters": k}})
        methods.append({"name": "gmm", "label": f"gmm_k{k}", "params": {"n_components": k}})
    # DBSCAN variants
    methods.append({"name": "dbscan", "label": "dbscan_eps0.5", "params": {"eps": 0.5, "min_samples": 5}})
    methods.append({"name": "dbscan", "label": "dbscan_eps1.0", "params": {"eps": 1.0, "min_samples": 5}})
    if HDBSCAN_AVAILABLE:
        methods.append({"name": "hdbscan", "label": "hdbscan_10", "params": {"min_cluster_size": 10}})
    return methods


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare clustering methods on drought-day time series")
    parser.add_argument("--df-annual", help="CSV of df_annual (feature_id, year, Drought_Days)")
    parser.add_argument("--gdf-maize", help="GEOData file (valid_maize) with ADM_NAME geometry")
    parser.add_argument("--spatial-weight", type=float, default=2.0)
    parser.add_argument("--n-pcs", type=int, default=5)
    parser.add_argument("--ks", nargs="*", type=int, default=[4,6,8])
    args = parser.parse_args(argv)

    df_annual, gdf_maize = load_inputs(args.df_annual, args.gdf_maize)
    features, pivot_ts = prepare_features(df_annual, gdf_maize, spatial_weight=args.spatial_weight, n_pcs=args.n_pcs)

    methods = build_default_methods(k_list=tuple(args.ks))
    metrics = run_clusterers(features, pivot_ts, methods)

    print("\nFinished. Metrics saved to:", os.path.join(OUTPUT_DIR, "metrics_summary.csv"))
    print(metrics)


if __name__ == "__main__":
    main()
