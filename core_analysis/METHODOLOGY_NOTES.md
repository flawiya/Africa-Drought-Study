# Spatial Regionalization & Centroid Methodology Notes

## 📌 Centroid vs. Representative Point Comparison

In spatial agglomerative clustering for Pan-African risk regionalization, geographic distance matrices require a 2D spatial coordinate for each administrative district. Two methods exist in GeoPandas / Shapely:

---

### 1. `2 spatial_agglomerative_clustering.py` (Baseline Script)
* **Spatial Feature**: `gdf.geometry.centroid` (Center of Mass in EPSG:3857)
* **Behavior**: Calculates the mean $(X, Y)$ coordinate of all polygon vertices.
* **Limitation**: For concave/horseshoe-shaped districts (crescent moon shapes around rivers/mountains) or multi-island archipelagos, the center of mass can fall **outside** the polygon boundary (in neighboring water, coastal bay, or adjacent territory). In our Pan-African dataset of 3,333 districts, 74 districts have centroids falling outside their boundaries.

---

### 2. `2_spatial_agglomerative_clustering_representative_points.py` (Inside-Boundary Compliant)
* **Spatial Feature**: `gdf.geometry.representative_point()` (GIS Gold Standard)
* **Behavior**: Guarantees a point that is **100% strictly inside the interior** of every district boundary polygon, regardless of concave or complex island geometry.
* **Impact on Regional Clustering**: For macro-regional spatial distance matrices (distances measured in 100s of km across Africa), the shift between `.centroid` and `.representative_point()` is very small (typically < 2–5 km shift per district), so Silhouette scores and major regional cluster boundaries remain 98–99% identical. However, `.representative_point()` eliminates all edge cases for spatial point-in-polygon joins.
