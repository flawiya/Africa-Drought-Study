import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def perform_spatio_temporal_clustering(df_annual, valid_maize, n_clusters=10, spatial_weight=2.0):
    """
    Step 14: Cluster districts based on both Drought Trajectory AND Location.
    
    Args:
        df_annual: The annual drought day counts from Step 7.
        valid_maize: The GeoDataFrame containing district centroids.
        n_clusters: How many groups (G1, G2, etc.) we want.
        spatial_weight: How much to force districts to be near each other. 
                        Higher = more compact regions. Lower = more scattered by weather.
    """
    print("\n" + "=" * 70)
    print(f"STEP 14: Clustering districts into {n_clusters} groups")
    print("=" * 70)

    # 1. Prepare Temporal Data: Pivot so years are columns
    # Rows: Districts | Columns: 2000, 2001, ..., 2025
    temporal_df = df_annual.pivot(index='feature_id', columns='year', values='Drought_Days').fillna(0)

    # 2. Prepare Spatial Data: Get Centroids
    # We need Lat/Long for every district in the temporal_df
    geo_data = valid_maize[['ADM_NAME', 'geometry']].copy()
    geo_data['lat'] = geo_data.geometry.centroid.y
    geo_data['lon'] = geo_data.geometry.centroid.x
    geo_coords = geo_data.set_index('ADM_NAME')[['lat', 'lon']]

    # 3. Merge Temporal and Spatial
    cluster_input = temporal_df.join(geo_coords, how='inner')
    district_names = cluster_input.index

    # 4. Scale the data (StandardScaler)
    # This is vital because 'degrees' and 'drought days' are different units
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_input)
    scaled_df = pd.DataFrame(scaled_data, columns=cluster_input.columns)

    # 5. Apply Spatial Weighting (Massimo's specific request)
    # We multiply the Lat/Lon columns to make them more "important" to the K-Means algorithm
    scaled_df['lat'] *= spatial_weight
    scaled_df['lon'] *= spatial_weight

    # 6. Run K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_df)

    # 7. Create results dataframe
    results = pd.DataFrame({
        'feature_id': district_names,
        'cluster_group': [f"Group {c+1}" for c in clusters]
    })

    print(f"Created {n_clusters} clusters across {len(results)} districts.")
    return results