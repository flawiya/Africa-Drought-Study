$dirs = @(
    'outputs/ENSO_Correlation_Analysis',
    'outputs/Ethiopia_SPI_SPEI_VHI_SSI_NDVI',
    'outputs/ET_SPI_in_Agricultural_Reanalysis',
    'outputs/Week_10',
    'outputs/Week_11_correlation_map',
    'outputs/Week_3_Report_Ethiopia',
    'outputs/Week_4_Report_Zambia_South',
    'outputs/Week_5_Report_Zambia_Southern_Province',
    'outputs/Week_5_Southern_province_new',
    'outputs/Week_6_Zambia_SSI_Data',
    'outputs/Week_7_Scaling',
    'outputs/Week_8_Scaling',
    'outputs/Week_9',
    'outputs/clustering_experiment',
    'outputs/cluster_comparison',
    'outputs/spatial_agglomerative_clustering',
    'outputs/Spatio_Temporal_K_means'
)

cd 'C:\Users\FlawiyaShirishMore\Downloads\Africa-Drought-Study'

foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
    Write-Host "Created: $dir"
}

Write-Host "`nAll directories created successfully!"
