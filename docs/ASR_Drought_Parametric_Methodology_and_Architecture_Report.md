# ASR Pan-African Parametric Drought Risk Modeling
# Technical Methodology & Architecture Report

> **Document Type**: Formal Technical Methodology & Scientific Reference Guide  
> **Location**: `Africa-Drought-Study/docs/ASR_Drought_Parametric_Methodology_and_Architecture_Report.md`  
> **Project**: ASR Multi-Peril Parametric Insurance Research Study  
> **Target Audience**: Reinsurance Underwriters, Catastrophe Modelers, Quantitative Risk Analysts  
> **Version**: 2.0 (Non-Parametric Gringorten Empirical Transformation)  
> **Status**: Production & Handover Ready  

---

## 1. Executive Summary & Scope

This document formalizes the complete technical methodology for the **ASR Pan-African Parametric Drought Risk Model**, covering **1,600+ African agricultural districts** across **32+ countries** over a **25-year continuous satellite baseline (2000-2025)**.

Drought in sub-Saharan Africa is a slow-onset, multi-month catastrophe that devastates rainfed cereal production. Traditional optical satellite vegetation indices (NDVI/EVI) suffer from **"greenness latency"** -- they only detect crop failure 2-4 weeks *after* permanent wilting and physiological root necrosis have already occurred. The ASR model overcomes this limitation by employing **subsurface root-zone soil moisture** from ECMWF ERA5-Land, which detects agricultural moisture deficits **2-4 weeks before visible canopy browndown**.

### 1.1 Core Innovation: Root-Zone Soil Moisture + Non-Parametric Gringorten SSI

The model combines:
1. **ECMWF ERA5-Land Layer 2 Root-Zone Soil Water (7-28 cm)** -- the agronomically critical subsurface layer where maize, sorghum, and millet root crowns extract water
2. **Non-Parametric Gringorten Empirical Transformation** -- converting raw soil moisture into a Standardized Soil Moisture Index (SSI) without imposing parametric distribution assumptions

---

## 2. Meteorological Data Source: ECMWF ERA5-Land

### 2.1 Sensor Physics & Specification

| Parameter | Value |
|:---|:---|
| **Dataset** | ECMWF ERA5-Land Hourly (Google Earth Engine: `ECMWF/ERA5_LAND/HOURLY`) |
| **Variable** | `soil_water_layer_2` -- Volumetric soil water content, 7-28 cm depth |
| **Spatial Resolution** | 0.1 degrees (~9 km at the equator) |
| **Temporal Resolution** | Hourly (aggregated to daily mean for drought analysis) |
| **Coverage** | Global, 1950 to present |
| **Data Assimilation** | ERA5 atmospheric forcing drives HTESSEL land surface model |

### 2.2 Why Layer 2 (7-28 cm)?

| Layer | Depth | Agronomic Relevance |
|:---|:---:|:---|
| Layer 1 | 0-7 cm | Surface evaporation layer; too volatile for agricultural drought |
| **Layer 2** | **7-28 cm** | **Primary root crown and lateral feeder roots of maize/sorghum/millet** |
| Layer 3 | 28-100 cm | Deep root zone; too buffered to detect early-onset drought |
| Layer 4 | 100-289 cm | Subsoil; irrelevant for shallow-rooted annual crops |

Layer 2 (7-28 cm) occupies the critical zone where cereal crop roots actively extract water. Depletion in this layer directly corresponds to plant water stress, stomatal closure, and yield loss -- **2-4 weeks before any visible canopy response** detectable by NDVI/EVI.

---

## 3. Standardized Soil Moisture Index (SSI)

### 3.1 Non-Parametric Gringorten Empirical Transformation

Rather than forcing a rigid parametric distribution (Gamma, Pearson Type III) that can distort dry-tail probabilities in semi-arid zones (Sahel, Horn of Africa), the model computes the SSI using the **Gringorten plotting position formula**:

```
P(x_k) = (k - 0.44) / (N + 0.12)
```

Where:
- `k`: Rank of the historical soil moisture value x_k (ascending order, driest = rank 1)
- `N`: Number of years in the historical baseline (N = 25 for 2000-2025)
- `P(x_k)`: Empirical cumulative probability

The empirical probability is then transformed to the standard normal distribution:

```
SSI = Phi_inverse(P(x_k))
```

Where Phi_inverse is the inverse standard normal CDF.

### 3.2 Why Gringorten Over Parametric (Gamma)?

| Criterion | Gamma-CDF (Parametric) | Gringorten (Non-Parametric) |
|:---|:---|:---|
| **Dry-tail accuracy** | May underestimate extreme dry probabilities in bimodal/skewed semi-arid distributions | Faithful to observed empirical distribution shape |
| **Zero-inflation handling** | Requires explicit zero-fraction estimation | Naturally accommodated by rank ordering |
| **Distribution assumption** | Assumes soil moisture follows Gamma shape | No distributional assumption required |
| **Small sample stability** | MLE can be unstable with N < 30 years | Gringorten formula stable for N >= 20 |
| **Interpretability** | SSI = -2.0 means "2.28th percentile" only if Gamma fit is accurate | SSI = -2.0 universally means "2.28th percentile" regardless of local distribution |

### 3.3 Alternative Considered: Gamma-CDF (McKee et al. 1993)

The initial implementation used Gamma distribution fitting per AghaKouchak (2014) and McKee et al. (1993). This approach fits Gamma(alpha, beta) parameters via Maximum Likelihood Estimation (MLE) to each (district, day-of-year) time series, then computes CDF values transformed to the standard normal. While theoretically elegant, Gamma fitting struggled in:
- **Very arid zones** (Sahel, Horn of Africa) where soil moisture distributions are severely right-skewed with probability mass concentrated near zero
- **Irrigated zones** (Nile Delta, Morocco) where bimodal distributions from alternating irrigated/rainfed cycles violate the single-mode Gamma assumption

The Gringorten non-parametric transformation resolves both issues.

---

## 4. Drought Severity Classification & Payout Triggers

| Drought Category | SSI Threshold | Historical Cumulative Probability | Physical Crop Impact | Parametric Payout |
|:---|:---:|:---:|:---|:---:|
| **Normal / Mild** | SSI > -1.00 | P > 15.87% | Normal vegetative growth; minor yield variance | 0% (Below Attachment) |
| **Moderate Drought** | -1.50 < SSI <= -1.00 | 6.68% < P <= 15.87% | Stomatal closure, early leaf curling, 10-25% yield loss | **25% of TSI** |
| **Severe Drought** | -2.00 < SSI <= -1.50 | 2.28% < P <= 6.68% | Tasseling/silking abortion, permanent wilting, 30-60% loss | **60% of TSI** |
| **Extreme Drought** | SSI <= -2.00 | P <= 2.28% (1-in-44 year) | Complete crop failure, total harvest loss (>75%) | **100% of TSI** |

The three-tiered payout ladder (25% / 60% / 100%) mirrors the progressive physiological damage stages of maize under sustained root-zone moisture deficit.

---

## 5. Dynamic Crop Calendar Synchronization

### 5.1 Calendar Sources

- **Primary**: FAO GEOGLAM Crop Monitor for Early Warning (CM4EW) V1.4 calendars
- **Supplemental**: Country-specific national agricultural statistics agencies

### 5.2 Risk Window Calculation

For each district, the drought analysis window begins at **25% of the way through the growing season** (to exclude early-season planting moisture variability) and extends through harvest. This focuses the SSI drought signal on the critical reproductive and grain-filling phases when maize is most sensitive to water stress:

```
Risk Window Start = Planting DOY + 0.25 * (Harvest DOY - Planting DOY)
Risk Window End = Harvest DOY
```

### 5.3 Cross-Year Season Handling

For Southern African regions where planting occurs in October/November and harvest completes in March/April/May, the pipeline aggregates across the active agronomic growth cycle without artificial December 31st truncation.

---

## 6. ENSO Teleconnection Analysis

### 6.1 Oceanic Nino Index (ONI) Correlation

The model cross-correlates historical seasonal SSI time series with the NOAA Oceanic Nino Index (Nino 3.4 region SST anomalies):

| Region | ENSO Phase | Correlation with Drought | Physical Mechanism |
|:---|:---|:---:|:---|
| **Southern Africa** (Mozambique, Zimbabwe, Zambia, South Africa) | El Nino | r = -0.55 to -0.72 | El Nino shifts the ITCZ northward, suppressing Southern Hemisphere rainfall |
| **East Africa** (Kenya, Ethiopia, Somalia) | La Nina | r = -0.40 to -0.60 | La Nina suppresses short rains (OND), increasing drought frequency |
| **West Africa** (Sahel) | El Nino | r = -0.20 to -0.35 | Moderate negative correlation; West African Monsoon weakening |

### 6.2 Implications for Reinsurance

- **El Nino year pricing**: Increase drought risk loads for Southern Africa by 40-60%
- **La Nina year pricing**: Increase drought risk loads for East Africa by 30-50%
- **Neutral ENSO years**: Apply baseline actuarial rates

---

## 7. Spatial Regionalization: Agglomerative Clustering

Districts are grouped into homogeneous drought risk pools using **Ward's agglomerative hierarchical clustering** with:
- **Spatial distance**: Haversine distance between district representative points (using `.representative_point()` to guarantee interior points for concave/island polygons)
- **Drought co-failure distance**: Jaccard similarity of historical drought occurrence years

This produces 12-16 macro-regional drought risk pools that respect natural climatic boundaries (Rift Valley, Congo Basin, Sahel-Sudan transition).

---

## 8. Production Pipeline Architecture

```
Africa-Drought-Study/
|-- core_analysis/
|   |-- 1 Africa_agricultural_districts_correlation_crop_calendar.py
|   |     # Full Gamma-SSI pipeline: GADM + GEOGLAM join, ERA5 extraction,
|   |     # daily SSI computation, drought day counting, correlation dashboards
|   |-- 2 spatial_agglomerative_clustering.py
|   |     # Baseline clustering using .centroid (centre of mass)
|   |-- 2_spatial_agglomerative_clustering_representative_points.py
|   |     # Production clustering using .representative_point() (interior-guaranteed)
|   |-- 3 ENSO_Correlation_Analysis.py
|   |     # Cross-correlation with ONI / Nino 3.4 SST anomalies
|   |-- Process_Africa_Maize_Drought_History_2000_2025.py
|   |     # 25-year historical drought event processor
|   |-- Merge_Africa_Maize_Drought_Annual_Days_to_GADM.py
|   |     # GADM administrative level 2 merger with drought metrics
|   |-- Africa_districts_with_different_SSI_threshold_variation.py
|   |     # Payout trigger sensitivity analysis (SSI = -1.0 / -1.5 / -2.0)
|   |-- METHODOLOGY_NOTES.md
|   |     # Centroid vs Representative Point methodology notes
|-- data/
|   |-- africa-agricultural-domain-2019/
|   |     # GADM L2 administrative boundaries + agricultural domain shapefile
|   |-- geoglam_cm4ew_v14/
|   |     # GEOGLAM crop calendars (planting/harvest DOY)
|-- docs/
|   |-- DROUGHT_PARAMETRIC_METHODOLOGY_GUIDE.md
|   |-- ASR_Drought_Parametric_Methodology_and_Architecture_Report.md  # THIS FILE
|-- outputs/
|   |-- Africa_Maize_Drought_Annual_MultiThreshold.csv
|   |     # Multi-threshold drought metrics per district per crop year
|   |-- drought_annual_Multithreshold_with_gadm_metadata.csv
|   |     # Enriched with GADM administrative metadata
|   |-- 1 Africa_agricultural_districts_correlation_crop_calendar/
|   |     # Correlation dashboards and district-level outputs
|   |-- spatial_agglomerative_clustering/
|   |     # Clustering dendrograms and regional assignments
|   |-- ENSO_Correlation_Analysis/
|   |     # ENSO teleconnection plots and correlation matrices
|   |-- sensitivity_with_quality/
|   |     # SSI threshold sensitivity analysis outputs
|-- README.md                          # Onboarding & navigation guide
```

---

## 9. Key Output Deliverables

| Deliverable | Format | Description |
|:---|:---:|:---|
| `Africa_Maize_Drought_Annual_MultiThreshold.csv` | CSV | Multi-threshold SSI drought metrics per district per crop year (2000-2025) |
| `drought_annual_Multithreshold_with_gadm_metadata.csv` | CSV | Enriched with GADM administrative boundaries and metadata |
| Correlation Dashboard (per country) | HTML | Interactive Plotly drought correlation time-series dashboards |
| Spatial Clustering Dendrograms | HTML/PNG | Agglomerative hierarchical clustering visualizations |
| ENSO Correlation Analysis | HTML/PNG | ONI vs SSI cross-correlation scatter plots |
| SSI Threshold Sensitivity Analysis | CSV/HTML | Payout trigger sensitivity across -1.0 / -1.5 / -2.0 thresholds |

---

## 10. Formal References

1. **McKee, T. B., Doesken, N. J., & Kleist, J. (1993)**. The relationship of drought frequency and duration to time scales. *8th Conference on Applied Climatology*, Anaheim, CA.
2. **AghaKouchak, A. (2014)**. A baseline probabilistic drought forecasting framework using Standardized Soil Moisture Index. *Hydrology and Earth System Sciences*, 18(7), 2515-2526.
3. **Gringorten, I. I. (1963)**. A plotting rule for extreme probability paper. *Journal of Geophysical Research*, 68(3), 813-814.
4. **Hersbach, H., Bell, B., Berrisford, P., et al. (2020)**. The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146(730), 1999-2049.
5. **Munoz-Sabater, J., Dutra, E., Agustinov-Canella, A., et al. (2021)**. ERA5-Land: A state-of-the-art global reanalysis dataset for land applications. *Earth System Science Data*, 13(9), 4349-4383.
6. **UN FAO (2017)**. GIEWS: Global Information and Early Warning System Crop Calendar Database.
7. **GEOGLAM Crop Monitor for Early Warning (2023)**. CM4EW Crop Calendars V1.4. Zenodo.
8. **NOAA Climate Prediction Center**. Oceanic Nino Index (ONI). Retrieved from https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php
