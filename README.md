# Africa Parametric Drought Risk Study

> **Module**: `Africa-Drought-Study`  
> **Project**: ASR Multi-Peril Parametric Insurance Research Study  
> **Target Region**: All 54 African Countries (1,600+ Agricultural Districts)  
> **Primary Peril**: Agricultural Drought Stress (Root-Zone Soil Moisture Deficit)  
> **Status**: Production & Handover Ready  

---

## 📌 1. Overview & Objective

The **Africa Parametric Drought Risk Study** delivers an early-warning, physics-informed parametric drought index insurance pricing and trigger framework for rainfed agricultural regions across Africa.

### Key Methodological Pillars:
1. **ECMWF ERA5-Land Layer 2 Root-Zone Soil Moisture ($7 - 28\text{ cm}$)**: Directly captures the root-zone moisture deficit 2–4 weeks earlier than optical satellite vegetation indices (NDVI/EVI).
2. **Non-Parametric Gringorten Empirical SSI Formulation**: Avoids rigid parametric distribution assumptions, providing unbiased drought probability estimates in semi-arid zones.
3. **Dynamic Crop Calendar Synchronization**: Links drought evaluation strictly to active planting-to-harvest windows without cross-year December 31st calendar truncation.
4. **ENSO Teleconnection Recurrence Modeling**: Quantifies El Niño / La Niña cyclical drought vulnerability across Southern and East Africa.

---

## 📂 2. Repository Directory Structure

```
Africa-Drought-Study/
├── core_analysis/                                # Production Python analysis engines
│   ├── 1 Africa_agricultural_districts_correlation_crop_calendar.py # Primary Extraction
│   ├── 2 spatial_agglomerative_clustering.py     # Spatial risk pooling engine
│   ├── 3 ENSO_Correlation_Analysis.py            # ENSO climate driver correlation
│   ├── Process_Africa_Maize_Drought_History_2000_2025.py # 25-Year historical baseline
│   ├── Merge_Africa_Maize_Drought_Annual_Days_to_GADM.py # GADM admin boundary merger
│   └── Africa_districts_with_different_SSI_threshold_variation.py # Payout trigger sensitivity
├── data/                                         # District shapefiles & crop calendars
├── docs/                                         # Technical documentation & guides
│   ├── DROUGHT_PARAMETRIC_METHODOLOGY_GUIDE.md   # Formal Technical Guide
│   └── METHODOLOGY_NOTES.md                      # Core analysis methodology notes
├── outputs/                                      # Dashboards & analysis CSVs
├── utils/                                        # Geospatial extraction utilities
└── README.md                                     # This Execution & Handover Guide
```

---

## 🚀 3. Pipeline Execution Sequence

```bash
# 1. Extract ERA5-Land Layer 2 Soil Water & Compute Daily SSI across 1,600+ districts
python "core_analysis/1 Africa_agricultural_districts_correlation_crop_calendar.py"

# 2. Process 25-Year Historical Drought Time Series (2000–2025)
python core_analysis/Process_Africa_Maize_Drought_History_2000_2025.py

# 3. Perform Spatial Agglomerative Clustering for Regional Risk Pooling
python "core_analysis/2 spatial_agglomerative_clustering.py"

# 4. Run ENSO Teleconnection Analysis (Niño 3.4 SST Anomaly Correlation)
python "core_analysis/3 ENSO_Correlation_Analysis.py"

# 5. Evaluate Multi-Threshold SSI Payout Curves (Moderate, Severe, Extreme)
python core_analysis/Africa_districts_with_different_SSI_threshold_variation.py
```

---

## 📖 4. Reference Documentation

- 📘 [DROUGHT_PARAMETRIC_METHODOLOGY_GUIDE.md](file:///c:/Users/FlawiyaShirishMore/OneDrive%20-%20Africa%20Specialty%20Risks%20Ltd/ASR-Parametric_Research_Study/Africa-Drought-Study/docs/DROUGHT_PARAMETRIC_METHODOLOGY_GUIDE.md)
