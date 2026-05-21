# African Agricultural Drought Study: SSI-Based Parametric Triggers
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## 🌍 Project Overview
This research focuses on the **Standardized Soil Moisture Index (SSI)** as a primary spatio-temporal trigger for agricultural drought insurance in Africa. 

## 🔬 Why SSI? (The Early Warning Advantage)
While traditional indices like NDVI (Vegetation) measure the *result* of drought, SSI measures the *physical supply* of water in the root zone. 
- **Early Trigger Capability**: SSI identifies moisture stress 2-4 weeks before biological signals (NDVI) appear.
- **Root Zone Focus**: We utilize **ERA5-Land Layer 2 (7-28cm)** soil moisture, which directly correlates with the survival of major African crops like Maize and Teff during the grain-filling stage.
- **Reduced Basis Risk**: By standardizing daily soil moisture against a 25-year Julian Day baseline, we capture "Flash Droughts" that monthly indices often miss.

## 📂 Repository Structure
- `core_analysis/`: Primary implementation of SSI calculation using ERA5-Land reanalysis data.
- `data_acquisition/`: GEE scripts to extract volumetric soil water and climate variables.
- `utils/`: Spatial-join tools to align GADM district boundaries with gridded soil data.
- `reports/`: Case studies for Ethiopia and Zambia Southern Province validating SSI against historical disaster records.

---
*Developed for Africa Specialty Risks Ltd.*
