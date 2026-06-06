# African Agricultural Drought Study: SSI-Based Parametric Triggers
Project focus: SSI Triggers, Satellite Data, Parametric Insurance.

![alt text](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![alt text](https://img.shields.io/badge/Google%20Earth%20Engine-4285F4?style=for-the-badge&logo=google-earth&logoColor=white) ![alt text](https://img.shields.io/badge/ERA5--Land-ECMWF-blue?style=for-the-badge) ![alt text](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![alt text](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=white) ![alt text](https://img.shields.io/badge/Geopandas-44A833?style=for-the-badge&logo=pandas&logoColor=white)

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

---
*Developed for Africa Specialty Risks Ltd.*
