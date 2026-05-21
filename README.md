# African Drought Analysis & Parametric Insurance Study
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## 🌍 Project Overview
This repository contains a comprehensive spatio-temporal research pipeline designed to identify agricultural drought triggers across Africa. The core of this research is the **PARI (Platinum Agricultural Reanalysis Index)**, a composite index that integrates atmospheric demand, soil moisture supply, and thermal stress to model parametric insurance payouts.

## 🔬 Methodology: The PARI Index
Unlike traditional indices, PARI uses a multi-factor approach to reduce Basis Risk:
*   **Atmospheric (20%)**: SPEI-3 (Standardized Precipitation Evapotranspiration Index).
*   **Soil Supply (50%)**: Root-zone moisture (7-28cm) from ERA5-Land.
*   **Thermal Stress (30%)**: LST (Land Surface Temperature) anomalies.

## 📂 Repository Structure
- `data_acquisition/`: Automated scripts for Google Earth Engine (CHIRPS, MODIS, ERA5).
- `core_analysis/`: Primary implementation of SSI, SPI, and SPEI indices.
- `utils/`: Data cleaning, CSV merging, and spatial-join utilities.
- `preliminary_eda/`: Exploratory data analysis and methodology drafting.

## 🚀 Getting Started
1. **Clone the repo:** `git clone https://github.com/flawiya/africa-drought-study.git`
2. **Install Dependencies:** `pip install -r requirements.txt`
3. **Data Access:** Note: Raw data files are excluded via `.gitignore` for privacy. Users must provide their own GEE credentials.

---
*Developed for Africa Specialty Risks Ltd.*
