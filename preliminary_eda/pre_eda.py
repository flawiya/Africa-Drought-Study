"""
DISSERTATION: A Spatio-Temporal Reconstruction of Extreme Weather Events
PHASE: Data Processing, Hazard Identification, and Financial Calibration
AUTHOR: [Your Name]
DATE: May 2026
"""
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import json
import warnings
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, classification_report

warnings.filterwarnings('ignore')
plt.style.use('ggplot')

# =============================================================================
# 1. DATA LOADING & HARMONIZATION (Aim 1)
# =============================================================================
def load_and_clean_data(rainfall_path, shp_path, emdat_path):
    print("📂 Loading and Harmonizing Datasets...")
    
    # Load Main Rainfall Data (40-year GEE Extraction)
    df = pd.read_csv(rainfall_path)
    df.columns = df.columns.str.lower()
    
    # Load Geospatial Boundary Files
    districts_shp = gpd.read_file(shp_path)
    
    # Load Ground Truth (EM-DAT Disasters)
    emdat = pd.read_csv(emdat_path)
    emdat.columns = emdat.columns.str.lower()

    # Standardization: Fix join failures ("White Holes")
    df['adm_name'] = df['adm_name'].astype(str).str.upper().str.strip()
    df['iso3'] = df['iso3'].astype(str).str.upper().str.strip()
    districts_shp['ADM_NAME'] = districts_shp['ADM_NAME'].astype(str).str.upper().str.strip()
    districts_shp['ISO3'] = districts_shp['ISO3'].astype(str).str.upper().str.strip()

    # Numeric conversion & Cleaning
    df['mean'] = pd.to_numeric(df['mean'], errors='coerce')
    df = df.dropna(subset=['mean', 'adm_name'])
    
    return df, districts_shp, emdat

# =============================================================================
# 2. METEOROLOGICAL METRIC DEVELOPMENT (Aim 2)
# =============================================================================
def calculate_drought_indices(df):
    print("📉 Calculating WMO Percent of Normal (PNI) and SPI-3...")
    
    # 2.1 PNI (Localized 40-year baseline)
    df['ltm'] = df.groupby(['iso3', 'adm_name', 'month'])['mean'].transform('mean')
    df['pni'] = (df['mean'] / df['ltm']) * 100

    # 2.2 SPI-3 (3-Month Rolling Standardized Index - Industry Standard)
    df = df.sort_values(['iso3', 'adm_name', 'year', 'month'])
    df['rain_3m'] = df.groupby(['iso3', 'adm_name'])['mean'].transform(lambda x: x.rolling(window=3).sum())
    
    def z_score(group):
        mu, sigma = group.mean(), group.std()
        return (group - mu) / (sigma if sigma > 0 else 1)

    df['spi_3'] = df.groupby(['iso3', 'adm_name', 'month'])['rain_3m'].transform(z_score)
    
    # Drought Severity Classification (WMO Standards)
    df['severity'] = pd.cut(df['pni'], bins=[0, 55, 70, 80, 200], labels=[3, 2, 1, 0])
    
    return df

# =============================================================================
# 3. PARAMETRIC UNDERWRITING CALIBRATION (Financial Logic)
# =============================================================================
def apply_parametric_logic(df, strike=70, exit=50):
    """
    Strike: Payout starts (70% of normal rainfall)
    Exit: 100% Payout reached (50% of normal rainfall)
    """
    print(f"💰 Calibrating Payouts: Strike {strike}%, Exit {exit}%...")
    
    def calculate_payout(pni):
        if pni >= strike: return 0.0
        if pni <= exit: return 1.0
        return (strike - pni) / (strike - exit)

    df['proxy_loss_ratio'] = df['pni'].apply(calculate_payout)
    return df

# =============================================================================
# 4. EXPLORATORY DATA ANALYSIS (Continental Pulse)
# =============================================================================
def generate_eda_plots(df):
    print("📊 Generating Visual A: Systemic Risk Reconstruction...")
    # Stacked Area Plot of Drought Frequency
    area_data = df.groupby(['year', 'severity']).size().unstack(fill_value=0)
    area_data.columns = ['Normal', 'Moderate', 'Severe', 'Extreme']
    
    plt.figure(figsize=(12, 6))
    area_data.plot(kind='area', stacked=True, colormap='YlOrRd', alpha=0.8, ax=plt.gca())
    plt.title("40-Year African Drought Stress Reconstruction")
    plt.savefig("Visual_A_Drought_Stack.png")
    
    print("🔗 Generating Visual B: Portfolio Diversification...")
    # Inter-Country Correlation
    top_economies = ['NGA', 'KEN', 'ZAF', 'ETH', 'EGY', 'GHA', 'ZMB']
    country_annual = df.groupby(['year', 'iso3'])['pni'].mean().reset_index()
    pivot_corr = country_annual.pivot(index='year', columns='iso3', values='pni')
    corr_matrix = pivot_corr[top_economies].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Inter-Country Drought Correlation (Risk Diversification)")
    plt.savefig("Visual_B_Correlation.png")

# =============================================================================
# 5. VALIDATION: EM-DAT GROUND TRUTH & ML (Aim 3)
# =============================================================================
def validate_and_train_ml(df, emdat):
    print("🤖 Running Validation and Machine Learning...")
    
    # Prepare Ground Truth binary target
    emdat_events = emdat[emdat['disaster subtype'] == 'Drought'].groupby(['iso', 'start year']).size().reset_index(name='disaster')
    emdat_events['disaster'] = emdat_events['disaster'].clip(upper=1)
    emdat_events = emdat_events.rename(columns={'iso': 'iso3', 'start year': 'year'})
    
    # Feature Engineering for ML
    annual_model = df.groupby(['iso3', 'year']).agg({
        'spi_3': 'min',
        'pni': 'mean',
        'mean': 'cv' # Coefficient of Variation
    }).reset_index()
    
    eval_df = pd.merge(annual_model, emdat_events, on=['iso3', 'year'], how='left').fillna(0)
    
    # Train Random Forest
    features = ['spi_3', 'pni', 'mean']
    X = eval_df[features].fillna(0)
    y = eval_df['disaster'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    rf = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
    y_probs = rf.predict_proba(X_test)[:, 1]
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    print(f"✅ ML Validation Complete. AUC Score: {auc(fpr, tpr):.2f}")
    
    return rf

# =============================================================================
# 6. GEOSPATIAL ANIMATION (The Showstopper)
# =============================================================================
def create_pulse_map(df, districts_shp):
    print("🗺️ Rendering Interactive HTML Pulse Map...")
    districts_shp['geometry'] = districts_shp.simplify(0.02)
    africa_json = json.loads(districts_shp.to_json())
    
    df_yearly = df.groupby(['year', 'adm_name'])['pni'].mean().reset_index()
    
    fig = px.choropleth(
        df_yearly, geojson=africa_json, locations='adm_name',
        featureidkey="properties.ADM_NAME", color='pni',
        animation_frame='year', color_continuous_scale="RdYlGn",
        range_color=[40, 120], title="Africa's Drought Pulse (1981-2024)"
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.write_html("Drought_Pulse_Reconstruction.html")

# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Update paths to your local drive
    RAINFALL_FILE = 'master_rainfall_africa.csv'
    SHP_FILE = 'africa_admx_shp/africa_admx.shp'
    EMDAT_FILE = 'data/drought.csv'
    
    try:
        # Run Pipeline
        df_master, shp, emdat_raw = load_and_clean_data(RAINFALL_FILE, SHP_FILE, EMDAT_FILE)
        df_master = calculate_drought_indices(df_master)
        df_master = apply_parametric_logic(df_master)
        
        generate_eda_plots(df_master)
        validate_and_train_ml(df_master, emdat_raw)
        create_pulse_map(df_master, shp)
        
        print("\n✨ Dissertation Analysis Complete. Visuals saved to directory.")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find files. Check paths. {e}")