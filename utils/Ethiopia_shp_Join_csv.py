#!/usr/bin/env python
import pandas as pd
import glob
import os

# 1. SETUP PATHS
# Using a raw string (r"") to handle Windows backslashes correctly
input_folder = r"C:/Users/FlawiyaShirishMore/OneDrive - Africa Specialty Risks Ltd/ASR-Parametric_Research_Study/africa_risk/Drought/Output/filtered_files/content/filtered_csvs/filtered_rainfall"
output_file = "ethiopia_rainfall_master_cleaned.csv"

# 2. GET FILE LIST
file_list = glob.glob(os.path.join(input_folder, "*.csv"))
print(f"📂 Found {len(file_list)} files to process.")

# 3. COMBINE DATASETS
# Instead of a complex loop, we read them into a list and concat once.
# Pandas automatically handles headers correctly if they are identical.
df_list = []
for file in file_list:
    try:
        df = pd.read_csv(file)
        df_list.append(df)
    except Exception as e:
        print(f"❌ Error reading {file}: {e}")

if not df_list:
    print("No data found. Exiting.")
    exit()

master_df = pd.concat(df_list, ignore_index=True)

# 4. DATA CLEANING
# Convert month to integer
master_df['month'] = master_df['month'].astype(int)

# 5. DATA QUALITY AUDIT
print("\n--- 📋 DATA QUALITY AUDIT ---")
print(f"Total Records: {len(master_df)}")

# Check for Nulls
null_counts = master_df.isnull().sum().sum()
print(f"Total Null Values: {null_counts}")

# Check for Duplicates (District + Year + Month)
dup_count = master_df.duplicated(subset=['ADM_NAME', 'year', 'month']).sum()
print(f"Duplicate records (District-Year-Month): {dup_count}")

# Check Range
print(f"Year Range: {master_df['year'].min()} - {master_df['year'].max()}")
print(f"Unique Months: {sorted(master_df['month'].unique())}")

# 6. VERIFY RECORDS PER YEAR
# In Ethiopia, if you have 38 districts, you should expect 38 * 12 = 456 records per year
print("\n--- 📊 RECORDS PER YEAR ---")
print(master_df['year'].value_counts().sort_index())

# 7. HANDLE DUPLICATES (Optional)
# If duplicates exist, you can drop them here:
if dup_count > 0:
    print(f"Removing {dup_count} duplicate records...")
    master_df = master_df.drop_duplicates(subset=['ADM_NAME', 'year', 'month'], keep='first')

# 8. SAVE THE CLEANED MASTER FILE
master_df.to_csv(output_file, index=False)
print(f"\n✅ Success! Master file saved as: {os.path.abspath(output_file)}")