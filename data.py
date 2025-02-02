import pandas as pd
import numpy as np

# Load the CSV file
file_path = "application_train.csv"  # Replace with your actual file path
df = pd.read_csv(file_path)

# Standardize column names before filtering
df.columns = df.columns.str.strip()
df.columns = df.columns.str.lower()
df.columns = df.columns.str.replace(' ', '_')
df.columns = df.columns.str.replace('[^a-z0-9_]', '', regex=True)

# Define important columns for credit risk modeling
important_columns = [
    "target", "amt_credit", "amt_annuity", "amt_income_total", 
    "days_birth", "days_employed", "days_id_publish", "days_registration", "ext_source_2",
    "cnt_children", "cnt_fam_members"
]

# Filter the dataset to keep only the important columns and drop missing values
df_filtered = df[important_columns].dropna()

# Generate a random CCF using a normal distribution (Mean=0.725, Std=0.075)
mean_CCF = 0.725
std_CCF = 0.075
df_filtered["CCF"] = np.random.normal(loc=mean_CCF, scale=std_CCF, size=len(df_filtered))

# Ensure CCF stays within 0.6 - 0.85 range
df_filtered["CCF"] = df_filtered["CCF"].clip(0.6, 0.85)

# Calculate Undrawn Amount (Assuming AMT_CREDIT is total loan and AMT_ANNUITY as paid amount)
df_filtered["Undrawn_Amount"] = df_filtered["amt_credit"] - df_filtered["amt_annuity"]
df_filtered["Undrawn_Amount"] = df_filtered["Undrawn_Amount"].clip(lower=0)  # Ensure no negative values

# Calculate EAD using the random CCF
df_filtered["EAD_With_Normal_CCF"] = df_filtered["amt_credit"] + (df_filtered["Undrawn_Amount"] * df_filtered["CCF"])

# Save the cleaned and processed dataset
df_filtered.to_csv("datasetNormalCCFEAD.csv", index=False)

print("Filtered dataset with EAD saved as datasetNormalCCFEAD.csv")
