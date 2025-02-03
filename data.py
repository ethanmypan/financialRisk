import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler



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
utilizationFactorMean = 0.725
std_UF = 0.075
df_filtered["Utilization_Factor"] = np.random.normal(loc=utilizationFactorMean, scale=std_UF, size=len(df_filtered))

# Ensure CCF stays within 0.6 - 0.85 range
df_filtered["Utilization_Factor"] = df_filtered["Utilization_Factor"].clip(0.6, 0.85)

# Calculate Undrawn Amount (Assuming AMT_CREDIT is total loan and AMT_ANNUITY as paid amount)
df_filtered["Undrawn_Amount"] = df_filtered["amt_credit"] - df_filtered["amt_annuity"]
df_filtered["Undrawn_Amount"] = df_filtered["Undrawn_Amount"].clip(lower=0)  # Ensure no negative values

# Calculate EAD using the random CCF
df_filtered["EAD_With_Normal_UF"] = df_filtered["amt_credit"] + (df_filtered["Undrawn_Amount"] * df_filtered["Utilization_Factor"])
df_filtered["EAD_With_Normal_UF"] = df_filtered["EAD_With_Normal_UF"].round(2)
# calculating recovery rate = tot amt paid/tot balance *100

df_filtered["Recovery_Rate"] = (df_filtered["amt_annuity"] / df_filtered["amt_credit"])

# caclulate loss given defaule
df_filtered["Loss_Given_Default"] = (1 - df_filtered["Recovery_Rate"])



#calc age
df_filtered["Age"] = (abs(df_filtered["days_birth"]) / 365).round(2)


#we need to scale columns
# cols_to_scale = ['amt_credit', 'amt_income_total', 'amt_annuity', 
#                      'Undrawn_Amount', 'EAD_With_Normal_UF']
# scaler = StandardScaler()
# df_filtered[cols_to_scale] = scaler.fit_transform(df_filtered[cols_to_scale])

# we need to calc credit score approximation

#fico uses 
'''
Payment history (35%)
Amount owed (30%)
Length of credit history (15%)
New credit (10%)
Credit mix (10%)
'''
# change







#print(df_filtered.head())

# Save the cleaned and processed dataset
df_filtered.to_csv("datasetWithEad.csv", index=False)

# print("Filtered dataset with EAD saved as datasetNormalCCFEAD.csv")
