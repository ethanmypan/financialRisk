#Baseline Model for loss given default.
#formula is EAD * (1- recovery rate)
import pandas as pd
import numpy as np 
# recovery rate is (total amount repaid/ total balance of loan) * 100
#total amount repaid = amt_annuity 
#total amount credit  = amt _credit
df = pd.read_csv("datasetNormalCCFEAD.csv")
df["LGD"] = 1-((df['amt_credit'] - df['amt_annuity'])/ df["EAD_With_Normal_CCF"]) 

#spliting into training 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import matplotlib as plt
modelcols = ['target', 'amt_credit', 'amt_annuity', 'amt_income_total', 'days_birth',
       'days_employed', 'days_id_publish', 'days_registration', 'ext_source_2',
       'cnt_children', 'cnt_fam_members', 'CCF', 'Undrawn_Amount',
       'EAD_With_Normal_CCF']

df_filtered = df[modelcols]
df_filtered = df_filtered.reset_index()

numeric_cols = df_filtered.select_dtypes(include=['number']).columns
df_filtered[numeric_cols] = df_filtered[numeric_cols].fillna(df_filtered[numeric_cols].mean())




categorical_cols = df_filtered.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_filtered[col] = le.fit_transform(df_filtered[col].astype(str))  
    label_encoders[col] = le

df_filtered["LGD"] = 1-((df['amt_credit'] - df['amt_annuity'])/ df["EAD_With_Normal_CCF"]) 
#split data up for training
X = df_filtered.drop(columns=['LGD', 'CCF', "EAD_With_Normal_CCF", 'amt_annuity', "Undrawn_Amount", "amt_credit", "index"])
y = df_filtered['LGD']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train[X_train.select_dtypes(include=['number']).columns] = scaler.fit_transform(X_train[X_train.select_dtypes(include=['number']).columns])
X_test[X_test.select_dtypes(include=['number']).columns] = scaler.transform(X_test[X_test.select_dtypes(include=['number']).columns])


rf = RandomForestRegressor(n_estimators=100,  
                           max_depth=10,      
                           min_samples_split=2, 
                           min_samples_leaf=1,  
                           random_state=42)


rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse}")

importances = rf.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 5))
plt.barh(feature_names, importances, color='skyblue')
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Random Forest Model")
plt.show()

