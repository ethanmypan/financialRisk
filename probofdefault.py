## this file outputs an array of probability of default in the value y_pred_proba
#This file can act as baseline model for prob of default for each person

import pandas as pd
import numpy as np
import matplotlib as plt

#Read in cleaned dataframe
df = pd.read_csv("datasetNormalCCFEAD.csv")

modelcols = ['target', 'amt_credit', 'amt_annuity', 'amt_income_total', 'days_birth',
       'days_employed', 'days_id_publish', 'days_registration', 'ext_source_2',
       'cnt_children', 'cnt_fam_members', 'CCF', 'Undrawn_Amount',
       'EAD_With_Normal_CCF']
#filtering cols - can adjust this bit here 
df_filtered = df[modelcols]
df_filtered = df_filtered.reset_index()

numeric_cols = df_filtered.select_dtypes(include=['number']).columns
df_filtered[numeric_cols] = df_filtered[numeric_cols].fillna(df_filtered[numeric_cols].mean())


#importing in sklearn stuff 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

#putting categorical columns in correct format 
categorical_cols = df_filtered.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_filtered[col] = le.fit_transform(df_filtered[col].astype(str))  
    label_encoders[col] = le
df_filtered

#split data up for training
X = df_filtered.drop(columns=['target'])
y = df_filtered['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#numerical cols fixed in correct format
scaler = StandardScaler()
X_train[X_train.select_dtypes(include=['number']).columns] = scaler.fit_transform(X_train[X_train.select_dtypes(include=['number']).columns])
X_test[X_test.select_dtypes(include=['number']).columns] = scaler.transform(X_test[X_test.select_dtypes(include=['number']).columns])

#training logistic regression
log_reg = LogisticRegression(max_iter=1000)  
log_reg.fit(X_train, y_train)


y_pred = log_reg.predict(X_test)
y_pred_proba = log_reg.predict_proba(X_test)[:, 1] 

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
y_pred_proba