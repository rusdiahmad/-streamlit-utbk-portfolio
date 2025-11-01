
# Script to (re)train models from the Excel dataset.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_excel("NILAI UTBK ANGK 4.xlsx")
numeric_cols = df.select_dtypes(include=[float,int]).columns.tolist()
df["total_score"] = df[numeric_cols].sum(axis=1)
df["lulus"] = (df["total_score"] >= df["total_score"].median()).astype(int)

X = df[numeric_cols].fillna(df[numeric_cols].median())
y_reg = df["total_score"]
y_clf = df["lulus"]

X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

reg = LinearRegression().fit(X_train, y_reg_train)
clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_clf_train)

joblib.dump(reg, "models/reg_model.pkl")
joblib.dump(clf, "models/class_model.pkl")
print("Saved models to models/")
