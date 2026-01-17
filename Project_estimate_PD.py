# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 10:16:05 2026

@author: kipoi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

loan_data = pd.read_csv("loan_data.csv")

print(loan_data.head())
print(loan_data.iloc[0])

print(loan_data.describe())
print(loan_data.info())

plt.figure(figsize=(10,6))

loan_data[loan_data['credit.policy']==1]['fico'].hist(bins = 30, color='red',
                                                      label = 'credit.policy = 1', 
                                                      alpha = 0.6)
loan_data[loan_data['credit.policy']==0]['fico'].hist(bins = 30, color='blue',
                                                      label = 'credit.policy = 0', 
                                                      alpha = 0.6)
plt.legend()
plt.xlabel('FICO')


plt.figure(figsize=(10,6))

loan_data[loan_data['not.fully.paid']==1]['fico'].hist(bins = 30, color='red',
                                                      label = 'not.fully.paid = 1', 
                                                      alpha = 0.6)
loan_data[loan_data['not.fully.paid']==0]['fico'].hist(bins = 30, color='blue',
                                                      label = 'not.fully.paid = 0', 
                                                      alpha = 0.6)
plt.legend()
plt.xlabel('FICO')


plt.figure(figsize=(10,6))
sns.countplot(x='purpose', hue='not.fully.paid', data = loan_data, palette='Set1')

sns.jointplot(x='fico', y='int.rate', data = loan_data)

sns.lmplot(x='fico', y='int.rate', data = loan_data, hue='credit.policy', col='not.fully.paid', palette='Set1')



final_data = pd.get_dummies(loan_data, 'purpose', drop_first=True)

print(final_data.info())


#Modelisation

X = final_data.drop(['not.fully.paid'],axis=1)
Y = final_data['not.fully.paid']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=101)
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
pred = dt.predict(X_test)

print(classification_report(Y_test, pred))

rf = RandomForestClassifier()
rf.fit(X_train, Y_train)
pred_rf = rf.predict(X_test)

print(classification_report(Y_test, pred_rf))



"""logit = LogisticRegression(max_iter=20000)
logit.fit(X_train, Y_train)

pred_logit = logit.predict(X_test)

print(classification_report(Y_test, pred_logit))"""

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('logit', LogisticRegression(
        max_iter=2000,
        solver='lbfgs',
        
    ))
])

pipe.fit(X_train, Y_train)

Y_pred = pipe.predict(X_test)

print(classification_report(Y_test, Y_pred))

PD_test = pipe.predict_proba(X_test)[:,1]
print(PD_test)
print(roc_auc_score(Y_test, PD_test))


# Copie de travail
df = final_data.copy()

# Binning en 5 groupes (quintiles)
df['fico_bin'] = pd.qcut(df['fico'], q=5, duplicates='drop')
target = 'not.fully.paid'

woe_tab = (
    df.groupby('fico_bin')[target]
      .agg(['count', 'sum'])
      .rename(columns={'sum': 'bad'})
)

woe_tab['good'] = woe_tab['count'] - woe_tab['bad']
print(woe_tab)
eps = 1e-6
# Totaux
total_bad = woe_tab['bad'].sum()
total_good = woe_tab['good'].sum()

# Proportions
woe_tab['bad_rate'] = woe_tab['bad'] / total_bad
woe_tab['good_rate'] = woe_tab['good'] / total_good
# WOE
woe_tab['WOE'] = np.log(
    (woe_tab['good_rate'] + eps) /
    (woe_tab['bad_rate'] + eps)
)

print(woe_tab)

woe_map = woe_tab['WOE'].to_dict()

print(woe_map)

# Nouvelle variable
df['fico_woe'] = df['fico_bin'].map(woe_map)
print(df)


X_woe = df.drop(['not.fully.paid', 'fico', 'fico_bin'], axis=1)
y = df['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(
    X_woe, y,
    test_size=0.30,
    random_state=101,
    stratify=y
)

model = Pipeline([
    ('scaler', StandardScaler()),
    ('logit', LogisticRegression(max_iter=3000, solver='lbfgs'))
])

model.fit(X_train, y_train)

PD_test = model.predict_proba(X_test)[:, 1]
auc_woe = roc_auc_score(y_test, PD_test)

print("AUC avec fico_woe =", auc_woe)


# Jointure
df_merge = df.merge(
    woe_tab,
    on='fico_bin',
    how='left'
)


df_merge = df_merge.rename(columns={'WOE': 'fico_woe'})

print(df_merge)

beta = model.named_steps['logit'].coef_[0]

# intercept
beta_0 = model.named_steps['logit'].intercept_[0]

# tableau 
coef_df = pd.DataFrame({
    'variable': X_train.columns,
    'beta': beta
}).sort_values('beta', ascending=False)

print("Intercept (beta_0):", beta_0)
print(coef_df)



