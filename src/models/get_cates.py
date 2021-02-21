import pandas as pd
import numpy as np
import sys

import seaborn as sns
import matplotlib.pyplot as plt



CYXT = pd.read_csv('/home/joe/covid_data_project/data/CYXT.csv').set_index(['PATIENT ID','treated','died','DATETIME'])
CYXT_timemean = CYXT.groupby(['PATIENT ID','treated','died']).mean()


from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, roc_curve


var_names = CYXT_timemean.columns.values



CYTX_train, CYTX_test = train_test_split(CYXT_timemean)

y_train = CYTX_train.reset_index()['died'] * 1
t_train = CYTX_train.reset_index()['treated'] * 1
X_train = CYTX_train[var_names].drop(['GLU/GLY_ING/INPAT','SAT_02_ING/INPAT'], axis=1).values

y_test = CYTX_test.reset_index()['died'] * 1
t_test = CYTX_test.reset_index()['treated'] * 1
X_test = CYTX_test[var_names].drop(['GLU/GLY_ING/INPAT','SAT_02_ING/INPAT'], axis=1).values


varnames_selected = np.array(CYTX_train[var_names].drop(['GLU/GLY_ING/INPAT','SAT_02_ING/INPAT'], axis=1).columns)



si = SimpleImputer()
ss = StandardScaler()

X_train_imp = si.fit_transform(X_train)
X_test_imp = si.fit_transform(X_test)
X_train_imp_std = ss.fit_transform(X_train_imp)
X_test_imp_std = ss.transform(X_test_imp)


rf_y = RandomForestClassifier()
rf_t = RandomForestClassifier()

rf_y.fit(X_train_imp_std, y_train)
y_train_preds = rf_y.predict_proba(X_train_imp_std)[:, 1]
y_test_preds = rf_y.predict_proba(X_test_imp_std)[:, 1]

rf_t.fit(X_train_imp_std, t_train)
t_train_preds = rf_t.predict_proba(X_train_imp_std)[:, 1]
t_test_preds = rf_t.predict_proba(X_test_imp_std)[:, 1]


from econml.drlearner import DRLearner



drl = DRLearner(model_propensity=RandomForestClassifier(),
                model_regression=RandomForestClassifier(),
                model_final=RandomForestRegressor())

drl.fit(y_train, t_train, X_train_imp_std)


import shap

X_train_df = CYTX_train[var_names].drop(['GLU/GLY_ING/INPAT','SAT_02_ING/INPAT'], axis=1)

y_explainer = shap.TreeExplainer(drl.models_regression[1], X_train_imp_std)
t_explainer = shap.TreeExplainer(drl.models_propensity[1], X_train_imp_std)
cate_explainer = shap.TreeExplainer(drl.model_final.models_cate[0], X_train_imp_std)

y_shap_values = y_explainer.shap_values(X_train_imp_std, check_additivity=False)[1]
t_shap_values = t_explainer.shap_values(X_train_imp_std, check_additivity=False)[1]
cate_shap_values = cate_explainer.shap_values(X_train_imp_std, check_additivity=False)




plt.figure()
shap.summary_plot(y_shap_values, X_train_df, show=False)
fig = plt.gcf()
fig.subplots_adjust(left=0.5)
fig.savefig('/home/joe/covid_data_project/figs/explainer_2.png', dpi=300)



plt.figure()
shap.summary_plot(t_shap_values, X_train_df, show=False)
fig = plt.gcf()
fig.subplots_adjust(left=0.5)
fig.savefig('/home/joe/covid_data_project/figs/explainer_1.png', dpi=300)



plt.figure()
shap.summary_plot(cate_shap_values, X_train_df, show=False)
fig = plt.gcf()
fig.subplots_adjust(left=0.5)
fig.savefig('/home/joe/covid_data_project/figs/explainer_3.png', dpi=300)

