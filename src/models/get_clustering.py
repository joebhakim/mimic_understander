import pandas as pd
import numpy as np
import sys

import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA


from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, LogisticRegression

from sklearn.metrics import roc_auc_score, roc_curve


from econml.drlearner import DRLearner
import shap

main_dir = '/home/joe/covid_data_project/'
data_dir = main_dir + 'data/'
fig_dir = main_dir + 'figs/'


def standardize_impute(covariate_df_splits):
    si = SimpleImputer()
    ss = StandardScaler()

    X_train_imp = si.fit_transform(covariate_df_splits['train'])
    X_test_imp = si.fit_transform(covariate_df_splits['test'])
    X_train_imp_std = ss.fit_transform(X_train_imp)
    X_test_imp_std = ss.transform(X_test_imp)

    return X_train_imp_std, X_test_imp_std


def print_unsupervised_clustering(covariate_df_splits, yt_df_splits, output_filepath, to_plot=True):
    X_train_imp_std, X_test_imp_std = standardize_impute(covariate_df_splits)

    pca = PCA(n_components=4)

    X_train_pca = pca.fit_transform(X_train_imp_std)
    X_test_pca = pca.transform(X_test_imp_std)

    if to_plot:
        train_pc_df = pd.DataFrame(X_train_pca, columns=['PC'+str(i+1) for i in range(4)],
                                   index=covariate_df_splits['train'].index)
        test_pc_df = pd.DataFrame(X_test_pca, columns=['PC'+str(i+1) for i in range(4)],
                                  index=covariate_df_splits['test'].index)

        train_df_to_plot = yt_df_splits['train'].merge(
            train_pc_df, on='patient_id')
        test_df_to_plot = yt_df_splits['test'].merge(
            test_pc_df, on='patient_id')

        plt.figure()
        ax_treated = sns.scatterplot(
            data=train_df_to_plot, x='PC1', y='PC2', hue='treated')
        ax_treated.get_figure().savefig(output_filepath + 'ax_treated.png', dpi=300)
        plt.figure()
        ax_outcome = sns.scatterplot(
            data=train_df_to_plot, x='PC1', y='PC2', hue='death')
        ax_outcome.get_figure().savefig(output_filepath + 'ax_outcome.png', dpi=300)
        plt.figure()
        ax_both = sns.scatterplot(
            data=train_df_to_plot, x='PC1', y='PC2', hue='treated', style='death')
        ax_both.get_figure().savefig(output_filepath + 'ax_both.png', dpi=300)


def print_supervised_clustering(covariate_df_splits, yt_df_splits, output_filepath):
    X_train_imp_std, X_test_imp_std = standardize_impute(covariate_df_splits)

    en = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5)
    en.fit(X_train_imp_std, yt_df_splits['train']['death'])

    rf = RandomForestClassifier(n_estimators=10)
    rf.fit(X_train_imp_std, yt_df_splits['train']['death'])

    train_rf_df = pd.DataFrame({'random forest': rf.predict_proba(X_train_imp_std)[:, 1],
                                'elasticnet': en.predict_proba(X_train_imp_std)[:, 1]},
                               index=covariate_df_splits['train'].index)

    test_rf_df = pd.DataFrame({'random forest': rf.predict_proba(X_test_imp_std)[:, 1],
                               'elasticnet': en.predict_proba(X_test_imp_std)[:, 1]},
                              index=covariate_df_splits['test'].index)

    train_df_to_plot = yt_df_splits['train'].merge(
        train_rf_df, on='patient_id')
    test_df_to_plot = yt_df_splits['test'].merge(test_rf_df, on='patient_id')

    for model_type in ['random forest', 'elasticnet']:
        fpr_train, tpr_train, _ = roc_curve(
            train_df_to_plot['death'], train_df_to_plot[model_type])
        fpr_test, tpr_test, _ = roc_curve(
            test_df_to_plot['death'], test_df_to_plot[model_type])
        train_auc = roc_auc_score(
            train_df_to_plot['death'], train_df_to_plot[model_type])
        test_auc = roc_auc_score(
            test_df_to_plot['death'], test_df_to_plot[model_type])

        roc_train = pd.DataFrame({'fpr': fpr_train, 'tpr': tpr_train})
        roc_train['train/test'] = 'train AUC = ' + str(np.round(train_auc, 3))

        roc_test = pd.DataFrame({'fpr': fpr_test, 'tpr': tpr_test})
        roc_test['train/test'] = 'test AUC = ' + str(np.round(test_auc, 3))

        rocs = pd.concat((roc_train, roc_test))

        plt.figure()
        ax = sns.lineplot(data=rocs, x='fpr', y='tpr',
                          hue='train/test', ci=None)
        ax.set_title(model_type)
        plt.savefig(output_filepath + '/' + model_type + '_ROC.png', dpi=300)

    top_feats_rf = np.argsort(rf.feature_importances_)[::-1][:20]
    top_feats_names_rf = np.array(
        covariate_df_splits['train'].columns)[top_feats_rf]

    top_feats_en = np.argsort(np.abs(en.coef_)).squeeze()[::-1][:20]
    top_feats_names_en = np.array(
        covariate_df_splits['train'].columns)[top_feats_en]


def print_cates_clustering(covariate_df_splits, yt_df_splits, output_filepath):
    X_train_imp_std, X_test_imp_std = standardize_impute(covariate_df_splits)
    y_train = yt_df_splits['train']['death'] * 1.0
    t_train = yt_df_splits['train']['treated'] * 1.0

    drl = DRLearner(model_propensity=RandomForestClassifier(),
                    model_regression=RandomForestClassifier(),
                    model_final=RandomForestRegressor())

    drl.fit(y_train, t_train, X_train_imp_std)

    X_train_df = covariate_df_splits['train']

    y_explainer = shap.TreeExplainer(drl.models_regression[1], X_train_imp_std)
    t_explainer = shap.TreeExplainer(drl.models_propensity[1], X_train_imp_std)
    cate_explainer = shap.TreeExplainer(
        drl.model_final.models_cate[0], X_train_imp_std)

    y_shap_values = y_explainer.shap_values(
        X_train_imp_std, check_additivity=False)[1]
    t_shap_values = t_explainer.shap_values(
        X_train_imp_std, check_additivity=False)[1]
    cate_shap_values = cate_explainer.shap_values(
        X_train_imp_std, check_additivity=False)

    plt.figure()
    shap.summary_plot(y_shap_values, X_train_df, show=False)
    fig = plt.gcf()
    fig.subplots_adjust(left=0.5)
    fig.savefig(output_filepath + '/explainer_y.png', dpi=300)

    plt.figure()
    shap.summary_plot(t_shap_values, X_train_df, show=False)
    fig = plt.gcf()
    fig.subplots_adjust(left=0.5)
    fig.savefig(output_filepath + '/explainer_t.png', dpi=300)

    plt.figure()
    cate_shap_values_trim = np.where(
        abs(cate_shap_values) < 500, cate_shap_values, np.nan)
    shap.summary_plot(cate_shap_values_trim, X_train_df, show=False)
    fig = plt.gcf()
    fig.subplots_adjust(left=0.5)
    fig.savefig(output_filepath + '/explainer_cate.png', dpi=300)

    plt.figure()
    cates_train = drl.model_final.models_cate[0].predict(X_train_imp_std)
    cates_train_trim = np.where(abs(cates_train) < 10, cates_train, np.nan)
    ax = sns.distplot(cates_train_trim)
    ax.set_yscale('log')
    ax.set_title('CATE distribution')
    ax.set_xlabel('CATEs. more negative -> more helped by drug')
    plt.savefig(output_filepath + '/cate_dist.png', dpi=300)


    # get covariate means for negative cates
    X_train_df_std = pd.DataFrame(X_train_imp_std, columns=X_train_df.columns, index=X_train_df.index)
    X_train_cate_df = X_train_df_std.copy()
    X_train_cate_df['cate negative'] = cates_train < 0
    cate_important_vars = X_train_cate_df.groupby('cate negative').mean()
    cate_important_vars.loc['diff'] = cate_important_vars.loc[True] - cate_important_vars.loc[False]
    cate_important_vars_sort = cate_important_vars.sort_values(by='diff', axis=1, ascending=False)
    print('most cate different features')
    print(cate_important_vars_sort.iloc[:, :20])


    # what about most posive cates?
    X_train_cate_df['cate positive'] = cates_train > 0
    cate_harmful_vars = X_train_cate_df.groupby('cate positive').mean().drop('cate negative', axis=1)
    cate_harmful_vars.loc['diff'] = cate_harmful_vars.loc[True] - cate_harmful_vars.loc[False]
    cate_harmful_vars = cate_harmful_vars.sort_values(by='diff', axis=1, ascending=True)
    print('most cate different features')
    print(cate_harmful_vars.iloc[:, :20].T)