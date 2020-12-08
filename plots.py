import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

from main import prep_insitu, prep_sat, experiment_dataset, experiment_model


# data
insitu = pd.read_csv('complete_in-situ.csv', low_memory=False)
sat = pd.read_csv('larger_landsat.csv', low_memory=False)
insitu = prep_insitu(insitu)
sat = prep_sat(sat)
chloro = insitu['chlorophyll'].copy()
log_chloro = np.log(chloro)
insitu['chlorophyll'] = log_chloro
insitu = insitu[np.isfinite(log_chloro)]
# model performance
results = pd.read_csv('results.csv')
# plot settings
current_palette = sb.color_palette()
palette = {'random_forest': current_palette[0],
           'gradient_boost': current_palette[1],
           'xgboost': current_palette[2]}


def plot_histograms():
    fig, axs = plt.subplots(ncols=2, figsize=(12, 4))
    sb.histplot(
        chloro,
        color='xkcd:algae',
        bins=15,
        ax=axs[0]
    )
    sb.despine()
    axs[0].set_xlabel('Chlorophyll')
    axs[0].set_title('Distribution of Chlorophyll Measurements')

    sb.histplot(
        log_chloro[np.isfinite(log_chloro)],
        color='xkcd:algae',
        bins=15,
        ax=axs[1]
    )
    sb.despine()
    axs[1].set_xlabel('log(Chlorophyll)')
    axs[1].set_title('Distribution of Chlorophyll Measurements'
                     '\n(log-transform)')
    return fig


def plot_feature_importance(ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    model = experiment_model('random_forest')
    variables = ['red', 'green', 'blue', 'nir', 'aerosol', 'areasqkm',
                 'dwl', 'swir1', 'swir2', 'tir1', 'tir2']
    X_train, y_train = experiment_dataset(
        insitu,
        sat,
        subset=variables,
        merge_thresh=3
    )
    model.fit(X_train, y_train)
    feature_importances = model.feature_importances_
    importance_series = pd.Series(dict(zip(variables, feature_importances)))
    importance_series = importance_series.sort_values(ascending=False)
    sb.barplot(
        y=importance_series.index,
        x=importance_series.values,
        color='xkcd:algae',
        ax=ax
    )
    sb.despine()
    ax.set_title('Feature Importances')
    return ax


def plot_thresh_performance(ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    merge_thresh_results = results \
        .groupby(['model_type', 'merge_thresh'])['mse'] \
        .min() \
        .reset_index() \
        .sort_values('model_type')
    sb.lineplot(
        data=merge_thresh_results,
        x='merge_thresh',
        y='mse',
        hue='model_type',
        palette=palette,
        ci=None,
        ax=ax
    )
    plt.legend(frameon=False)
    sb.despine()
    ax.set_title('CV Error vs Merge Threshold')
    ax.set_xlabel('Merge threshold (days)')
    ax.set_ylabel('CV Mean Squared Error')
    return ax


def plot_subset_performance(ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    subset_results = results \
        .groupby(['model_type', 'subset'])['mse'] \
        .min() \
        .reset_index() \
        .sort_values('model_type')
    sb.lineplot(
        data=subset_results,
        x='subset',
        y='mse',
        hue='model_type',
        palette=palette,
        ci=None,
        ax=ax
    )
    plt.axvline(3, linestyle='dotted', color='grey')
    plt.text(3, 1.8, 'areasqkm', ha='right')
    plt.legend(frameon=False)
    sb.despine()
    ax.set_title('CV Error for Different Feature Subsets')
    ax.set_xlabel('Feature Subset')
    ax.set_ylabel('CV Mean Squared Error')
    return ax


def plot_predicted_values(ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    predictions = pd.read_csv('predictions.csv')
    sb.scatterplot(
        data=predictions,
        x='dwl',
        y='pred_chlorophyll',
        color='xkcd:algae',
        linewidth=0,
        ax=ax
    )
    sb.despine()
    ax.set_title('Predicted Values')
    ax.set_xlabel('Dominant wavelength')
    ax.set_ylabel('Predicted chlorophyll value $\\frac{\\mu g}{mL}$')
    return ax


if __name__ == '__main__':
    plot_histograms()
    plt.savefig('figures/log_chloro_hist.png', dpi=200, bbox_inches='tight')
    plot_feature_importance()
    plt.savefig('figures/feature_importance.png', dpi=200, bbox_inches='tight')
    plot_thresh_performance()
    plt.savefig('figures/merge_thresh.png', dpi=200, bbox_inches='tight')
    plot_subset_performance()
    plt.savefig('figures/subset.png', dpi=200, bbox_inches='tight')
    plot_predicted_values()
    plt.savefig('figures/predictions.png', dpi=200, bbox_inches='tight')
