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
chloro = insitu['chlorophyll']
log_chloro = np.log(chloro)
insitu['chlorophyll'] = log_chloro
insitu = insitu[np.isfinite(log_chloro)]

# exploration
sb.histplot(chloro, color='xkcd:algae', bins=15)
sb.despine()
plt.xlabel('Chlorophyll')
plt.title('Distribution of Chlorophyll Measurements')
plt.savefig('figures/chloro_hist.png', dpi=200, bbox_inches='tight')

sb.histplot(log_chloro, color='xkcd:algae', bins=15)
sb.despine()
plt.xlabel('Chlorophyll')
plt.title('Distribution of Chlorophyll Measurements\n(log-transform)')
plt.savefig('figures/log_chloro_hist.png', dpi=200, bbox_inches='tight')

# feature importance
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
    color='xkcd:algae'
)
sb.despine()
plt.title('Feature Importances')
plt.savefig('figures/feature_importances.png', dpi=200, bbox_inches='tight')

# model performance
results = pd.read_csv('results.csv')
current_palette = sb.color_palette()
palette = {'random_forest': current_palette[0],
           'gradient_boost': current_palette[1],
           'xgboost': current_palette[2]}
# merge threshold vs model performance
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
    ci=None
)
plt.legend(frameon=False)
sb.despine()
plt.title('CV Error vs Merge Threshold')
plt.xlabel('Merge threshold (days)')
plt.ylabel('CV Mean Squared Error')
plt.savefig('figures/merge_thresh.png', dpi=200, bbox_inches='tight')
# feature subset vs model performance
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
)
plt.axvline(3, linestyle='dotted', color='grey')
plt.text(3, 1.8, 'areasqkm', ha='right')
plt.legend(frameon=False)
sb.despine()
plt.title('CV Error for Different Feature Subsets')
plt.xlabel('Feature Subset')
plt.ylabel('CV Mean Squared Error')
plt.savefig('figures/subset.png', dpi=200, bbox_inches='tight')

# predicted values
predictions = pd.read_csv('predictions.csv')
sb.scatterplot(
    data=predictions,
    x='dwl',
    y='pred_chlorophyll',
    color='xkcd:algae',
    linewidth=0
)
sb.despine()
plt.title('Predicted Values')
plt.xlabel('Dominant wavelength')
plt.ylabel('Predicted chlorophyll value $\\frac{\\mu g}{mL}$')
plt.savefig('figures/predictions.png', dpi=200, bbox_inches='tight')
