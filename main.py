import datetime as dt

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (cross_val_score,
                                     RandomizedSearchCV,
                                     train_test_split)
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


pd.options.mode.chained_assignment = None


def prep_insitu(insitu):
    """ Prepare insitu data. """
    insitu.columns = [x.lower() for x in insitu.columns]
    insitu = insitu[['comid', 'date', 'analyte', 'raw_value']]
    insitu['date'] = pd.to_datetime(insitu['date'])
    insitu['analyte'] = insitu['analyte'].str.lower()
    insitu['analyte'] = insitu['analyte'].apply(
        lambda x: 'chlorophyll' if 'chloro' in x else 'secchi'
    )
    insitu_chloro = insitu[insitu['analyte'] == 'chlorophyll']
    insitu_secchi = insitu[insitu['analyte'] == 'secchi']
    insitu_chloro.drop('analyte', axis=1, inplace=True)
    insitu_secchi.drop('analyte', axis=1, inplace=True)
    insitu_chloro.rename({'raw_value': 'chlorophyll'}, axis=1, inplace=True)
    insitu_secchi.rename({'raw_value': 'secchi'}, axis=1, inplace=True)
    insitu_chloro = insitu_chloro \
        .groupby(['comid', 'date']) \
        .mean() \
        .reset_index()
    insitu_secchi = insitu_secchi \
        .groupby(['comid', 'date']) \
        .mean() \
        .reset_index()
    insitu = insitu_chloro.merge(insitu_secchi, how='left')
    insitu.dropna(subset=['comid', 'date', 'chlorophyll'], inplace=True)
    return insitu


def prep_sat(sat):
    """ Prepare satellite data. """
    sat.columns = [x.lower() for x in sat.columns]
    sat = sat[['comid', 'date', 'red', 'blue', 'green', 'areasqkm', 'distance',
               'aerosol', 'nir', 'swir1', 'swir2', 'tir1', 'tir2',
               'pcount_dswe1', 'pcount_dswe3', 'dwl']]
    sat['date'] = pd.to_datetime(sat['date'])
    sat = sat.groupby(['comid', 'date']).mean().reset_index()
    return sat


def thresh_date_merge(left, right, date_col='date',
                      merge_cols=[], thresh=1, strict=True):
    """ Merge two dataframes, accepting +- some number of days for matches.

    Parameters
    ----------
    left, right : dataframe
        Input dataframes to be merged.
    date_col : str
        Column name indicating the date of observation. This must exist in both
        the `left` and `right` dataframes.
    merge_cols : list of str
        Other columns to use a merge keys. Column names must be the same in the
        `left` and `right` dataframes.
    thresh : int
        The difference in days between observations in `left` and observations
        in `right` that is acceptable for matching.
    strict : bool
        Whether to prevent observations in `left` from being associated
        with multiple different observations of `right`.

    Returns
    -------
    dataframe
    """
    left = left.copy()
    left = left \
        .reset_index(drop=True) \
        .reset_index() \
        .rename({'index': 'obs_id'}, axis=1)
    right = right.copy()
    # create a list to hold the individual merge results
    assemble = []
    # merge on equal dates
    merge_keys = [date_col] + merge_cols
    assemble.append(left.merge(right, on=merge_keys))
    for i in range(1, thresh + 1):
        # make copies of the right dataframe
        right_shifted_fwd = right.copy()
        right_shifted_bwd = right.copy()
        # shift date column in copies
        right_shifted_fwd['date'] = right['date'] + dt.timedelta(days=i)
        right_shifted_bwd['date'] = right['date'] + dt.timedelta(days=-i)
        # merge on right date is i days behind left date
        assemble.append(left.merge(right_shifted_fwd, on=merge_keys))
        # merge on right date is i days ahead of left date
        assemble.append(left.merge(right_shifted_bwd, on=merge_keys))
    # create a new dataframe from individual merge results
    df = pd.concat(assemble)
    if strict:
        df = df.drop_duplicates(subset=['obs_id'])
    df = df.drop('obs_id', axis=1)
    return df


def experiment_dataset(insitu, sat, subset=None,
                       merge_thresh=1, split='train'):
    """ Create a dataset for an algae bloom prediction experiment.

    Parameters
    ----------
    insitu : dataframe
    sat : dataframe
    subset : list of str
        Subset of predictors to use.
    merge_thresh : int
        Time delta, in number of days to allow for merging in-situ
        data to satellite data.

    Returns
    -------
    (X, y) : tuple of array
        Predictor and response data as numpy arrays. Observations are subset
        to train or test set based on `split`.
    """
    df = thresh_date_merge(
        insitu,
        sat,
        merge_cols=['comid'],
        thresh=merge_thresh,
        strict=True
    )
    df = df.set_index(['comid', 'date'])
    encoder = OneHotEncoder(drop='first', sparse=False)
    month_values = df.index.get_level_values(1).month.values.reshape(-1, 1)
    month_indicators = encoder.fit_transform(month_values)
    month_indicators = pd.DataFrame(
        month_indicators,
        columns=encoder.categories_[0][1:],
        index=df.index
    )
    df = pd.concat([df, month_indicators], axis=1)
    if 'month' in subset:
        subset = subset.copy()
        subset.remove('month')
        subset += list(encoder.categories_[0][1:])

    X_train, X_test, y_train, y_test = train_test_split(
        df[subset],
        df['chlorophyll'],
        test_size=0.33,
        random_state=42
    )
    if split == 'train':
        return X_train, y_train
    if split == 'test':
        return X_test, y_test
    if split == 'complete':
        return df[subset].values, df['chlorophyll'].values.reshape(-1, 1)


def experiment_model(model_type='random_forest'):
    """ Create a model for an algae bloom prediction experiment.

    Parameters
    ----------
    model_type : 'random_forest', 'gradient_boost', or 'xgboost'
        Statistical model to use.

    Returns
    -------
    predictor
        Model object.
    """
    model_lookup = {'random_forest': RandomForestRegressor,
                    'gradient_boost': GradientBoostingRegressor,
                    'xgboost': XGBRegressor}
    model = model_lookup[model_type](random_state=42)
    return model


def run_experiments():
    """ Run algae bloom experiments with different parameters. """
    insitu = pd.read_csv('complete_in-situ.csv', low_memory=False)
    sat = pd.read_csv('larger_landsat.csv', low_memory=False)
    insitu = prep_insitu(insitu)
    sat = prep_sat(sat)

    log_chloro = np.log(insitu['chlorophyll'])
    insitu['chlorophyll'] = log_chloro
    insitu = insitu[np.isfinite(log_chloro)]

    subsets = [['red', 'green', 'blue'],
               ['red', 'green', 'blue', 'nir'],
               ['red', 'green', 'blue', 'nir', 'aerosol'],
               ['red', 'green', 'blue', 'nir', 'aerosol', 'areasqkm', 'dwl'],
               ['red', 'green', 'blue', 'nir', 'aerosol', 'areasqkm', 'dwl',
                'swir1', 'swir2', 'tir1', 'tir2'],
               ['red', 'green', 'blue', 'nir', 'aerosol', 'areasqkm', 'dwl',
                'swir1', 'swir2', 'tir1', 'tir2', 'month']]
    output_table = []
    for model_type in ['random_forest', 'gradient_boost', 'xgboost']:
        for s, subset in enumerate(subsets):
            for merge_thresh in range(0, 11):
                X_train, y_train = experiment_dataset(
                    insitu,
                    sat,
                    subset=subset,
                    merge_thresh=merge_thresh,
                    split='train'
                )
                model = experiment_model(model_type)
                cv_mse = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    scoring='neg_mean_squared_error',
                    cv=5
                )
                mse = np.abs(cv_mse).mean()
                row = [model_type, s, merge_thresh, np.exp(mse)]
                output_table.append(row)
    output_table = pd.DataFrame(
        output_table,
        columns=['model_type', 'subset', 'merge_thresh', 'mse']
    )
    print(output_table.to_string())
    output_table.to_csv('results.csv', index=False)

    # evaluate test set error with best model
    best_experiment = output_table.loc[output_table['mse'].idxmin()]
    X_train, y_train = experiment_dataset(
        insitu,
        sat,
        subset=subsets[best_experiment['subset']],
        merge_thresh=best_experiment['merge_thresh'],
        split='train'
    )
    X_test, y_test = experiment_dataset(
        insitu,
        sat,
        subset=subsets[best_experiment['subset']],
        merge_thresh=best_experiment['merge_thresh'],
        split='train'
    )
    X, y = experiment_dataset(
        insitu,
        sat,
        subset=subsets[best_experiment['subset']],
        merge_thresh=best_experiment['merge_thresh'],
        split='train'
    )
    best_model = experiment_model(best_experiment['model_type'])
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    score = np.exp(mean_squared_error(y_train, y_pred))
    print(f'Best model MSE on test set: {score:.4f}')
    best_model.fit(X, y)
    # use the best model to get predictions
    unobs_X = sat.set_index(['comid', 'date'])
    encoder = OneHotEncoder(drop='first', sparse=False)
    month_values = unobs_X \
        .index.get_level_values(1).month.values.reshape(-1, 1)
    month_indicators = encoder.fit_transform(month_values)
    month_indicators = pd.DataFrame(
        month_indicators,
        columns=encoder.categories_[0][1:],
        index=unobs_X.index
    )
    unobs_X = pd.concat([unobs_X, month_indicators], axis=1)
    subset = subsets[best_experiment['subset']].copy()
    if 'month' in subset:
        subset.remove('month')
        subset += list(encoder.categories_[0][1:])
    unobs_X = unobs_X[subset]
    y_pred = np.exp(best_model.predict(unobs_X))
    y_pred = pd.Series(y_pred, index=unobs_X.index)
    sat_pred = unobs_X.copy()
    sat_pred['pred_chlorophyll'] = y_pred
    sat_pred.to_csv('predictions.csv')


if __name__ == '__main__':
    run_experiments()
