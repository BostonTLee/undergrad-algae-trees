import datetime as dt

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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


def experiment(insitu, sat, model_type='random_forest', subset=None,
               merge_thresh=1, bloom_thresh=None):
    """ Run a algae bloom prediction experiment.

    Parameters
    ----------
    insitu : dataframe
    sat : dataframe
    subset : list of str
        Subset of predictors to use.
    model_type : 'random_forest', 'gradient_boost', or 'xgboost'
        Statistical model to use.
    merge_thresh : int
        Time delta, in number of days to allow for merging in-situ
        data to satellite data.
    bloom_thresh : int or float
        If not None, a binary response variable will be used indicating
        chlorophyll > bloom_thresh.

    Returns
    -------
    float
        MSE on test set.
    """
    df = thresh_date_merge(
        insitu,
        sat,
        merge_cols=['comid'],
        thresh=merge_thresh
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

    if bloom_thresh is None:
        X_train, X_test, y_train, y_test = train_test_split(
            df[subset],
            df['chlorophyll'],
            test_size=0.33,
            random_state=42
        )
        model_lookup = {'random_forest': RandomForestRegressor,
                        'gradient_boost': GradientBoostingRegressor,
                        'xgboost': XGBRegressor}
        model = model_lookup[model_type](random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return mean_squared_error(y_true=y_test, y_pred=y_pred)


def run_experiments():
    """ Run algae bloom experiments with different parameters. """
    insitu = pd.read_csv('complete_in-situ.csv', low_memory=False)
    sat = pd.read_csv('larger_landsat.csv', low_memory=False)
    insitu = prep_insitu(insitu)

    log_chloro = np.log(insitu['chlorophyll'])
    insitu['chlorophyll'] = log_chloro
    insitu = insitu[np.isfinite(log_chloro)]

    sat = prep_sat(sat)
    subsets = [['red', 'green', 'blue'],
               ['red', 'green', 'blue', 'nir'],
               ['red', 'green', 'blue', 'nir', 'aerosol'],
               ['red', 'green', 'blue', 'nir', 'aerosol', 'areasqkm', 'dwl'],
               ['red', 'green', 'blue', 'nir', 'aerosol', 'areasqkm', 'dwl',
                'month'],
               ['red', 'green', 'blue', 'nir', 'aerosol', 'areasqkm', 'dwl',
                'swir1', 'swir2', 'tir1', 'tir2']]
    output_table = []
    for model_type in ['random_forest', 'gradient_boost', 'xgboost']:
        for s, subset in enumerate(subsets):
            for merge_thresh in range(6):
                mse = experiment(
                    insitu,
                    sat,
                    model_type=model_type,
                    subset=subset,
                    merge_thresh=merge_thresh,
                    strict=True
                )
                output_table.append([model_type, s, merge_thresh, np.exp(mse)])
    output_table = pd.DataFrame(
        output_table,
        columns=['model_type', 'subset', 'merge_thresh', 'mse']
    )
    return output_table


if __name__ == '__main__':
    print(run_experiments().to_string())
