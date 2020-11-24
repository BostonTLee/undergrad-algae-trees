import datetime as dt

import pandas as pd


def thresh_date_merge(left, right, date_col='date', merge_cols=[], thresh=1):
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

    Returns
    -------
    dataframe
    """
    left = left.copy()
    right = right.copy()
    # create a list to hold the individual merge results
    assemble = []
    # merge on equal dates
    merge_keys = [date_col] + merge_cols
    assemble.append(left.merge(right, on=merge_keys))
    for i in range(1, thresh + 1):
        # add columns to sat that hold the date plus/minus i days
        right[f'date_plus_{i}'] = right['date'] + dt.timedelta(days=i)
        right[f'date_minus_{i}'] = right['date'] + dt.timedelta(days=-i)
        # merge on right date is i days ahead of left date
        merge_keys_r = [f'date_plus_{i}'] + merge_cols
        assemble.append(
            left.merge(right, left_on=merge_keys, right_on=merge_keys_r)
        )
        # merge on right date is i days behind left date
        merge_keys_r = [f'date_minus_{i}'] + merge_cols
        assemble.append(
            left.merge(right, left_on=merge_keys, right_on=merge_keys_r)
        )
    # create a new dataframe from individual merge results
    df = pd.concat(assemble)
    return df


if __name__ == '__main__':
    insitu = pd.read_csv('complete_in-situ.csv', low_memory=False)
    sat = pd.read_csv('larger_landsat.csv', low_memory=False)
    insitu.columns = [x.lower() for x in insitu.columns]
    sat.columns = [x.lower() for x in sat.columns]
    insitu['date'] = pd.to_datetime(insitu['date'])
    sat['date'] = pd.to_datetime(sat['date'])
    df = thresh_date_merge(insitu, sat, merge_cols=['comid'], thresh=10)
    print(len(df))
