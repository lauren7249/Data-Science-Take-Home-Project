import pandas
from collections import OrderedDict
import h2o
from predict_open_invoices import ID_COLUMNS


def months_between(start_date: pandas.Series, end_date: pandas.Series) -> int:
    """ Calculate number of months between two dates in a pandas Dataframe """
    months = (end_date.dt.to_period('M') - start_date.dt.to_period('M'))
    months = months.map(lambda m: m.n if not pandas.isnull(m) else None)
    return months


def apply_filters(filters: OrderedDict, apply_to_df: pandas.DataFrame) -> (pandas.DataFrame, pandas.DataFrame):
    """ Perform sequential filter steps on a pandas Dataframe and report impact on the input data """
    filter_stats = pandas.DataFrame(columns=['Bad Data', '% of Unfiltered Data', 'Rows Filter Applied To',
                                             '% Filtered at Step'])
    filtered_data = apply_to_df.copy()
    step_num = 0
    for filter_step, exclude_rows in filters.items():
        step_num += 1
        rows_applied_to = filtered_data.__len__()
        filtered_data = filtered_data.loc[~filtered_data.index.isin(exclude_rows.index)]
        step_stats = [filter_step,
                      exclude_rows.__len__() / apply_to_df.__len__(),
                      rows_applied_to,
                      1 - filtered_data.__len__() / rows_applied_to]
        filter_stats.loc[step_num] = step_stats
    return filtered_data, filter_stats


def get_h2o_frame(df: pandas.DataFrame) -> h2o.H2OFrame:
    """Convert a pandas dataframe to a properly formatted H2O frame for training and prediction."""
    h2o.init(nthreads=-1, max_mem_size=12)
    id_columns_h2o = [col for col in ID_COLUMNS if col in df.columns]
    return h2o.H2OFrame(df.select_dtypes(exclude='datetime'),
                        column_types=dict(zip(id_columns_h2o, ["string"] * len(id_columns_h2o))))
