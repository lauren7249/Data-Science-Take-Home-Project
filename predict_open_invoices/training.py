from collections import OrderedDict
import pandas
import random
import numpy
import h2o
from h2o.automl import H2OAutoML
from predict_open_invoices import ID_COLUMNS
from predict_open_invoices.utils import months_between, apply_filters
from predict_open_invoices.csv_test_data_io import get_csv_test_data
from predict_open_invoices.pre_processing import preprocess_invoices_with_payments
from predict_open_invoices.feature_engineering import feature_engineering


def __select_forecast_date(invoice_id, invoice_date, max_forecast_date, forecast_frequency: str = 'M',
                           first_transaction_date: pandas.Timestamp = None) -> pandas.Timestamp:
    """Given a set of invoice data with end dates, which can vary per row, and an overall start date, sample a date
    period with the range during which the invoice was OPEN."""
    if pandas.isnull(max_forecast_date):
        return None
    # begin forecast window when the invoice is active and the payments data is complete
    period_start = max(invoice_date, first_transaction_date) if first_transaction_date else invoice_date
    period_range = pandas.period_range(period_start, max_forecast_date, freq=forecast_frequency)
    if len(period_range) == 0:
        return None
    # ensure consistent forecast month per invoice across payments
    pseudorandom = random.Random(invoice_id)
    return pseudorandom.choice(period_range).to_timestamp()


_select_forecast_date = numpy.vectorize(__select_forecast_date)


def _normalize_by_company(invoice_point_in_time: pandas.DataFrame) -> pandas.DataFrame:
    """Add row weights to invoices for training."""
    invoice_point_in_time['converted_amount_inv'] = (
            invoice_point_in_time.amount_inv * invoice_point_in_time.root_exchange_rate_value_inv)
    totals_by_company = invoice_point_in_time.groupby("company_id", as_index=False).converted_amount_inv.sum()
    data = invoice_point_in_time.merge(totals_by_company, on="company_id", suffixes=('', '_company'))
    inv_pct_of_company_total = data.converted_amount_inv / data.converted_amount_inv_company
    data['inv_pct_of_company_total'] = inv_pct_of_company_total
    data.drop(columns=["converted_amount_inv"], inplace=True, errors='ignore')
    assert data.inv_pct_of_company_total.sum() == data.company_id.nunique(), \
        'Company amount normalization does not sum to 1 per company'
    return data


def _assign_open_forecast_date(invoices_with_payments: pandas.DataFrame) -> pandas.DataFrame:
    """Add a randomly sampled forecast date while the invoice was open."""
    invoices_with_payments['forecast_date_collected'] = _select_forecast_date(
        invoices_with_payments.invoice_id, invoices_with_payments.invoice_date, invoices_with_payments.collected_date)
    assert invoices_with_payments.groupby("invoice_id").forecast_date_collected.nunique().max() == 1, \
        'Multiple forecast dates per collected invoice'
    # Open invoices can be used in training.
    invoices_with_payments['forecast_date_uncollected'] = _select_forecast_date(
        invoices_with_payments.invoice_id, invoices_with_payments.invoice_date, invoices_with_payments.final_date_open)
    assert invoices_with_payments.groupby("invoice_id").forecast_date_uncollected.nunique().max() == 1, \
        'Multiple forecast dates per uncollected invoice'
    # should be the same date ranges for both options
    assert (invoices_with_payments.forecast_date_uncollected.agg(['min', 'max']).values ==
            invoices_with_payments.forecast_date_collected.agg(['min', 'max']).values).max()
    invoices_with_payments['forecast_date'] = invoices_with_payments.forecast_date_collected.fillna(
            invoices_with_payments.forecast_date_uncollected)
    return invoices_with_payments


def _post_process_invoice_outcomes(invoices_with_payments: pandas.DataFrame) -> (pandas.DataFrame, pandas.DataFrame):
    """ Process, validate, filter, and assign collection date to invoices with payments.
     Sample forecast dates between invoice date and collection date (if present) per invoice for training."""
    # invoice is collected if/when payments accumulate to the invoice amount in the original currency.
    invoices_with_payments = invoices_with_payments.merge(
        invoices_with_payments.loc[invoices_with_payments.amount_pmt_pct_cum == 1, ['invoice_id', 'transaction_date']]
        .rename(columns={"transaction_date": "collected_date"}), on="invoice_id", how="left")
    assert invoices_with_payments.groupby("invoice_id").collected_date.nunique().max() == 1, \
        'Multiple collected dates per invoice'
    invoices_with_payments = _assign_open_forecast_date(invoices_with_payments)
    # data is sorted by invoice and transaction date.
    invoices = invoices_with_payments.drop_duplicates(subset=['invoice_id', 'forecast_date'], keep='last').copy()
    assert invoices.invoice_id.value_counts().max() == 1, 'Multiple forecast dates per invoice'
    invoices['final_remaining_inv_pct'] = 1 - invoices.amount_pmt_pct_cum.fillna(0)
    filters = OrderedDict()
    filters['Collected < opened'] = invoices.query("collected_date<invoice_date")
    filters['Collected, not cleared'] = invoices[(invoices.collected_date.isnull() is False)
                                                 & (invoices.status != 'CLEARED')]
    filters['Cleared < collected'] = invoices.query("cleared_date<collected_date")
    filters['Opened outside of collections date range: could be missing payments'] = \
        invoices[((invoices.invoice_date < invoices.collected_date.min()) |
                  (invoices.invoice_date > invoices.collected_date.max()))]
    filters['NZD currency: not enough rows'] = invoices.query("currency=='NZD'")
    data, filter_stats = apply_filters(filters, invoices)
    return data, filter_stats


def _get_h2o_training_data(feature_data: pandas.DataFrame) -> OrderedDict[str: h2o.H2OFrame]:
    """Given a set of featurized invoices, return h2o data frames split sequentially into training, testing, and
    validation based on forecast date, weighted by the invoice amount's percentage of the client company's total."""
    normalized_feature_data = _normalize_by_company(feature_data)
    normalized_feature_data['inv_company_weight'] = normalized_feature_data.inv_pct_of_company_total \
        * normalized_feature_data.invoice_id.nunique() / normalized_feature_data.company_id.nunique()
    months_to_final_state = months_between(
        normalized_feature_data.forecast_date,
        normalized_feature_data.collected_date.fillna(normalized_feature_data.final_date_open))
    normalized_feature_data['month_collected'] = months_to_final_state + 1
    normalized_feature_data['collected_per_month'] = \
        (normalized_feature_data.remaining_inv_pct - normalized_feature_data.final_remaining_inv_pct) \
        / normalized_feature_data.month_collected
    assert (normalized_feature_data.collected_per_month.isnull()).sum() == 0, 'Collection rate not populated'
    normalized_feature_data['forecast_date_fold'] = (normalized_feature_data.forecast_date.rank(pct=True) * 6).round()
    id_columns_h2o = [col for col in ID_COLUMNS if col in feature_data.columns]
    h2o.init(nthreads=-1, max_mem_size=12)
    invoices_to_model_h2o = h2o.H2OFrame(normalized_feature_data.select_dtypes(exclude='datetime'),
                                         column_types=dict(zip(id_columns_h2o, ["string"] * len(id_columns_h2o))))
    h2o_frames = OrderedDict()
    # time-based split: cross-validating on future data relative to what is being trained
    h2o_frames['train'] = invoices_to_model_h2o[invoices_to_model_h2o['forecast_date_fold'] <= 3]
    h2o_frames['test'] = invoices_to_model_h2o[(invoices_to_model_h2o['forecast_date_fold'] > 3)
                                                     & (invoices_to_model_h2o['forecast_date_fold'] <= 4)]
    h2o_frames['validation'] = invoices_to_model_h2o[invoices_to_model_h2o['forecast_date_fold'] > 4]
    return h2o_frames


def get_training_data_from_csvs() -> (OrderedDict[str: h2o.H2OFrame], pandas.DataFrame):
    """Pre-filter raw CSV test data, preprocess, create outcome variables, sample forecast dates,
    featurize inputs, and return train/test/validation data with a report summarizing all the data filters."""
    invoices, payments = get_csv_test_data()
    payments_training_filters = OrderedDict()
    # last month of payments data is incomplete and not from a representative period in the month
    payments_training_filters['Incomplete month (2021-05)'] = payments[payments.transaction_date >= '2021-5-1']
    payments_to_model, training_payments_filter_stats = apply_filters(payments_training_filters, payments)
    training_payments_filter_stats = pandas.concat([training_payments_filter_stats], names=['Dataset'],
                                                   keys=['payments'])
    invoices_with_payments, preprocess_filter_stats = preprocess_invoices_with_payments(invoices, payments_to_model)
    invoices_to_model, post_process_filter_stats = _post_process_invoice_outcomes(invoices_with_payments)
    post_process_filter_stats = pandas.concat([post_process_filter_stats], names=['Dataset'],
                                              keys=['preprocessed invoices'])
    filter_stats = pandas.concat([training_payments_filter_stats, preprocess_filter_stats, post_process_filter_stats],
                                 names=['Step Type'], keys=['Filtering', 'Pre-processing', 'Filtering'])
    h2o_frames = _get_h2o_training_data(feature_engineering(invoices_to_model))
    return h2o_frames, filter_stats


def train_model(train: h2o.H2OFrame, test: h2o.H2OFrame, predictors: list[str] = ['due_per_month'],
                metric: str = 'mae', y: str = 'collected_per_month', distribution: str = 'huber',
                max_runtime_secs: int = 60, exclude_algos: list[str] = ['StackedEnsemble'], nfolds: int = 0,
                stopping_tolerance: float = 0.01) -> h2o.estimators.H2OEstimator:
    """Given training and testing h2oframes, list of predictors, outcome variable, outcome distribution,
    and ML metric, return a trained H2O model."""
    # some hyperparameter tuning options are addressed by using AutoML and specifying sort and stopping metrics.
    aml = H2OAutoML(max_runtime_secs=max_runtime_secs, distribution=distribution, exclude_algos=exclude_algos,
                    sort_metric=metric, stopping_metric=metric if metric != 'r2' else 'mae',
                    stopping_tolerance=stopping_tolerance, nfolds=nfolds)
    aml_model = aml.train(training_frame=train, blending_frame=test, x=predictors, y=y,
                          weights_column='inv_company_weight')
    return aml_model


def _test_training_on_csvs(metric: str) -> \
        (h2o.estimators.H2OEstimator, OrderedDict[str: h2o.H2OFrame], pandas.DataFrame):
    """Test module on local CSV data."""
    h2o_frames, filter_stats_csv = get_training_data_from_csvs()
    params = dict(metric=metric, y='collected_per_month', distribution='huber', predictors=['due_per_month'],
                  max_runtime_secs=60)
    uni_variate_model = train_model(h2o_frames['train'], h2o_frames['test'], **params)
    return uni_variate_model, h2o_frames, filter_stats_csv


if __name__ == "__main__":
    pandas.set_option('expand_frame_repr', False)
    ml_metric = 'mae'
    baseline_model, split_h2o_frames, training_filter_stats = _test_training_on_csvs(ml_metric)
    print(training_filter_stats)
    for split in split_h2o_frames.keys():
        h2o_frame = split_h2o_frames[split]
        print(f"Baseline model {ml_metric} on {split} data: {baseline_model.model_performance(h2o_frame)[ml_metric]}")
