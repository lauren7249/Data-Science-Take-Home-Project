from _collections import OrderedDict
import pandas
import random
import numpy
import h2o
from h2o.automl import H2OAutoML
from predict_open_invoices import ID_COLUMNS
from predict_open_invoices.utils import months_between
from predict_open_invoices.csv_test_data_io import get_csv_test_data
from predict_open_invoices.pre_processing import preprocess_invoices_with_payments, apply_filters
from predict_open_invoices.feature_engineering import feature_engineering


def select_forecast_date(invoice_id, invoice_date, max_forecast_date, forecast_frequency: str = 'M',
                         first_transaction_date: pandas.Timestamp = None):
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


select_forecast_date = numpy.vectorize(select_forecast_date)


def assign_open_forecast_date(invoices_with_payments: pandas.DataFrame):
    """Add a randomly sampled forecast date while the invoice was open."""
    invoices_with_payments['forecast_date_collected'] = select_forecast_date(
        invoices_with_payments.invoice_id, invoices_with_payments.invoice_date, invoices_with_payments.collected_date)
    assert invoices_with_payments.groupby("invoice_id").forecast_date_collected.nunique().max() == 1, \
        'Multiple forecast dates per collected invoice'
    # Open invoices can be used in training.
    invoices_with_payments['forecast_date_uncollected'] = select_forecast_date(
        invoices_with_payments.invoice_id, invoices_with_payments.invoice_date, invoices_with_payments.final_date_open)
    assert invoices_with_payments.groupby("invoice_id").forecast_date_uncollected.nunique().max() == 1, \
        'Multiple forecast dates per uncollected invoice'
    # should be the same date ranges for both options
    assert (invoices_with_payments.forecast_date_uncollected.agg(['min', 'max']).values ==
            invoices_with_payments.forecast_date_collected.agg(['min', 'max']).values).max()
    invoices_with_payments['forecast_date'] = invoices_with_payments.forecast_date_collected.fillna(
            invoices_with_payments.forecast_date_uncollected)
    return invoices_with_payments


def postprocess_invoice_outcomes(invoices_with_payments: pandas.DataFrame):
    """ Process, validate, filter, and assign collection date to invoices with payments.
     Sample forecast dates between invoice date and collection date (if present) per invoice for training."""
    # invoice is collected if/when payments accumulate to the invoice amount in the original currency.
    invoices_with_payments = invoices_with_payments.merge(
        invoices_with_payments.loc[invoices_with_payments.amount_pmt_pct_cum == 1, ['invoice_id', 'transaction_date']]
        .rename(columns={"transaction_date": "collected_date"}), on="invoice_id", how="left")
    assert invoices_with_payments.groupby("invoice_id").collected_date.nunique().max() == 1, \
        'Multiple collected dates per invoice'
    invoices_with_payments = assign_open_forecast_date(invoices_with_payments)
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
    data, filter_stats = apply_filters(filters, invoices)

    return data, filter_stats


def train_cash_flow_model(feature_data: pandas.DataFrame):
    assert feature_data.inv_pct_of_company_total.sum() == feature_data.company_id.nunique(), \
        'Company amount normalization does not sum to 1 per company'
    feature_data['inv_company_weight'] = feature_data.inv_pct_of_company_total \
        * feature_data.invoice_id.nunique() / feature_data.company_id.nunique()
    months_to_final_state = months_between(feature_data.forecast_date,
                                           feature_data.collected_date.fillna(feature_data.final_date_open))
    feature_data['collected_per_month'] = (feature_data.remaining_inv_pct -
                                           feature_data.final_remaining_inv_pct) / (months_to_final_state + 1)
    # always has a collection rate
    assert (feature_data.collected_per_month.isnull()).sum() == 0, 'Collection rate not populated'
    feature_data['forecast_date_fold'] = (feature_data.forecast_date.rank(pct=True) * 6).round()
    h2o.init(nthreads=-1,  max_mem_size=12)
    id_columns_h2o = [col for col in ID_COLUMNS if col in feature_data.columns]
    invoices_to_model_h2o = h2o.H2OFrame(feature_data,
                                         column_types=dict(zip(id_columns_h2o, ["string"] * len(id_columns_h2o))))
    # time-based split: cross-validating on future data relative to what is being trained
    train = invoices_to_model_h2o[invoices_to_model_h2o['forecast_date_fold'] <= 3]
    blend = invoices_to_model_h2o[(invoices_to_model_h2o['forecast_date_fold'] > 3)
                                  & (invoices_to_model_h2o['forecast_date_fold'] <= 4)]
    valid = invoices_to_model_h2o[invoices_to_model_h2o['forecast_date_fold'] > 4]
    y_numeric = 'collected_per_month'
    x = ['months_allowed', 'amount_inv', 'inv_pct_of_company_total', 'currency', 'months_open', 'due_per_month',
         'remaining_inv_pct']
    # huber is a bi-modal distribution
    # hyperparameter tuning is addressed by using AutoML and specifying sort and stopping metrics.
    # train, blend, and validation dataframes are binned sequentially by forecast month
    # this enforces the time-based split during hyperparameter tuning.
    aml = H2OAutoML(max_runtime_secs=60, distribution='huber', sort_metric='mae', stopping_metric='mae',
                    stopping_tolerance=0.01)
    aml_model = aml.train(x=x, y=y_numeric, training_frame=train, blending_frame=blend, validation_frame=valid,
                          weights_column='inv_company_weight')
    return aml_model


def train_on_csv_test_data():
    """ Get CSV test data"""
    invoices, payments = get_csv_test_data()
    payments_training_filters = OrderedDict()
    # last month of payments data is incomplete and not from a representative period in the month
    payments_training_filters['Incomplete month (2021-05)'] = payments[payments.transaction_date >= '2021-5-1']
    payments_to_model, training_payments_filter_stats = apply_filters(payments_training_filters, payments)
    training_payments_filter_stats = pandas.concat([training_payments_filter_stats], names=['Dataset'],
                                                   keys=['payments'])
    invoices_with_payments, preprocess_filter_stats = preprocess_invoices_with_payments(invoices, payments_to_model)
    invoices_to_model, postprocess_filter_stats = postprocess_invoice_outcomes(invoices_with_payments)
    postprocess_filter_stats = pandas.concat([postprocess_filter_stats], names=['Dataset'],
                                             keys=['preprocessed invoices'])
    training_filter_stats = pandas.concat([training_payments_filter_stats, preprocess_filter_stats,
                                           postprocess_filter_stats], names=['Step Type'],
                                          keys=['Filtering', 'Pre-processing', 'Filtering'])
    feature_data = feature_engineering(invoices_to_model)
    model = train_cash_flow_model(feature_data)
    return model, training_filter_stats


if __name__ == "__main__":
    pandas.set_option('expand_frame_repr', False)
    trained_model, train_filter_stats = train_on_csv_test_data()
    print(train_filter_stats)

