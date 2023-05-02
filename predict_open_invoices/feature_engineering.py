import pandas
from predict_open_invoices.utils import months_between
from predict_open_invoices.csv_test_data_io import get_csv_test_data
from predict_open_invoices.pre_processing import preprocess_invoices_with_payments


def _add_date_quantities(invoice_point_in_time: pandas.DataFrame):
    """Sub-section of feature engineering"""
    invoice_point_in_time['months_open'] = (
            months_between(invoice_point_in_time.invoice_date, invoice_point_in_time.forecast_date
                           )+1).clip(upper=13, lower=1)
    invoice_point_in_time['month_due'] = (
            months_between(invoice_point_in_time.forecast_date, invoice_point_in_time.due_date
                           )+1).clip(upper=13, lower=1)
    invoice_point_in_time['due_per_month'] = 1 / invoice_point_in_time.month_due.clip(lower=1)
    return invoice_point_in_time


def feature_engineering(invoices_with_payments: pandas.DataFrame) -> pandas.DataFrame:
    """ Prepares preprocessed invoices with payments and forecast dates for model training and scoring.
    Summarizes invoice with payments to one row per invoice after filtering out payments that are in the future relative
    to the forecast date. Assumes that forecasts are generated once at the beginning of the month."""
    assert invoices_with_payments.forecast_date.count() == invoices_with_payments.__len__(), 'Missing forecast date'
    assert (invoices_with_payments.forecast_date < invoices_with_payments.invoice_month).sum() == 0, \
        'Forecast < invoice opened'
    invoices_with_payments = invoices_with_payments.sort_values(by=['invoice_id', 'transaction_date']).copy()
    pmt_columns = [col for col in invoices_with_payments.columns if '_pmt' in col or 'transaction_' in col]
    # last payment before the forecast date for each invoice
    last_prior_payment_state = invoices_with_payments.query("transaction_date<forecast_date or amount_pmt_pct_cum==0")\
        .drop_duplicates(subset='invoice_id', keep='last')[['invoice_id'] + pmt_columns]
    invoice_begin_state = invoices_with_payments.drop(columns=pmt_columns).drop_duplicates()
    invoice_point_in_time = invoice_begin_state.merge(last_prior_payment_state, on="invoice_id", how="left",
                                                      suffixes=('', '_prior'))
    assert invoice_point_in_time.invoice_id.value_counts().max() == 1, 'Inputs not preprocessed as expected'
    invoice_point_in_time = _add_date_quantities(invoice_point_in_time)
    assert invoice_point_in_time.invoice_id.nunique() == invoices_with_payments.invoice_id.nunique(), \
        'Invoices dropped in feature engineering'
    invoice_point_in_time['remaining_inv_pct'] = 1 - invoice_point_in_time.amount_pmt_pct_cum.fillna(0)
    return invoice_point_in_time


def test_feature_engineering():
    """Test feature engineering on CSV test data, using the date the invoice was opened as the date forecasted."""
    invoices, payments = get_csv_test_data()
    invoices_with_payments, preprocess_filter_stats = preprocess_invoices_with_payments(invoices, payments)
    # test using the invoice date as an arbitrary point in time to generate the data.
    invoices_with_payments['forecast_date'] = invoices_with_payments.invoice_date
    feature_data = feature_engineering(invoices_with_payments)
    return feature_data, preprocess_filter_stats


if __name__ == "__main__":
    pandas.set_option('expand_frame_repr', False)
    data, filter_stats = test_feature_engineering()
    print(filter_stats)
    print(data.columns)
    print(data.shape)

