import pandas
from collections import OrderedDict
from predict_open_invoices.utils import apply_filters, months_between
from predict_open_invoices.csv_test_data_io import get_csv_test_data


def prepare_payments(payments: pandas.DataFrame) -> (pandas.DataFrame, pandas.DataFrame):
    """ Validate raw payments data and filter out rows that will not be scored or used in model training."""
    assert payments.invoice_id.count() == payments.__len__(), 'Missing invoice ids'
    assert payments.transaction_date.count() == payments.__len__(), 'Missing transaction dates'
    assert payments.root_exchange_rate_value.count() == payments.__len__(), 'Missing exchange rates'
    assert payments.root_exchange_rate_value.min() > 0, 'Exchange rates <= 0'
    assert payments.amount.min() > 0, 'Amounts <= 0'
    assert (payments.amount.isnull() == payments.converted_amount.isnull()).min() == 1, \
        'Converted amounts populated inconsistently from amounts'
    exclude_payments = OrderedDict({'Missing Amount': payments[payments.amount.isnull()]})
    return apply_filters(exclude_payments, payments)


def prepare_invoices(invoices: pandas.DataFrame) -> (pandas.DataFrame, pandas.DataFrame):
    """ Validate raw invoices data and filter out rows that will not be scored or used in model training."""
    assert invoices.id.count() == invoices.__len__(), 'Missing IDs'
    assert invoices.invoice_date.count() == invoices.__len__(), 'Missing invoice dates'
    assert invoices.status.count() == invoices.__len__(), 'Missing statuses'
    assert invoices.amount_inv.count() == invoices.__len__(), 'Missing amounts'
    assert invoices.root_exchange_rate_value.count() == invoices.__len__(), 'Missing exchange rates'
    assert invoices.currency.count() == invoices.__len__(), 'Missing currencies'
    assert invoices.company_id.count() == invoices.__len__(), 'Missing company IDs'
    assert invoices.customer_id.count() == invoices.__len__(), 'Missing customer IDs'
    assert invoices.root_exchange_rate_value.min() > 0, 'Exchange rates <= 0'
    assert invoices.amount_inv.min() > 0, 'Amounts <= 0'
    invoices = invoices.rename(columns={"id": "invoice_id", "amount_inv": "amount"})\
        .drop(columns=['account_id'], errors='ignore')
    invoices.loc[invoices.status == 'OPEN', 'cleared_date'] = None
    invoices['invoice_month'] = invoices.invoice_date.dt.to_period('M').dt.to_timestamp()
    invoices['due_month'] = invoices.due_date.dt.to_period('M').dt.to_timestamp()
    invoices['months_allowed'] = months_between(invoices.invoice_month, invoices.due_month)
    exclude_invoices = OrderedDict()
    exclude_invoices['Missing due date'] = invoices[invoices.due_date.isnull()]
    exclude_invoices['Due before opened'] = invoices[invoices.due_date < invoices.invoice_date]
    exclude_invoices['Due over 3 months after opened'] = invoices[invoices.months_allowed > 3]
    # filter out dates with high variation - TODO: Filter stats
    exclude_invoices['Due before 2011-10'] = invoices.query("due_date<'2011-10-01'")
    exclude_invoices['USD exchange rate out of range [0.7,1.3]'] = invoices[
        (invoices.currency == 'USD') & (~invoices.root_exchange_rate_value.between(0.7, 1.3))]
    exclude_invoices['Cleared < opened'] = invoices.query("cleared_date<invoice_date")
    months_to_clear = months_between(invoices.invoice_date, invoices.cleared_date)
    exclude_invoices['Cleared 13+ months after opened'] = invoices[months_to_clear > 12]
    invoices_filtered, filter_stats = apply_filters(exclude_invoices, invoices)
    return invoices_filtered, filter_stats


def combine_invoices_payments(invoices_prepared: pandas.DataFrame, payments_prepared: pandas.DataFrame) -> (
        pandas.DataFrame, pandas.DataFrame):
    """ Merge, validate, and create combined variables from prepared invoice and payments data.
    Output data has one row per invoice and cumulative amount paid, rounded to 4 decimal places."""
    # consolidate invoices with payments
    invoices_prepared['converted_amount'] = invoices_prepared.amount * invoices_prepared.root_exchange_rate_value
    invoice_payments = invoices_prepared.merge(payments_prepared, on="invoice_id", suffixes=('_inv', '_pmt'),
                                               how='left')
    assert (invoice_payments.amount_pmt > invoice_payments.amount_inv).sum() == 0, 'Payment amount > invoice amount'
    assert invoice_payments.dropna(subset=['company_id_pmt']).query("company_id_pmt!=company_id_inv").__len__() == 0, \
        'Company does not match between payments and invoices'
    last_transaction_date = invoice_payments.transaction_date.max()
    # invoices with no transactions: use payments data end date as date of 0 amount
    invoice_payments.transaction_date = invoice_payments.transaction_date.fillna(last_transaction_date)
    invoice_payments['final_date_open'] = invoice_payments.cleared_date.map(
        lambda cleared_date: min(cleared_date, last_transaction_date)).fillna(last_transaction_date)
    invoice_payments = invoice_payments.rename(columns={"company_id_inv": "company_id"})\
        .drop(columns=['company_id_pmt']).sort_values(by=['invoice_id', 'transaction_date'])
    invoice_payments['amount_pmt_pct'] = (invoice_payments.amount_pmt / invoice_payments.amount_inv)
    # round to eliminate the impact of negligible payments. hence, an invoice is "collected" when paid > 99.99%.
    invoice_payments['amount_pmt_pct_cum'] = invoice_payments.groupby("invoice_id").amount_pmt_pct.cumsum()\
        .fillna(0).round(4)
    # small percent of payments represent overpayments - filter out
    invoice_payments = invoice_payments[invoice_payments.amount_pmt_pct_cum <= 1].copy()
    # dedupe by invoice id and payment date, using the last transaction for each
    invoice_payments.drop_duplicates(subset=['invoice_id', 'transaction_date'], keep='last', inplace=True)
    # dedupe by invoice id and cumulative amount paid, using the first transaction for each (dupes are very rare)
    invoice_payments.drop_duplicates(subset=['invoice_id', 'amount_pmt_pct_cum'], keep='first', inplace=True)
    return invoice_payments


def preprocess_invoices_with_payments(invoices: pandas.DataFrame, payments: pandas.DataFrame) -> \
        (pandas.DataFrame, OrderedDict):
    """ Takes raw invoices and payments separately and prepares them for feature engineering.
    Output data has one row per payment and cumulative amount paid, rounded to 4 decimal places."""
    # confirm invoices are a superset of payments before filtering.
    assert len(set(payments.invoice_id) - set(invoices.id)) == 0, "Not all payments have invoice data"
    # validate and filter inputs separately
    payments_prepared, payment_filter_stats = prepare_payments(payments)
    invoices_prepared, invoice_filter_stats = prepare_invoices(invoices)
    # compare and consolidate inputs
    data = combine_invoices_payments(invoices_prepared, payments_prepared)
    filter_stats_dict = OrderedDict({"payments": payment_filter_stats, "invoices": invoice_filter_stats})
    filter_stats = pandas.concat(filter_stats_dict, names=['Dataset', 'Step Num'])
    return data, filter_stats


def test_pre_processing():
    invoices, payments = get_csv_test_data()
    invoices_with_payments, filter_stats = preprocess_invoices_with_payments(invoices, payments)
    return invoices_with_payments, filter_stats


if __name__ == "__main__":
    pandas.set_option('expand_frame_repr', False)
    preprocessed_data, preprocess_filter_stats = test_pre_processing()
    print(preprocess_filter_stats)
    print(preprocessed_data.columns)
    print(preprocessed_data.shape)
