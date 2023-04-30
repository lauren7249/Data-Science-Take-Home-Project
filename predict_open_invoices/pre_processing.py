import pandas
from collections import OrderedDict
from predict_open_invoices.utils import apply_filters, months_between
from predict_open_invoices.csv_test_data_io import get_csv_test_data


def prepare_payments(payments: pandas.DataFrame) -> (pandas.DataFrame, pandas.DataFrame):
    """ Validate and filter payments data """
    assert payments.invoice_id.count() == payments.__len__(), 'Missing invoice ids'
    assert payments.transaction_date.count() == payments.__len__(), 'Missing transaction dates'
    assert payments.root_exchange_rate_value.count() == payments.__len__(), 'Missing exchange rates'
    assert payments.root_exchange_rate_value.min() > 0, 'Exchange rates <= 0'
    assert payments.amount.min() > 0, 'Amounts <= 0'
    assert (payments.amount.isnull() == payments.converted_amount.isnull()).min() == 1, \
        'Converted amounts populated inconsistently from amounts'
    payments_prepared = payments.drop(columns=['company_id', 'converted_amount'], errors='ignore')
    payments_prepared['transaction_month'] = payments_prepared.transaction_date.dt.to_period('M').dt.to_timestamp()
    exclude_payments = OrderedDict({'Missing Amount': payments_prepared.loc[payments_prepared.amount.isnull()]})
    return apply_filters(exclude_payments, payments_prepared)


def prepare_invoices(invoices: pandas.DataFrame) -> (pandas.DataFrame, pandas.DataFrame):
    """ Validate invoices data and filter out rows that will not be scored or used in model training."""
    assert invoices.id.count() == invoices.__len__(), 'Missing IDs'
    assert invoices.invoice_date.count() == invoices.__len__(), 'Missing invoice dates'
    assert invoices.status.count() == invoices.__len__(), 'Missing statuses'
    assert invoices.amount_inv.count() == invoices.__len__(), 'Missing amounts'
    assert invoices.root_exchange_rate_value.count() == invoices.__len__(), 'Missing exchange rates'
    assert invoices.currency.count() == invoices.__len__(), 'Missing currencies'
    assert invoices.company_id.count() == invoices.__len__(), 'Missing company IDs'
    assert invoices.customer_id.count() == invoices.__len__(), 'Missing customer IDs'
    assert invoices.forecast_month.count() == invoices.__len__(), 'Missing forecast month'
    assert invoices.root_exchange_rate_value.min() > 0, 'Exchange rates <= 0'
    assert invoices.amount_inv.min() > 0, 'Amounts <= 0'
    assert (invoices.forecast_month < invoices.invoice_date.dt.to_period('M')).sum() == 0, \
        'Forecast month < invoice month'
    invoices = invoices.rename(columns={"id": "invoice_id", "amount_inv": "amount"})\
        .drop(columns=['account_id'], errors='ignore')
    invoices['invoice_month'] = invoices.invoice_date.dt.to_period('M').dt.to_timestamp()
    invoices['due_month'] = invoices.due_date.dt.to_period('M').dt.to_timestamp()
    invoices['months_allowed'] = months_between(invoices.invoice_month, invoices.due_month)
    exclude_invoices = OrderedDict()
    exclude_invoices['Missing due date'] = invoices.loc[invoices.due_date.isnull()]
    exclude_invoices['Due before opened'] = invoices.loc[invoices.due_date < invoices.invoice_date]
    exclude_invoices['Due over 3 months after opened'] = invoices.loc[invoices.months_allowed > 3]
    invoices_filtered, filter_stats = apply_filters(exclude_invoices, invoices)
    return invoices_filtered, filter_stats


def consolidate_invoices_payments(invoices_prepared: pandas.DataFrame, payments_prepared: pandas.DataFrame) -> (
        pandas.DataFrame, pandas.DataFrame):
    """ Merge and validate prepared invoice and payments data """
    # consolidate invoices with payments
    consolidated_data = invoices_prepared.merge(payments_prepared, on="invoice_id", suffixes=('_inv', '_pmt'),
                                                how='left')
    assert consolidated_data.loc[consolidated_data.amount_pmt > consolidated_data.amount_inv].__len__() == 0, \
        'Payment amount > invoice amount'
    # data is sorted by invoice and transaction date.
    consolidated_data.drop_duplicates(subset=['invoice_id', 'forecast_month'], inplace=True, keep='last')
    return consolidated_data


def feature_engineering(consolidated_data: pandas.DataFrame) -> pandas.DataFrame:
    """ Modifies consolidated invoice and payments data in place """
    # invoices with no transactions: use payments data end date as date of 0 amount
    consolidated_data.transaction_month = consolidated_data.transaction_month.fillna(
        consolidated_data.transaction_month.max())
    consolidated_data.sort_values(by=['invoice_id', 'transaction_date'], inplace=True)
    consolidated_data.drop_duplicates(subset='invoice_id', keep='last')
    # normalize by company
    consolidated_data['converted_amount_inv'] = (consolidated_data.amount_inv *
                                                 consolidated_data.root_exchange_rate_value_inv)
    totals_by_company = consolidated_data.groupby("company_id", as_index=False).converted_amount_inv.sum()
    consolidated_data = consolidated_data.merge(totals_by_company, on="company_id", suffixes=('', '_company'))
    inv_pct_of_company_total = consolidated_data.converted_amount_inv/consolidated_data.converted_amount_inv_company
    consolidated_data['inv_pct_of_company_total'] = inv_pct_of_company_total
    consolidated_data.drop(columns=["converted_amount_inv"], inplace=True, errors='ignore')
    # date quantities
    consolidated_data['month_billing'] = (
            consolidated_data.forecast_month - consolidated_data.invoice_month.dt.to_period('M')
    ).map(lambda m: m.n+1).clip(upper=13, lower=1)
    consolidated_data['month_due'] = (
            consolidated_data.due_month.dt.to_period('M') - consolidated_data.forecast_month
    ).map(lambda m: m.n+1).clip(upper=13, lower=1)
    consolidated_data['due_per_month'] = 1 / consolidated_data.month_due
    consolidated_data.forecast_month = consolidated_data.forecast_month.dt.to_timestamp()
    return consolidated_data


def preprocess_invoices_with_payments(invoices: pandas.DataFrame, payments: pandas.DataFrame) -> \
        (pandas.DataFrame, OrderedDict):
    """ Takes invoices and payments separately and prepares them to be scored at a point in time.
    Invoices can be open or closed when scored."""
    # confirm invoices are a superset of payments before filtering.
    assert len(set(payments.invoice_id) - set(invoices.id)) == 0, "Not all payments have invoice data"
    # validate and filter inputs separately
    payments_prepared, payment_filter_stats = prepare_payments(payments)
    invoices_prepared, invoice_filter_stats = prepare_invoices(invoices)
    # compare and consolidate inputs
    consolidated_data = consolidate_invoices_payments(invoices_prepared, payments_prepared)
    data = feature_engineering(consolidated_data)
    filter_stats_dict = OrderedDict({"payments": payment_filter_stats, "invoices": invoice_filter_stats})
    filter_stats = pandas.concat(filter_stats_dict, names=['Dataset', 'Step Num'])
    return data, filter_stats


def test_pre_processing():
    invoices, payments = get_csv_test_data()
    # test using the invoice date as an arbitrary point in time to generate the data.
    invoices['forecast_month'] = invoices.invoice_date.dt.to_period('M')
    data, filter_stats = preprocess_invoices_with_payments(invoices, payments)
    return data, filter_stats


if __name__ == "__main__":
    pandas.set_option('expand_frame_repr', False)
    preprocessed_data, preprocess_filter_stats = test_pre_processing()
    print(preprocess_filter_stats)
    print(preprocessed_data.columns)
    print(preprocessed_data.shape)
