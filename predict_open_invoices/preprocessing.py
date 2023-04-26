import pandas
from collections import OrderedDict


def months_between(start_date: pandas.Series, end_date: pandas.Series):
    """ Calculate number of months between two dates in a pandas Dataframe """
    months = (end_date.dt.to_period('M') - start_date.dt.to_period('M'))
    months = months.map(lambda m: m.n if not pandas.isnull(m) else None)
    return months


def apply_filters(filters: OrderedDict, apply_to_df: pandas.DataFrame):
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


def prepare_payments(payments: pandas.DataFrame):
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


def prepare_invoices(invoices: pandas.DataFrame, date_range: pandas.DatetimeIndex):
    """ Validate and filter invoices data. Date_range is applied to the invoice month. """
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

    invoices_prepared = invoices.rename(columns={"id": "invoice_id", "amount_inv": "amount"})
    invoices_prepared.drop(columns=['account_id'], inplace=True, errors='ignore')
    invoices_prepared['invoice_month'] = invoices_prepared.invoice_date.dt.to_period('M').dt.to_timestamp()
    invoices_prepared['due_month'] = invoices_prepared.due_date.dt.to_period('M').dt.to_timestamp()
    invoices_prepared['months_allowed'] = months_between(invoices_prepared.invoice_month, invoices_prepared.due_month)
    exclude_invoices = OrderedDict()
    exclude_invoices['Missing due date'] = invoices_prepared.loc[invoices_prepared.due_date.isnull()]
    exclude_invoices['Opened outside of Payment data time period'] = invoices_prepared.loc[
        (invoices_prepared.invoice_month > date_range.max()) | (invoices_prepared.invoice_month < date_range.min())]
    exclude_invoices['Due before opened'] = invoices_prepared.loc[invoices_prepared.due_month <
                                                                  invoices_prepared.invoice_month]
    exclude_invoices['Due over 3 months after opened'] = invoices_prepared.loc[invoices_prepared.months_allowed > 3]
    exclude_invoices['Forecast month < invoice month'] = invoices_prepared.loc[
        invoices_prepared.forecast_month < invoices_prepared.invoice_month.dt.to_period('M')]
    return apply_filters(exclude_invoices, invoices_prepared)


def prepare_consolidated_cash_data(consolidated_data: pandas.DataFrame):
    """ Validate and filter consolidated cash data """
    exclude_consolidation = OrderedDict({
        'Missing invoice data': consolidated_data.loc[consolidated_data.amount_inv.isnull()],
        'Due before October 2011': consolidated_data.loc[consolidated_data.due_month < '2011-10-01']
        })
    consolidated_data, consolidated_filter_stats = apply_filters(exclude_consolidation, consolidated_data)

    assert consolidated_data.loc[consolidated_data.amount_pmt > consolidated_data.amount_inv].__len__() == 0, \
        'Payment amount > invoice'
    assert consolidated_data.loc[consolidated_data.amount_pmt < 0].__len__() == 0, 'Negative payment amount'
    consolidated_data = feature_engineering(consolidated_data)
    return consolidated_data, consolidated_filter_stats


def feature_engineering(consolidated_data: pandas.DataFrame):
    consolidated_data.sort_values(by=['invoice_id', 'transaction_date'], inplace=True)
    consolidated_data.drop_duplicates(subset='invoice_id', keep='last')
    #normalize by company
    consolidated_data['converted_amount_inv'] = (consolidated_data.amount_inv *
                                                 consolidated_data.root_exchange_rate_value_inv)
    totals_by_company = consolidated_data.groupby("company_id", as_index=False).converted_amount_inv.sum()
    consolidated_data = consolidated_data.merge(totals_by_company,on="company_id", suffixes=('', '_company'))
    inv_pct_of_company_total = consolidated_data.converted_amount_inv/consolidated_data.converted_amount_inv_company
    consolidated_data['inv_pct_of_company_total'] = inv_pct_of_company_total
    consolidated_data.drop(columns=["converted_amount_inv", "cleared_date"], inplace=True, errors='ignore')
    #date quantities
    consolidated_data['month_billing'] = (
            consolidated_data.forecast_month - consolidated_data.invoice_month.dt.to_period('M')
    ).map(lambda m: m.n+1).clip(upper=13, lower=1)
    consolidated_data['month_due'] = (
            consolidated_data.due_month.dt.to_period('M') - consolidated_data.forecast_month
    ).map(lambda m: m.n+1).clip(upper=13, lower=1)
    consolidated_data['due_per_month'] = 1 / consolidated_data.month_due
    consolidated_data.forecast_month = consolidated_data.forecast_month.dt.to_timestamp()
    return consolidated_data


def prepare_raw_inputs(invoices: pandas.DataFrame, payments: pandas.DataFrame):
    """ Takes in invoices and payments and prepares them to be scored at a point in time. Can be open or closed when
     scored. Closed invoices would be used for model training. """
    # validate and filter input datasets
    payments_prepared, payment_filter_stats = prepare_payments(payments)
    payments_date_range = pandas.date_range(start=payments_prepared.transaction_month.min(),
                                            end=payments_prepared.transaction_month.max(), periods=2)
    invoices_prepared, invoice_filter_stats = prepare_invoices(invoices, payments_date_range)
    # consolidate invoices with payments
    consolidated_data = invoices_prepared.merge(payments_prepared, on="invoice_id", suffixes=('_inv', '_pmt'),
                                                how='outer')
    # invoices with no transactions: use payments data end date as date of 0 amount
    consolidated_data.transaction_month = consolidated_data.transaction_month.fillna(payments_date_range[-1])
    # filter and validate consolidated data
    consolidated_data, consolidated_filter_stats = prepare_consolidated_cash_data(consolidated_data)
    filter_stats = pandas.concat(
        {"invoices": invoice_filter_stats, "payments": payment_filter_stats, "consolidated": consolidated_filter_stats}
    )
    filter_stats.index.set_names(['Dataset', 'Step Number'], inplace=True)
    return consolidated_data, filter_stats


def test():
    data_folder = '../data_analysis/data'
    date_format = '%Y-%m-%d'
    invoices = pandas.read_csv(data_folder + '/invoice.csv', na_values='inf',
                               parse_dates=['invoice_date', 'due_date', 'cleared_date'], date_format=date_format)
    assert invoices.__len__() == 113085, "Test invoices data has been modified. Future checks will not be valid"
    payments = pandas.read_csv(data_folder + '/invoice_payments.csv', na_values='inf',
                               parse_dates=['transaction_date'], date_format=date_format)
    assert payments.__len__() == 111623, "Test payments data has been modified. Future checks will not be valid"
    # test using the invoice date as the point in time to generate a forecast.
    invoices['forecast_month'] = invoices.invoice_date.dt.to_period('M')
    consolidated_data, filter_stats = prepare_raw_inputs(invoices, payments)
    # data is sorted by invoice and transaction date.
    consolidated_data.drop_duplicates(subset=['invoice_id', 'forecast_month'], inplace=True, keep='last')
    print(filter_stats)
    print(consolidated_data.columns)
    print(consolidated_data.shape)

if __name__ == "__main__":
    test()
