import pandas
from collections import OrderedDict


def months_between(start_date: pandas.Series, end_date: pandas.Series):
    months = (end_date.dt.to_period('M') - start_date.dt.to_period('M'))
    months = months.map(lambda m: m.n if not pandas.isnull(m) else None)
    return months


def apply_filters(filters: OrderedDict, apply_to_df: pandas.DataFrame):
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
    assert payments.invoice_id.count() == payments.__len__(), 'Missing invoice ids'
    assert payments.transaction_date.count() == payments.__len__(), 'Missing transaction dates'
    assert payments.root_exchange_rate_value.count() == payments.__len__(), 'Missing exchange rates'
    assert payments.root_exchange_rate_value.min() > 0, 'Exchange rates <= 0'
    assert payments.amount.min() > 0, 'Amounts <= 0'
    assert (payments.amount.isnull() == payments.converted_amount.isnull()).min() == 1, \
        'Converted amounts populated inconsistently from amounts'
    payments_prepared = payments.drop(columns=['company_id', 'converted_amount'], errors='ignore')
    exclude_payments = OrderedDict({'Missing Amount': payments_prepared.loc[payments_prepared.amount.isnull()]})
    return apply_filters(exclude_payments, payments_prepared)


def prepare_invoices(invoices: pandas.DataFrame, date_range: pandas.DatetimeIndex):
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
    invoices_prepared['months_allowed'] = months_between(invoices_prepared.invoice_date, invoices_prepared.due_date)
    exclude_invoices = OrderedDict()
    exclude_invoices['Missing due date'] = invoices_prepared.loc[invoices_prepared.due_date.isnull()]
    exclude_invoices['Opened outside of Payment data time period'] = invoices_prepared.loc[
        (invoices_prepared.invoice_date > date_range.max()) | (invoices_prepared.invoice_date < date_range.min())]
    exclude_invoices['Due before active'] = invoices_prepared.loc[invoices_prepared.months_allowed < 0]
    exclude_invoices['Due over a year after opening '] = invoices_prepared.loc[invoices_prepared.months_allowed > 12]
    return apply_filters(exclude_invoices, invoices_prepared)


def prepare_raw_inputs(invoices: pandas.DataFrame, payments: pandas.DataFrame):
    """ Takes in invoices and payments and prepares them to be scored at a point in time. Can be open or closed when
     scored. Closed invoices would be used for model training. """
    payments_prepared, payment_filter_stats = prepare_payments(payments)
    payments_date_range = pandas.date_range(start=payments.transaction_date.min(), end=payments.transaction_date.max(),
                                            periods=2)
    invoices_prepared, invoice_filter_stats = prepare_invoices(invoices, payments_date_range)

    consolidated_data = payments_prepared.merge(invoices_prepared, on="invoice_id", suffixes=('_pmt', '_inv'),
                                                how='outer')
    exclude_consolidation = OrderedDict({
        'Missing invoice data': consolidated_data.loc[consolidated_data.status.isnull()],
        })
    consolidated_data, consolidated_filter_stats = apply_filters(exclude_consolidation, consolidated_data)
    assert consolidated_data.loc[consolidated_data.amount_pmt > consolidated_data.amount_inv].__len__() == 0, \
        'Payment amount > invoice'
    assert consolidated_data.loc[consolidated_data.amount_pmt < 0].__len__() == 0, 'Negative payment amount'

    filter_stats = pandas.concat(
        {"invoices": invoice_filter_stats, "payments": payment_filter_stats, "consolidated": consolidated_filter_stats}
    )
    filter_stats.index.set_names(['Dataset', 'Step Number'], inplace=True)
    return consolidated_data, filter_stats


def test():
    data_folder = '../data_analysis/data'
    date_format = '%Y-%M-%d'
    invoices = pandas.read_csv(data_folder + '/invoice.csv', na_values='inf',
                               parse_dates=['invoice_date', 'due_date', 'cleared_date'], date_format=date_format)
    payments = pandas.read_csv(data_folder + '/invoice_payments.csv', na_values='inf',
                               parse_dates=['transaction_date'], date_format=date_format)

    consolidated_data, filter_stats = prepare_raw_inputs(invoices, payments)
    print(filter_stats)
    print(consolidated_data.columns)
    #assert list(consolidated_data.status.unique()) == ['CLEARED']


if __name__ == "__main__":
    test()
