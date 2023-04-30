import pandas
from _collections import OrderedDict
from predict_open_invoices.csv_test_data_io import get_csv_test_data
from predict_open_invoices.pre_processing import preprocess_invoices_with_payments, apply_filters


def train_cash_flow_model(invoices_to_model: pandas.DataFrame):
    return invoices_to_model


def train_on_csv_test_data():
    """ Get CSV test data"""
    invoices, payments = get_csv_test_data()
    # test using the invoice date as an arbitrary point in time to generate the data.
    invoices['forecast_month'] = invoices.invoice_date.dt.to_period('M')
    payments_training_filters = OrderedDict()
    # last month of payments data is incomplete and not from a representative period in the month
    payments_training_filters['Incomplete month (2021-05)'] = payments[payments.transaction_date >= '2021-5-1']
    payments_to_model, training_payments_filter_stats = apply_filters(payments_training_filters, payments)
    training_payments_filter_stats = pandas.concat([training_payments_filter_stats], names=['Dataset'],
                                                   keys=['payments'])
    preprocessed_invoices, preprocess_filter_stats = preprocess_invoices_with_payments(invoices, payments_to_model)
    invoices_training_filters = OrderedDict()
    invoices_training_filters['Opened outside of payment data time period: could be missing payments'] = \
        preprocessed_invoices.loc[(
                (preprocessed_invoices.invoice_month < preprocessed_invoices.transaction_month.min()) |
                (preprocessed_invoices.invoice_month >= preprocessed_invoices.transaction_month.max())
        )]
    invoices_training_filters['Cleared < opened'] = preprocessed_invoices.query("cleared_date<invoice_date")
    # almost all invoices are cleared within a year.
    invoices_to_model, training_invoices_filter_stats = apply_filters(invoices_training_filters, preprocessed_invoices)
    training_invoices_filter_stats = pandas.concat([training_invoices_filter_stats], names=['Dataset'],
                                                   keys=['preprocessed invoices'])
    training_filter_stats = pandas.concat([training_payments_filter_stats, preprocess_filter_stats,
                                           training_invoices_filter_stats], names=['Step Type'],
                                          keys=['Filtering', 'Pre-processing', 'Filtering'])
    train_cash_flow_model(invoices_to_model)
    return training_filter_stats


if __name__ == "__main__":
    pandas.set_option('expand_frame_repr', False)
    train_filter_stats = train_on_csv_test_data()
    print(train_filter_stats)

