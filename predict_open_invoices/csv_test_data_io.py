import pandas

DATA_FOLDER = '../data_analysis/data'
DATE_FORMAT = '%Y-%m-%d' # truncate datetimes to dates
ID_COLUMNS = ["id", "company_id", "invoice_id", "account_id", "customer_id"]
ID_COLUMN_TYPES = dict(zip(ID_COLUMNS, [str] * len(ID_COLUMNS)))


def get_csv_test_data() -> (pandas.DataFrame, pandas.DataFrame):
    """Get CSV data as properly formatted pandas dataframes"""
    invoices = pandas.read_csv(DATA_FOLDER + '/invoice.csv', na_values='inf', dtype=ID_COLUMN_TYPES,
                               parse_dates=['invoice_date', 'due_date', 'cleared_date'], date_format=DATE_FORMAT)
    assert invoices.__len__() == 113085, "Test invoices data has been modified. Future checks will not be valid"
    payments = pandas.read_csv(DATA_FOLDER + '/invoice_payments.csv', na_values='inf', dtype=ID_COLUMN_TYPES,
                               parse_dates=['transaction_date'], date_format=DATE_FORMAT)
    assert payments.__len__() == 111623, "Test payments data has been modified. Future checks will not be valid"
    return invoices, payments
