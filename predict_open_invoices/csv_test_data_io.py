import pandas
from predict_open_invoices import ID_COLUMNS
DATA_FOLDER = '../data_analysis/data'
# truncate datetimes to dates
DATE_FORMAT = '%Y-%m-%d'
ID_COLUMN_TYPES = dict(zip(ID_COLUMNS, [str] * len(ID_COLUMNS)))


def get_csv_test_data() -> (pandas.DataFrame, pandas.DataFrame):
    """Get CSV data as properly formatted pandas dataframes"""
    invoices = pandas.read_csv(DATA_FOLDER + '/invoice.csv', na_values='inf', dtype=ID_COLUMN_TYPES,
                               parse_dates=['invoice_date', 'due_date', 'cleared_date'], date_format=DATE_FORMAT)
    assert invoices.__len__() == 113085, "Rows in invoices test CSV have been modified. Future checks will not be valid"
    assert invoices.cleared_date.isnull().sum() == 0, "Columns in invoices test CSV have been modified"
    payments = pandas.read_csv(DATA_FOLDER + '/invoice_payments.csv', na_values='inf', dtype=ID_COLUMN_TYPES,
                               parse_dates=['transaction_date'], date_format=DATE_FORMAT)
    assert payments.__len__() == 111623, "Rows in payments test CSV have been modified. Future checks will not be valid"
    return invoices, payments
