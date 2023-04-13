import pandas


def process(invoices: pandas.DataFrame, payments: pandas.DataFrame):
    return invoices, payments


def test():
    data_folder = '../data_analysis/data'
    invoices = pandas.read_csv(data_folder + '/invoice.csv')
    payments = pandas.read_csv(data_folder + '/invoice_payments.csv')
    process(invoices, payments)


if __name__ == "__main__":
    test()
