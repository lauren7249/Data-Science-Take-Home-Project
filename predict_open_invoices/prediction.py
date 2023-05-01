import pandas
from predict_open_invoices.utils import apply_filters
from predict_open_invoices.csv_test_data_io import get_csv_test_data
from predict_open_invoices.pre_processing import preprocess_invoices_with_payments
from predict_open_invoices.feature_engineering import feature_engineering


def predict_collection_date(invoice_features: pandas.DataFrame):
    return invoice_features


def test_prediction_on_open_invoices(invoices: pandas.DataFrame, payments: pandas.DataFrame):
    invoices_with_payments, preprocess_filter_stats = preprocess_invoices_with_payments(invoices, payments)
    open_invoices_with_payments = invoices_with_payments.query("status=='OPEN'")
    open_invoices_with_payments['forecast_date'] = payments.transaction_date.max()
    open_invoices_feature_data = feature_engineering(open_invoices_with_payments.query("forecast_date>=invoice_date"))
    assert open_invoices_feature_data.invoice_id.value_counts().max() == 1, 'Duplicates per open invoice'
    return predict_collection_date(open_invoices_feature_data)


if __name__ == "__main__":
    pandas.set_option('expand_frame_repr', False)
    invoices, payments = get_csv_test_data()
    open_invoice_predictions = test_prediction_on_open_invoices(invoices, payments)
    print(open_invoice_predictions)

