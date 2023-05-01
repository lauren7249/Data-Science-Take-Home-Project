import pandas
import os
import glob
import h2o
from predict_open_invoices.csv_test_data_io import get_csv_test_data
from predict_open_invoices.pre_processing import preprocess_invoices_with_payments
from predict_open_invoices.feature_engineering import feature_engineering
h2o.init(nthreads=-1, max_mem_size=12)


def predict_collection_date(invoice_features: pandas.DataFrame, model: h2o.estimators.H2OEstimator) -> pandas.DataFrame:
    invoice_features_h2o = h2o.H2OFrame(invoice_features)
    return model.predict(invoice_features_h2o)


def test_prediction_on_open_invoices(invoices_raw: pandas.DataFrame, payments_raw: pandas.DataFrame) \
        -> pandas.DataFrame:
    """Given raw input datasets, return invoices with feature data and predictions"""
    invoices_with_payments, preprocess_filter_stats = preprocess_invoices_with_payments(invoices_raw, payments_raw)
    open_invoices_with_payments = invoices_with_payments.query("status=='OPEN'").copy()
    open_invoices_with_payments['forecast_date'] = payments_raw.transaction_date.max()
    open_invoices_feature_data = feature_engineering(open_invoices_with_payments.query("forecast_date>=invoice_date"))
    assert open_invoices_feature_data.invoice_id.value_counts().max() == 1, 'Duplicates per open invoice'
    list_of_files = glob.glob('trained_models/*')  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    model = h2o.load_model(latest_file)
    return predict_collection_date(open_invoices_feature_data, model)


if __name__ == "__main__":
    pandas.set_option('expand_frame_repr', False)
    invoices, payments = get_csv_test_data()
    open_invoice_predictions = test_prediction_on_open_invoices(invoices, payments)
    print(open_invoice_predictions)

