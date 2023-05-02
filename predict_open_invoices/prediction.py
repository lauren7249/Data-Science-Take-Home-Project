import os
import tempfile
import h2o
import pandas
import neptune
from predict_open_invoices.csv_test_data_io import get_csv_test_data
from predict_open_invoices.pre_processing import preprocess_invoices_with_payments
from predict_open_invoices.feature_engineering import feature_engineering
h2o.init(nthreads=-1, max_mem_size=12)
NEPTUNE_PROJECT_NAME = "open-invoices-model"
NEPTUNE_PROJECT = neptune.init_project(project=NEPTUNE_PROJECT_NAME, api_token=os.getenv('NEPTUNE_API_TOKEN'))
NEPTUNE_PROJECT_ID = NEPTUNE_PROJECT['sys/id'].fetch()
NEPTUNE_MODEL_ID = "CSVDATA"
try:
    NEPTUNE_MODEL = neptune.init_model(key=NEPTUNE_MODEL_ID, project=NEPTUNE_PROJECT_NAME,
                                       name="Trained on CSV data.", api_token=os.getenv('NEPTUNE_API_TOKEN'))
except neptune.exceptions.NeptuneModelKeyAlreadyExistsError:
    NEPTUNE_MODEL = neptune.init_model(with_id=f"{NEPTUNE_PROJECT_ID}-{NEPTUNE_MODEL_ID}", project=NEPTUNE_PROJECT_NAME)


def get_best_model(sort_column: str = 'monthly_mape_test') -> h2o.estimators.H2OEstimator:
    NEPTUNE_MODEL.sync()
    model_versions_table = NEPTUNE_MODEL.fetch_model_versions_table().to_pandas()
    best_column_value = model_versions_table.sort_values(by=sort_column).iloc[0][sort_column]
    best_model_info = model_versions_table[model_versions_table[sort_column] == best_column_value]\
        .sort_values(by='test_metric/r2', ascending=False).iloc[0]
    print(best_model_info)
    best_model_version = neptune.init_model_version(with_id=best_model_info['sys/id'], project=NEPTUNE_PROJECT_NAME)
    temp_path = tempfile.NamedTemporaryFile().file.name
    best_model_version['model_file'].download(temp_path)
    best_model = h2o.load_model(temp_path)
    return best_model


def predict_collection_date(invoice_features: pandas.DataFrame, model: h2o.estimators.H2OEstimator) -> pandas.Series:
    invoice_features_h2o = h2o.H2OFrame(invoice_features)
    return model.predict(invoice_features_h2o).as_data_frame()['predict']


def test_prediction_on_open_invoices(invoices_raw: pandas.DataFrame, payments_raw: pandas.DataFrame) \
        -> pandas.DataFrame:
    """Given raw input datasets, return invoices with feature data and predictions"""
    invoices_with_payments, preprocess_filter_stats = preprocess_invoices_with_payments(invoices_raw, payments_raw)
    open_invoices_with_payments = invoices_with_payments.query("status=='OPEN'").copy()
    open_invoices_with_payments['forecast_date'] = payments_raw.transaction_date.max()
    open_invoices_feature_data = feature_engineering(open_invoices_with_payments.query("forecast_date>=invoice_date"))
    assert open_invoices_feature_data.invoice_id.value_counts().max() == 1, 'Duplicates per open invoice'
    model = get_best_model()
    return predict_collection_date(open_invoices_feature_data, model)


if __name__ == "__main__":
    pandas.set_option('expand_frame_repr', False)
    invoices, payments = get_csv_test_data()
    open_invoice_predictions = test_prediction_on_open_invoices(invoices, payments)
    print(open_invoice_predictions.value_counts())

