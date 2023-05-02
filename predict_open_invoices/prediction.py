import os
import tempfile
import numpy
import h2o
import pandas
import neptune
from predict_open_invoices import NEPTUNE_PROJECT_NAME, NEPTUNE_MODEL_ID
from predict_open_invoices.csv_test_data_io import get_csv_test_data
from predict_open_invoices.pre_processing import preprocess_invoices_with_payments
from predict_open_invoices.feature_engineering import feature_engineering
h2o.init(nthreads=-1, max_mem_size=12)
NEPTUNE_PROJECT = neptune.init_project(project=NEPTUNE_PROJECT_NAME, api_token=os.getenv('NEPTUNE_API_TOKEN'))
NEPTUNE_PROJECT_ID = NEPTUNE_PROJECT['sys/id'].fetch()


def _get_best_model(sort_column: str = 'monthly_mape_test') -> h2o.estimators.H2OEstimator:
    """Pick the model version from neptune that minimizes the mean absolute percentage diff between
     monthly amount collected (normalized by company) and monthly amount forecasted on test data."""
    neptune_model = neptune.init_model(with_id=f"{NEPTUNE_PROJECT_ID}-{NEPTUNE_MODEL_ID}", project=NEPTUNE_PROJECT_NAME)
    neptune_model.sync()
    model_versions_table = neptune_model.fetch_model_versions_table().to_pandas()
    best_column_value = model_versions_table.sort_values(by=sort_column).iloc[0][sort_column]
    best_model_info = model_versions_table[model_versions_table[sort_column] == best_column_value]\
        .sort_values(by='test_metric/r2', ascending=False).iloc[0]
    print(best_model_info)
    best_model_version = neptune.init_model_version(with_id=best_model_info['sys/id'], project=NEPTUNE_PROJECT_NAME)
    temp_path = tempfile.NamedTemporaryFile().file.name
    best_model_version['model_file'].download(temp_path)
    best_model = h2o.load_model(temp_path)
    return best_model


def predict(invoice_features: pandas.DataFrame, model: h2o.estimators.H2OEstimator) -> pandas.Series:
    """Predict month collected relative to forecast date on set of invoices and payments summarized up to the point
    in time being forecasted"""
    invoice_features_h2o = h2o.H2OFrame(invoice_features)
    predictions = model.predict(invoice_features_h2o).as_data_frame()['predict']
    # predictions represent collection rates
    if predictions.min() == 0:
        return (1 / predictions).replace(numpy.inf, None).round(0)
    # predictions represent future month collected
    return predictions.round(0).astype(int)


def _test_prediction_on_open_invoices(invoices_raw: pandas.DataFrame, payments_raw: pandas.DataFrame) \
        -> pandas.DataFrame:
    """Given raw input datasets, return OPEN invoices with feature data and predictions"""
    invoices_with_payments, preprocess_filter_stats = preprocess_invoices_with_payments(invoices_raw, payments_raw)
    open_invoices_with_payments = invoices_with_payments.query("status=='OPEN'").copy()
    open_invoices_with_payments['forecast_date'] = payments_raw.transaction_date.max()
    open_invoices_feature_data = feature_engineering(open_invoices_with_payments.query("forecast_date>=invoice_date"))
    assert open_invoices_feature_data.invoice_id.value_counts().max() == 1, 'Duplicates per open invoice'
    model = _get_best_model()
    return predict(open_invoices_feature_data, model)


if __name__ == "__main__":
    pandas.set_option('expand_frame_repr', False)
    invoices, payments = get_csv_test_data()
    open_invoice_predictions = _test_prediction_on_open_invoices(invoices, payments)
    print(open_invoice_predictions.value_counts())

