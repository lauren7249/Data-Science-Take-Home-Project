from collections import OrderedDict
import os
import tempfile
import ast
import numpy
import h2o
import pandas
import optuna
import neptune
from neptune.types import File
from neptune.utils import stringify_unsupported
from predict_open_invoices import ID_COLUMNS
from predict_open_invoices import DATA_FOLDER
from predict_open_invoices.training import get_training_data_from_csvs, train_model
NEPTUNE_PROJECT_NAME = "open-invoices-model"
NEPTUNE_PROJECT = neptune.init_project(project=NEPTUNE_PROJECT_NAME, api_token=os.getenv('NEPTUNE_API_TOKEN'))
NEPTUNE_PROJECT_ID = NEPTUNE_PROJECT['sys/id'].fetch()
NEPTUNE_MODEL_ID = "CSVDATA"
try:
    NEPTUNE_MODEL = neptune.init_model(key=NEPTUNE_MODEL_ID, project=NEPTUNE_PROJECT_NAME,
                                       name="Trained on CSV data.", api_token=os.getenv('NEPTUNE_API_TOKEN'))
except neptune.exceptions.NeptuneModelKeyAlreadyExistsError:
    NEPTUNE_MODEL = neptune.init_model(with_id=f"{NEPTUNE_PROJECT_ID}-{NEPTUNE_MODEL_ID}", project=NEPTUNE_PROJECT_NAME)


def create_neptune_csv_model() -> neptune.Model:
    """Create base model. Only needs to be run once."""
    url_to_project_brief = 'https://docs.google.com/presentation/d/1bWMEfc8l3tMlpzF1RijV2j-JU8gDEwDq6Ks4bkT47Lo'
    NEPTUNE_PROJECT["general/brief"] = url_to_project_brief
    NEPTUNE_PROJECT["general/eda_html_for_download"].upload("../data_analysis/eda.html")
    NEPTUNE_PROJECT["raw_dataset/invoices"].track_files(f'{DATA_FOLDER}/invoice.csv')
    NEPTUNE_PROJECT["raw_dataset/payments"].track_files(f'{DATA_FOLDER}/invoice_payments.csv')
    h2o_frames, filter_stats_csv = get_training_data_from_csvs()
    NEPTUNE_MODEL['data/filter_stats'].upload(File.as_html(filter_stats_csv))
    for split in h2o_frames.keys():
        split_df = h2o_frames[split].as_data_frame()
        summary_stats = split_df.drop(columns=['status']).describe(include='all', percentiles=[]) \
            .T.drop(columns=['50%', 'std', 'top', 'freq'])
        NEPTUNE_MODEL[f'data/{split}_stats'].upload(File.as_html(summary_stats))
        NEPTUNE_MODEL[f'data/{split}'].upload(File.as_pickle(split_df))


def get_data_splits(keys: list[str] = ['train', 'test']) -> list[h2o.H2OFrame]:
    if len(keys) == 0:
        return keys
    h2o.init(nthreads=-1, max_mem_size=12)
    h2o_frames = []
    for key in keys:
        temp_path = tempfile.NamedTemporaryFile()
        NEPTUNE_MODEL[f'data/{key}'].download(temp_path.file.name)
        df = pandas.read_pickle(temp_path.file.name)
        id_columns_h2o = [col for col in ID_COLUMNS if col in df.columns]
        h2o_frames.append(h2o.H2OFrame(df, column_types=dict(zip(id_columns_h2o, ["string"] * len(id_columns_h2o)))))
    return h2o_frames


def get_monthly_forecast_error(h2o_frame: h2o.H2OFrame, model: h2o.estimators.H2OEstimator,
                               y: str = 'collected_per_month'):
    results = h2o_frame[[y, 'inv_pct_of_company_total']].cbind(model.predict(h2o_frame)).as_data_frame()
    results.rename(columns={"predict": f"predict_{y}"}, inplace=True)
    if results[y].between(0, 1).min():
        results[f"{y}_inverse"] = (1 / results[y]).replace(numpy.inf, numpy.nan).round(0)
        results[f"predict_{y}_inverse"] = (1 / results[f"predict_{y}"]).replace(numpy.inf, None).round(0)
    results = results.groupby(f"predict_{y}_inverse", as_index=False).inv_pct_of_company_total.sum().merge(
        results.groupby(f"{y}_inverse", as_index=False).inv_pct_of_company_total.sum().rename(
            columns={f"{y}_inverse": f"predict_{y}_inverse"}), on=f"predict_{y}_inverse", suffixes=('_predict', ''))
    results['abs_diff'] = (results.inv_pct_of_company_total - results.inv_pct_of_company_total_predict).abs()
    return results.abs_diff.sum()/results.inv_pct_of_company_total_predict.sum()


def train_model_version(params: dict = dict(metric='mae', predictors=['due_per_month'], y='collected_per_month',
                                            distribution='huber', max_runtime_secs=60)):
    neptune_model_version = neptune.init_model_version(model=f"{NEPTUNE_PROJECT_ID}-{NEPTUNE_MODEL_ID}",
                                                       project=NEPTUNE_PROJECT_NAME)
    neptune_model_version['predictors'] = stringify_unsupported(params['predictors'])
    neptune_model_version['response_column'] = params['y']
    neptune_model_version['response_distribution'] = params['distribution']
    neptune_model_version['max_runtime_secs'] = params['max_runtime_secs']
    neptune_model_version['ml_sort_metric'] = params['metric']
    neptune_model_version['ml_log_metrics'] = stringify_unsupported(['r2', 'mae', 'rmsle'])
    train, test = get_data_splits(['train', 'test'])
    model = train_model(train, test, **params)
    neptune_model_version['monthly_mape_train'] = get_monthly_forecast_error(train, model, y=params['y'])
    neptune_model_version['monthly_mape_test'] = get_monthly_forecast_error(test, model, y=params['y'])
    neptune_model_version['algo'] = model.algo
    weights_column, varimp = model.actual_params['weights_column'], model.varimp(use_pandas=True)
    if weights_column:
        neptune_model_version['weights_column'] = weights_column['column_name']
    if varimp:
        neptune_model_version['feature_importance'].upload(File.as_html(varimp))
    temp_dir = tempfile.TemporaryDirectory().name
    model_path = h2o.save_model(model=model, path=temp_dir, force=True)
    neptune_model_version['model_file'].upload(model_path)
    for metric in ast.literal_eval(neptune_model_version['ml_log_metrics'].fetch()):
        neptune_model_version[f'train_metric/{metric}'] = model.model_performance()[metric]
        neptune_model_version[f'test_metric/{metric}'] = model.model_performance(test)[metric]
    return neptune_model_version['monthly_mape_test']


def get_best_model():
    model_versions_table = NEPTUNE_MODEL.fetch_model_versions_table().to_pandas()
    best_model_version_id = model_versions_table.sort_values(by='monthly_mape_test').iloc[0]['sys/id']
    best_model_version = neptune.init_model_version(with_id=best_model_version_id, project=NEPTUNE_PROJECT_NAME)
    temp_path = tempfile.NamedTemporaryFile()
    best_model_version['model_file'].download(temp_path.file.name)
    best_model = h2o.load_model(temp_path.file.name)
    return best_model


def optuna_objective(trial):
    params = {'predictors': ['due_per_month', 'currency', 'root_exchange_rate_value_inv', 'amount_inv',
                             'remaining_inv_pct', 'months_open', 'inv_pct_of_company_total'],
              'y': 'collected_per_month',
              'exclude_algos': trial.suggest_categorical('exclude_algos', [[], ['StackedEnsemble']]),
              'distribution': trial.suggest_categorical('distribution', ['huber', 'laplace', 'gamma']),
              'metric': trial.suggest_categorical('metric', ['mae', 'r2', 'rmsle']),
              'max_runtime_secs': trial.suggest_loguniform('max_runtime_secs', 30, 60 * 5)}
    return train_model_version(params)


if __name__ == '__main__':
    pandas.set_option('expand_frame_repr', False)
    create_neptune_csv_model()
    study = optuna.create_study(direction='minimize')
    study.optimize(optuna_objective, n_trials=1)
    model = get_best_model()
