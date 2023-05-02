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
from predict_open_invoices import DATA_FOLDER
from predict_open_invoices.training import get_training_data_from_csvs, train_model
from predict_open_invoices.prediction import predict
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
    NEPTUNE_MODEL.sync()


def get_data_splits(keys: list[str] = ['train', 'test']) -> list[pandas.DataFrame]:
    if len(keys) == 0:
        return keys
    #h2o.init(nthreads=-1, max_mem_size=12)
    frames = []
    for key in keys:
        temp_path = tempfile.NamedTemporaryFile().file.name
        NEPTUNE_MODEL[f'data/{key}'].download(temp_path)
        df = pandas.read_pickle(temp_path)
        #id_columns_h2o = [col for col in ID_COLUMNS if col in df.columns]
        frames.append(df)
    return frames


def get_monthly_forecast_error(df: pandas.DataFrame, h2o_model: h2o.estimators.H2OEstimator,
                               y: str = 'collected_per_month'):
    predictions = predict(df, h2o_model)
    df = df[['collected_per_month', 'month_collected', 'inv_pct_of_company_total']]
    if y == 'collected_per_month':
        df["predict_month_collected"] = (1 / predictions).replace(numpy.inf, None).round(0)
    else:
        df['predict_month_collected'] = predictions.round(0).astype(int)
    df = df.groupby("predict_month_collected", as_index=False).inv_pct_of_company_total.sum().merge(
        df.groupby("month_collected", as_index=False).inv_pct_of_company_total.sum()
        .rename(columns={"month_collected": "predict_month_collected"}), on="predict_month_collected",
        suffixes=('_predict', ''), how="left")
    df['abs_diff'] = (df.inv_pct_of_company_total - df.inv_pct_of_company_total_predict).abs()
    mape = float(df.abs_diff.sum()/df.inv_pct_of_company_total_predict.sum())
    return mape


def train_model_version(params: dict = dict(metric='mae', predictors=['due_per_month'], y='collected_per_month',
                                            distribution='huber', max_runtime_secs=60)) -> float:
    neptune_model_version = neptune.init_model_version(model=f"{NEPTUNE_PROJECT_ID}-{NEPTUNE_MODEL_ID}",
                                                       project=NEPTUNE_PROJECT_NAME)
    neptune_model_version['predictors'] = stringify_unsupported(params['predictors'])
    neptune_model_version['response_column'] = params['y']
    neptune_model_version['response_distribution'] = params['distribution']
    neptune_model_version['max_runtime_secs'] = params['max_runtime_secs']
    neptune_model_version['ml_sort_metric'] = params['metric']
    neptune_model_version['ml_log_metrics'] = stringify_unsupported(['r2', 'mae', 'rmsle'])
    neptune_model_version['nfolds'] = params['nfolds']
    #neptune_model_version['stopping_tolerance'] = params['stopping_tolerance']
    train, test, valid = get_data_splits(['train', 'test', 'validation'])
    h2o_model = train_model(train, test, **params)
    neptune_model_version['monthly_mape_train'] = get_monthly_forecast_error(train, h2o_model, y=params['y'])
    mape_test = get_monthly_forecast_error(test, h2o_model, y=params['y'])
    neptune_model_version['monthly_mape_test'] = mape_test
    neptune_model_version['monthly_mape_valid'] = get_monthly_forecast_error(valid, h2o_model, y=params['y'])
    neptune_model_version['algo'] = h2o_model.algo
    weights_column, varimp = h2o_model.actual_params['weights_column'], h2o_model.varimp(use_pandas=True)
    if weights_column:
        neptune_model_version['weights_column'] = weights_column['column_name']
    if varimp is not None:
        neptune_model_version['feature_importance'].upload(File.as_html(varimp))
    temp_dir = tempfile.TemporaryDirectory().name
    model_path = h2o.save_model(model=h2o_model, path=temp_dir, force=True)
    neptune_model_version['model_file'].upload(model_path)
    for metric in ast.literal_eval(neptune_model_version['ml_log_metrics'].fetch()):
        neptune_model_version[f'train_metric/{metric}'] = h2o_model.model_performance()[metric]
        neptune_model_version[f'test_metric/{metric}'] = h2o_model.model_performance(test)[metric]
    return mape_test


def optuna_objective(trial):
    #y = trial.suggest_categorical('y', ['collected_per_month', 'month_collected'])
    # predictors = ['due_per_month', 'root_exchange_rate_value_inv', 'amount_inv',
    #               'remaining_inv_pct', 'months_open']
    y = 'collected_per_month'
    predictors = ['months_allowed', 'months_open', 'remaining_inv_pct', 'amount_inv']

    predictors += ['due_per_month'] if y == 'collected_per_month' else ['month_due']
    params = \
        {'y': y, 'predictors': predictors, 'metric': 'mae',
         'distribution': 'poisson' if y == 'month_collected' else 'gaussian',
         #'exclude_algos': [],
         'exclude_algos': ast.literal_eval(trial.suggest_categorical('exclude_algos', ("[]", "['StackedEnsemble']"))),
         #'metric': trial.suggest_categorical('metric', ('mae', 'rmsle')),
         'nfolds': trial.suggest_categorical('nfolds', [-1, 0, 2, 4, 5, 6]),
         'max_runtime_secs': trial.suggest_int('max_runtime_secs', 60 * 3, 60 * 7),
          #'stopping_tolerance': trial.suggest_float('stopping_tolerance', 0.003, 0.03)
         }
    metric_to_minimize = train_model_version(params)
    return metric_to_minimize


if __name__ == '__main__':
    pandas.set_option('expand_frame_repr', False)
    create_neptune_csv_model()
    study = optuna.create_study(direction='minimize')
    study.optimize(optuna_objective, n_trials=3)
