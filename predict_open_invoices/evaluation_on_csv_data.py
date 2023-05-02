import os
import tempfile
import ast
import h2o
import pandas
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
    neptune_model_version['algo'] = model.algo
    neptune_model_version['weights_column'] = model.actual_params['weights_column']['column_name']
    neptune_model_version['feature_importance'].upload(File.as_html(model.varimp(use_pandas=True)))
    temp_dir = tempfile.TemporaryDirectory().name
    model_path = h2o.save_model(model=model, path=temp_dir, force=True)
    neptune_model_version['model_file'].upload(model_path)
    for metric in ast.literal_eval(neptune_model_version['ml_log_metrics'].fetch()):
        neptune_model_version[f'train_metric/{metric}'] = model.model_performance()[metric]
        neptune_model_version[f'test_metric/{metric}'] = model.model_performance(test)[metric]


def get_best_model():
    model_versions_table = NEPTUNE_MODEL.fetch_model_versions_table().to_pandas()
    model_versions_table['avg_mae'] = model_versions_table[['train_metric/mae', 'test_metric/mae']].mean(axis=1)
    best_model_version_id = model_versions_table.sort_values(by='avg_mae').iloc[0]['sys/id']
    best_model_version = neptune.init_model_version(with_id=best_model_version_id, project=NEPTUNE_PROJECT_NAME)
    temp_path = tempfile.NamedTemporaryFile()
    best_model_version['model_file'].download(temp_path.file.name)
    best_model = h2o.load_model(temp_path.file.name)
    return best_model


if __name__ == '__main__':
    pass
    # create_neptune_csv_model()
    # train_model_version()
    # train_model_version(dict(metric='mae', y='collected_per_month', distribution='huber', max_runtime_secs=60,
    #                          predictors=['due_per_month', 'currency', 'root_exchange_rate_value_inv', 'amount_inv',
    #                                      'remaining_inv_pct', 'months_open', 'inv_pct_of_company_total']))
    # train_model_version(dict(metric='mae', y='collected_per_month', distribution='huber', max_runtime_secs=60*5,
    #                          predictors=['due_per_month', 'currency', 'root_exchange_rate_value_inv', 'amount_inv',
    #                                      'remaining_inv_pct', 'months_open', 'inv_pct_of_company_total']))
    # model = get_best_model()

