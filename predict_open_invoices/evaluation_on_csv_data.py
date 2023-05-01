import os
import tempfile
import neptune
from neptune.types import File
from neptune.utils import stringify_unsupported
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
    NEPTUNE_PROJECT["raw_dataset/invoices"].track_files('../data_analysis/data/invoice.csv')
    NEPTUNE_PROJECT["raw_dataset/payments"].track_files('../data_analysis/data/invoice_payments.csv')
    h2o_frames, filter_stats_csv = get_training_data_from_csvs()
    for split in h2o_frames.keys():
        NEPTUNE_MODEL[f'data/{split}'].upload(File.as_html(h2o_frames[split].as_data_frame()))
    NEPTUNE_MODEL['data/filter_stats'].upload(File.as_html(filter_stats_csv))



def get_data_splits():
    temp_path = tempfile.NamedTemporaryFile()
    NEPTUNE_MODEL['data/train'].download(temp_path.file.name)


def train_model_version(params: dict = dict(metric='mae', predictors=['due_per_month'], y='collected_per_month',
                                            distribution='huber',max_runtime_secs=60)):
    neptune_model_version = neptune.init_model_version(model=f"{NEPTUNE_PROJECT_ID}-{NEPTUNE_MODEL_ID}",
                                                       project=NEPTUNE_PROJECT_NAME)
    neptune_model_version['parameters/metrics'] = stringify_unsupported(['r2', 'mae', 'mse', 'rmse', 'rmsle'])
    neptune_model_version['parameters/training'] = stringify_unsupported(
        ['response_column', 'metalearner_algorithm', 'metalearner_nfolds','max_runtime_secs', 'weights_column'])
    # uni_variate_model = train_model(h2o_frames['train'], h2o_frames['test'], **params)
    # filter_stats, model, time_based_split_h2o_frames = train_on_csv_test_data(**model_params)
    #
    # for key in neptune_model_version['parameters/training']:
    #     neptune_model_version[f'train/{key}'] = model.actual_params[key]
    # model_params['algo'] = model.algo
    # neptune_model_version["model/parameters"] = model_params
    # model_params['mae']
    # #neptune_model_version["model"].upload(PATH_TO_MODEL)


if __name__ == '__main__':
    create_neptune_csv_model()
