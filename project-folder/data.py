import os
import sys
import kaggle
import pandas as pd

from dotenv import load_dotenv


class DataPipeline:
    def __init__(self):
        # List of preprocessing steps
        # Applied in the order they appear
        self.preprocessing_steps = [
                                        self.placeholder_callable_1,
                                        self.placeholder_callable_2,
                                    ]
        self.data_path = os.path.join(os.getcwd(), 'data')
        self.skin_data: pd.DataFrame

        # Loading environment variables
        dotenv_path = os.path.dirname(os.getcwd())
        load_dotenv(dotenv_path + '/.env')

    '''
        Preprocessing steps used to clean/feature engineer the data
        Example: normalization
    '''

    def placeholder_callable_1(self, row):
        return row

    def placeholder_callable_2(self, row):
        return row

    '''
        End preprocessing steps
    '''

    def download_files_from_kaggle(self):
        if not os.path.isfile('data/hmnist_28_28_RGB.csv'):
            try:
                kaggle.api.authenticate()
                kaggle.api.dataset_download_files('kmader/skin-cancer-mnist-ham10000',  # noqa
                                            path=self.data_path, unzip=True)
            except TimeoutError:
                print('TimeoutError: manually download and try again')
                sys.exit()
        self.skin_data = pd.read_csv('data/hmnist_28_28_RGB.csv')

    # This will be the entry point
    def data_pipeline_runner(self):
        self.download_files_from_kaggle()
        for func in self.preprocessing_steps:
            # If it's easier, function can be applied to column instead of row
            self.skin_data = self.skin_data.apply(func, axis=1)
            print('Finished applying {}'.format(func.__name__))
