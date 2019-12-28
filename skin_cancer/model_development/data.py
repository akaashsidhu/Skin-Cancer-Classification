import os
import sys
import kaggle
import pandas as pd
import numpy as np

from dotenv import load_dotenv

kaggle.api.authenticate()


class DataPipeline:
    def __init__(self):
        # List of preprocessing steps
        # Applied in the order they appear

        self.data_path = os.path.join(os.getcwd(), 'data')
        self.skin_data: pd.DataFrame
        self.X: np.array
        self.y: np.array

        # Loading environment variables
        dotenv_path = os.path.dirname(os.getcwd())
        load_dotenv(dotenv_path + '/.env')

    '''
        Preprocessing steps used to clean/feature engineer the data
        Example: normalization
    '''

    def normalization(self):
        '''
            We want to divide by each column by 255 to remove distortions
            caused by lights and shadows in an image.The range can be described
            with a 0.0-1.0 where 0.0 means 0 (0x00) and 1.0 means 255 (0xFF)
        '''
        self.X = self.skin_data.drop(columns='label')/255
        self.y = self.skin_data['label']

    def reshape_img(self):
        '''
            Reshape images to 28 by 28 and hot encode the Y label.
        '''
        num_rows, num_cols = 28, 28
        num_classes = 7

        self.X = np.array(self.X)
        self.X = self.X.reshape(self.X.shape[0], num_rows, num_cols, 3)
        self.y = np.eye(num_classes)[np.array(self.y.astype(int)).reshape(-1)]

    '''
        End preprocessing steps
    '''
    def download_files_from_kaggle(self):
        if not os.path.isfile(os.path.join(self.data_path, 'hmnist_28_28_RGB.csv')):  # noqa
            try:
                kaggle.api.authenticate()
                kaggle.api.dataset_download_files('kmader/skin-cancer-mnist-ham10000',  # noqa
                                            path=self.data_path, unzip=True)
            except TimeoutError:
                print('TimeoutError: manually download and try again')
                sys.exit()
        self.skin_data = pd.read_csv(os.path.join(self.data_path, 'hmnist_28_28_RGB.csv'))  # noqa

    # This will be the entry point
    def data_pipeline_runner(self):
        self.download_files_from_kaggle()
        self.normalization()
        self.reshape_img()
