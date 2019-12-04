import os
import sys
import kaggle
import pandas as pd

from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

kaggle.api.authenticate()

class DataPipeline:
    def __init__(self):
        # List of preprocessing steps
        # Applied in the order they appear
        self.preprocessing_steps = [
                                        self.normalization,
                                        self.reshape_img,self.data_split
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

    def normalization(self):
        '''
            We want to divide by each column by 255 to remove distortions
            caused by lights and shadows in an image.The range can be described
            with a 0.0-1.0 where 0.0 means 0 (0x00) and 1.0 means 255 (0xFF)
        '''
        self.X = self.skin_data.drop(columns='label')/255
        self.Y = self.skin_data['label']
        return self.normalization()

    def reshape_img(self):
        '''
            Reshape images to 28 by 28 and hot encode the Y label.
        '''
        num_rows, num_cols = 28, 28
        num_classes = 7

        self.X = np.array(self.X)
        self.X = self.X.reshape(self.X.shape[0], num_rows, num_cols, 3)
        self.Y = np.eye(num_classes)[np.arrays(self.Y.astype(int)).reshape(-1)]
        return self.reshape_img()

    def data_split(self):
        '''
            Split the dataset into training, validation, and test using
            scikit-learn
        '''
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.20, random_state=42)
        self.X_train, self.X_validate, self.y_train, self.y_validate = train_test_split(self.X_train, self.y_train, test_size=0.10, random_state=42)
        return self.X_train, self.X_test, self.X_validate, self.y_train, self.y_test, self.y_validate

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
