import os
import sys
import kaggle
import pandas as pd
import numpy as np
import pickle
import cv2
import random

from dotenv import load_dotenv
from tqdm import tqdm

kaggle.api.authenticate()


class DataPipeline:

    def __init__(self):

        self.search_dir = os.getcwd() + '/data/'
        self.image_path = os.getcwd() + '/model_development/data/HAM10000_images/'  # noqa
        self.data_output_path = os.getcwd() + '/model_development/'

        self.skin_data: pd.DataFrame
        self.dx_unique: np.array

        self.train_folder = self.image_path + 'Train'
        self.test_folder = self.image_path + 'Test'
        # Creating folder

        self.train = self.train_folder + '/Training'
        self.test = self.test_folder + '/Testing'
        # Moving folder

        self.categories = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        self.IMG_SIZE = 224

        self.training_data = []
        self.testing_data = []

        self.X_train = []
        self.y_train = []

        self.X_test = []
        self.y_test = []

        dotenv_path = os.path.dirname(os.getcwd())
        load_dotenv(dotenv_path + '/.env')
        # Loading environment variables

        self.t = 0.10
        # percentage of test data

    def download_files_from_kaggle(self):

        '''This will download files from kaggle into the correct folder
        if it's not there already and will read into the csv to get the
        unique dx'''

        if os.path.isfile('model_development/data/HAM10000_metadata.csv'):
            self.skin_data = pd.read_csv('model_development/data/HAM10000_metadata.csv')  # noqa
            self.dx_unique = self.skin_data.dx.unique()
            print('Data already downloaded')
        else:
            try:
                kaggle.api.authenticate()
                kaggle.api.dataset_download_files('kmader/skin-cancer-mnist-ham10000',  # noqa
                                                    path=self.search_dir, unzip=True)  # noqa
            except TimeoutError:
                print('TimeoutError: manually download and try again')
                sys.exit()

    def organize_folders(self):

        '''Make the folders, create folders for each id and move images into them'''  # noqa

        os.mkdir(self.train_folder)
        os.mkdir(self.test_folder)

        os.mkdir(self.train)
        os.mkdir(self.test)

        for f in os.listdir(self.image_path):
            pics = os.listdir(self.image_path)
            numpics = len(pics)
            numtestpics = round(self.t * numpics)

        test_pics = random.sample(pics, numtestpics)

        for p in test_pics:
            file_path = f'{self.image_path}/{p}'
            test_path = f'{self.test}/{p}'  # noqa

        os.chdir(self.train)

        for f in self.dx_unique():
            os.makedirs(f'{f}/')

        os.chdir(self.test)

        for f in self.dx_unique():
            os.makedirs(f'{f}/')

        os.chdir(self.image_path)

        for p in self.skin_data.itertuples():
            try:
                filepath = f'{self.train_folder}/{p.image_id}.jpg'
                trainpath = f'{self.train}/{p.dx}/{p.image_id}.jpg'
                os.rename(f'{filepath}', f'{trainpath}')
            except OSError:
                pass
            # This moves the images based on their dx category into sub-folders to be used for training  # noqa

        for p in self.skin_data.itertuples():
            try:
                filepath = f'{self.test_folder}/{p.image_id}.jpg'  # noqa
                testpath = f'{self.test}/{p.dx}/{p.image_id}.jpg'  # noqa
                os.rename(f'{filepath}', f'{testpath}')
            except OSError:
                pass

        # This moves the images based on their dx category into sub-folders to be used for testing  # noqa

    def create_training_data(self):
        for category in self.categories:

            path = os.path.join(self.train, category)  # create path to training images  # noqa
            class_num = self.categories.index(category)  # get the classification  # noqa

            for img in tqdm(os.listdir(path)):  # iterate over each image per category  # noqa
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.COLOR_BGR2RGB)  # convert to array  # noqa
                    img_array_RGB = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                    new_array = cv2.resize(img_array_RGB, (self.IMG_SIZE, self.IMG_SIZE))  # resize to normalize data size  # noqa
                    self.training_data.append([new_array, class_num])  # add this to our training_data  # noqa
                except Exception as e:  # in the interest in keeping the output clean...  # noqa
                    pass

    def create_testing_data(self):
        for category in self.categories:

            path = os.path.join(self.test, category)
            class_num = self.categories.index(category)

            for img in tqdm(os.listdir(path)):
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.COLOR_BGR2RGB)  # convert to array  # noqa
                    img_array_RGB = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                    new_array = cv2.resize(img_array_RGB, (self.IMG_SIZE, self.IMG_SIZE))  # resize to normalize data size  # noqa
                    self.testing_data.append([new_array, class_num])  # add this to our training_data  # noqa
                except Exception as e:  # in the interest in keeping the output clean...  # noqa
                    pass

    def reshape(self):
        random.shuffle(self.training_data)
        random.shuffle(self.testing_data)

        for features, label in self.training_data:
            self.X_train.append(features)
            self.y_train.append(label)

        for features, label in self.testing_data:
            self.X_test.append(features)
            self.y_test.append(label)

        self.X_train = np.array(self.X_train).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 3)  # noqa
        self.y_train = np.asarray(self.y_train, dtype=np.int64)

        self.X_test = np.array(self.X_test).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 3)  # noqa
        self.y_test = np.asarray(self.y_test, dtype=np.int64)

    def pickle_data(self):
        os.chdir(self.data_output_path)

        pickle_out = open('X_train.pickle', 'wb')
        pickle.dump(self.X_train, pickle_out)
        pickle_out.close()

        pickle_out = open('X_test.pickle', 'wb')
        pickle.dump(self.X_test, pickle_out)
        pickle_out.close()

        pickle_out = open('y_train.pickle', 'wb')
        pickle.dump(self.y_train, pickle_out)
        pickle_out.close()

        pickle_out = open('y_test.pickle', 'wb')
        pickle.dump(self.y_test, pickle_out)
        pickle_out.close()

    def data_pipeline_runner(self):
        self.download_files_from_kaggle()
        self.organize_folders()
        self.create_training_data()
        self.create_testing_data()
        self.reshape()
        self.pickle_data()
