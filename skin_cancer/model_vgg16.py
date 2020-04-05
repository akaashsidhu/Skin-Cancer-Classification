import numpy as np
import pickle
from keras.models import Model
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras import layers
from skin_cancer.model_development.data_vgg16 import DataPipeline


class Transfer_Model:

    def __init__(self):

        self.epochs = 30
        self.IMG_SIZE = 224
        self.batch_size = 64
        self.num_classes = 7

        self.accuracy_threshold = 0.8
        self.f1_threshold = 0.75

        self.model: Sequential

        self.learning_rate_reduction = ReduceLROnPlateau(monitor='acc', patience=3, verbose=1, factor=0.5,  # noqa
                                                    min_lr=0.000001, cooldown=3)  # noqa
        self.train_datagen = ImageDataGenerator(rotation_range=60, width_shift_range=0.2, height_shift_range=0.2,  # noqa
                                           shear_range=0.2, zoom_range=0.2, fill_mode='nearest')  # noqa

    def load_data(self, X, y):

        pickle_in = open('X_train.pickle', 'rb')
        self.X_train = pickle.load(pickle_in)

        pickle_in = open('X_test.pickle', 'rb')
        self.X_test = pickle.load(pickle_in)

        pickle_in = open('y_train.pickle', 'rb')
        self.y_train = pickle.load(pickle_in)

        pickle_in = open('y_test.pickle', 'rb')
        self.y_test = pickle.load(pickle_in)

    def manipulate_data(self, X, y):

        self.X_train = self.X_train/255
        self.y_train = np.eye(self.num_classes)[np.array(self.y_train.astype(int)).reshape(-1)]  # noqa

        self.X_test = self.X_test/255
        self.y_test = np.eye(self.num_classes)[np.array(self.y_test.astype(int)).reshape(-1)]  # noqa

    def recall_m(self, y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        self.recall = true_positives / (possible_positives + K.epsilon())
        return self.recall

    def precision_m(self, y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1_m(self, y_true, y_pred):

        precision = self.precision_m(y_true, y_pred)
        recall = self.recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def pretrained(self):

        self.pre_trained_model = VGG16(input_shape=(self.IMG_SIZE, self.IMG_SIZE, 3), include_top=False, weights='imagenet')  # noqa
        last_layer = self.pre_trained_model.get_layer('block5_pool')
        last_output = last_layer.output

        self.x = layers.GlobalMaxPooling2D()(last_output)
        self.x = layers.Dense(512, activation='relu')(self.x)
        self.x = layers.Dropout(0.5)(self.x)
        self.x = layers.Dense(7, activation='softmax')(self.x)

        model = Model(self.pre_trained_model.input, self.x)

        self.train_datagen.fit(self.X_train)

        for layer in model.layers[:15]:
            layer.trainable = False
        for layer in model.layers[15:]:
            layer.trainable = True

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['acc', self.f1_m, self.precision_m, self.recall_m])  # noqa

        return model

    def model_runner(self, X, y):

        self.load_data(X, y)
        self.manipulate_data(X, y)
        self.model = self.pretrained()
        self.model.fit_generator(self.train_datagen.flow(self.X_train, self.y_train,  # noqa
                       batch_size=self.batch_size), epochs=self.epochs, verbose=1,  # noqa
                       callbacks=[self.learning_rate_reduction])
        loss, accuracy, f1_score, precision, recall = model.evaluate(self.X_test, self.y_test, verbose=0)  # noqa

        if accuracy >= self.test_performance_threshold and f1_score >= self.f1_threshold:  # noqa
            self.model.save('skin_cancer/data/models/VGG16_weights.h5')


if __name__ == '__main__':

    def train_model():

        data = DataPipeline()
        data.data_pipeline_runner()
        model = Transfer_Model()
        model.model_runner(data.X, data.y)
