import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


class Model:
    def __init__(self):
        # Model parameters
        self.epochs = 10
        self.batch_size = 32
        self.num_class = 6
        self.num_rows = 28
        self.num_cols = 28

        # Data
        self.X: np.array
        self.y: np.array

        self.model: Sequential

    def define_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(self.num_rows, self.num_cols, 3)))  # noqa
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.1))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    def reshape_data(self, data):
        y = data['label']
        X = data.drop(columns='label')
        X = np.array(X)
        self.X = X.reshape(X.shape[0], self.num_rows, self.num_cols, 3)
        self.y = np.eye(self.num_classes)[np.array(y.astype(int)).reshape(-1)]

    # Entry point to training the model
    # Consumes data from DataPipeline
    def model_runner(self, data):
        self.reshape_data(data)
        self.model = self.define_model()
        self.model.fit(self.X, self.y,
                       batch_size=self.batch_size,
                       epochs=self.epochs)
