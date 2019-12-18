import numpy as np
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


class Model:
    def __init__(self):
        # Model parameters
        self.epochs = 50
        self.batch_size = 100
        self.num_class = 7
        self.num_rows = 28
        self.num_cols = 28
        self.learning_rate_reduction = ReduceLROnPlateau(
                                        monitor='val_accuracy', patience=3,
                                        verbose=1, factor=0.5,
                                        min_lr=0.00001)

        self.datagen = ImageDataGenerator(
                        featurewise_center=False,
                        samplewise_center=False,
                        featurewise_std_normalization=False,
                        samplewise_std_normalization=False,
                        zca_whitening=False, rotation_range=10,
                        zoom_range=0.1, width_shift_range=0.1,
                        height_shift_range=0.1, horizontal_flip=False,
                        vertical_flip=False)

        # Data
        self.X: np.array
        self.y: np.array

        self.model: Sequential

    def split(self):
        self.X_train, self.X_test, self.y_train, self.y_test =
        train_test_split(self.X, self.y, test_size=0.20, random_state=42)
        self.X_train, self.X_validate, self.y_train, self.y_test
        = train_test_split(self.X_train, self.y_train, test_size=0.20,
                           random_state=42)

    def define_model(self):
        model = Sequential()
        model.add(Conv2D(32, 3, padding='same', activation='relu',
                  input_shape=(self.num_rows, self.num_cols, 3)))
        model.add(Conv2D(32, 3, padding='same', activation='relu'))
        model.add(MaxPooling2D())
        model.add(Dropout(0.25))

        model.add(Conv2D(64, 3, padding='same', activation='relu',
                  input_shape=(self.num_rows, self.num_cols, 3)))
        model.add(Conv2D(64, 3, padding='same', activation='relu'))
        model.add(MaxPooling2D())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, 3, padding='same', activation='relu'))
        model.add(MaxPooling2D())
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.55))
        model.add(Dense(7, activation='softmax'))

        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])

    # Entry point to training the model
    # Consumes data from DataPipeline
    def model_runner(self, data):
        self.reshape_data(data)
        self.datagen.fit(self.X_train)
        self.model = self.define_model()
        self.model.fit(self.X_train, self.y_train,
                       batch_size=self.batch_size,
                       steps_per_epoch=self.X_train.shape[0] //
                       self.batch_size,
                       epochs=self.epochs,
                       validation_data=(self.X_validate, self.y_validate),
                       validation_steps=self.X_validate.shape[0] //
                       self.batch_size,
                       callbacks=[self.learning_rate_reduction])
        self.model.evaluate(self.X_test, self.y_test, verbose=1)
