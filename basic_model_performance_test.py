import os
import pytest
import numpy as np

from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils

from skin_cancer.model_development.model import Model

model = Model().define_model()
model.load_weights('skin_cancer/data/models/weights.h5')

@pytest.mark.parametrize('pic, expected_result', [
    (pic, pic.split('.jpg')[0])
    for pic in os.listdir('examples')
]
)
def test_basic_model_performance(pic, expected_result):
    image = Image.open(os.path.join('examples', pic))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = image.resize((28, 28))
    image = img_to_array(image)/255
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    predicted_class = model.predict_classes(image)

    assert str(predicted_class[0]) == expected_result
