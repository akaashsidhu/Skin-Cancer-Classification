from fastapi import APIRouter, File
from PIL import Image
import io
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils

from skin_cancer.model_development.model import Model


router = APIRouter()


@router.post('/predict')
def skin_lesion_classification(image_file: bytes = File(...)):
    model = Model().define_model()
    model.load_weights('data/models/weights.hdf5')

    image = Image.open(io.BytesIO(image_file))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    preds = model.predict(image)

    return {'result': preds}
