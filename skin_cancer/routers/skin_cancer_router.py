from fastapi import APIRouter, File
from PIL import Image
import io
import numpy as np
import tensorflow as tf

from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils

from skin_cancer.model_development.model import Model

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post('/predict')
def skin_lesion_classification(image_file: bytes = File(...)):
    model = Model().define_model()
    model.load_weights('skin_cancer/data/models/weights.h5')

    image = Image.open(io.BytesIO(image_file))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = image.resize((28, 28))

    logger.info('Image has been resized')

    image = img_to_array(image)/255
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    logger.info('Predicting image ...')

    graph = tf.get_default_graph()

    with graph.as_default():
        predicted_class = model.predict_classes(image)

    mapping = {
        0: 'Melanocytic nevi (non-cancerous)',
        1: 'Melanoma (cancerous)',
        2: 'Benign keratosis-like lesions (non-cancerous)',
        3: 'Basal cell carcinoma (cancerous)',
        4: 'Actinic keratoses (non-cancerous)',
        5: 'Vascular lesions (non-cancerous)',
        6: 'Dermatofibroma (non-cancerous)'
    }

    return {'predicted class': mapping[predicted_class[0]]}
