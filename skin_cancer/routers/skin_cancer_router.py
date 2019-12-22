from fastapi import APIRouter
from starlette.responses import JSONResponse


def skin_cancer_prediction(text):
    return text


router = APIRouter()


@router.post('/predict')
def lesion_classification(lesion_image: str):
    return JSONResponse(skin_cancer_prediction(lesion_image))
