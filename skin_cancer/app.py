from fastapi import FastAPI
from skin_cancer.routers import skin_cancer_router

app = FastAPI()
app.include_router(skin_cancer_router.router, prefix='/skin-cancer-classification')  # noqa


@app.get('/healthcheck', status_code=200)
async def healthcheck():
    return 'Skin cancer classifier is all ready to go!'
