import os
from dotenv import load_dotenv

dotenv_path = os.path.dirname(os.getcwd())
load_dotenv(dotenv_path + '/.env')


class Model:
    pass
