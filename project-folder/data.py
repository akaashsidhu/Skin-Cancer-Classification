#!bin/bash/env python3

import os
import zipfile
import pandas as pd
import numpy as np
import kaggle

from typing import Optional
from kaggle.api.kaggle_api_extended import KaggleApi

data_path = os.path.join(os.getcwd(), 'data')

print(data_path)

kaggle.api.authenticate()
kaggle.api.dataset_download_files('kmader/skin-cancer-mnist-ham10000', path=data_path, unzip = True)
