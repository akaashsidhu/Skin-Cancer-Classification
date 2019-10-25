# SM_project

## Usage Requirements

Install all dependencies using `pip install -r requirements.txt`

This repo uses the Kaggle API to fetch the dataset. Follow these steps to access the Kaggle API:

1. Create a Kaggle account
2. Generate an API token from `kaggle.com/USERNAME/account` -- this prompts you to download a `kaggle.json` file which contains the credentials
3. Populate `.env` file with your Kaggle credentials to set them as your environment variable.

We also use `pre-commit` hooks to keep our code clean. Please enable this using `pre-commit install` prior to making any code changes.
