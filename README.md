# Skin Cancer Classification

Skin cancer is classified by two main types: melanoma and non-melanoma.  Currently, 2-3 million non-melanoma and 132,000 melanoma skin cancers are diagnosed globally each year. The incidence of skin cancer is evident by a statistic based on the World Health Organization (WHO) that states that every 1 in 3 cancers diagnosed is a skin cancer. The increased incidence of skin cancer is attributed to the depleted ozone level and as the atmophosphere loses its protective filter function, an increased amount of solar UV radiation will reach the Earth's surface. Additionally, increased reactreation exposure to the sun and history of sunburn are main factors that predispose an individual to the development of skin cancer. These factors are preventable and each individual is responsible to protect themselves from the sun (i.e., using sunblock). 

## Getting Started
### Usage Requirements

Install all dependencies using `pip install -r requirements.txt`

We are using `.env` to manage environment variables such as API tokens. Copy `.env.example` as `.env` and fill in the environment variables accordingly.

### Connecting to the Kaggle API

This repo uses the Kaggle API to fetch the dataset. Follow these steps to access the Kaggle API:

1. Create a Kaggle account
2. Generate an API token from `kaggle.com/USERNAME/account` -- this prompts you to download a `kaggle.json` file which contains the credentials
3. Populate `.env` file with your Kaggle credentials to set them as your environment variable.

We also use `pre-commit` hooks to keep our code clean. Please enable this using `pre-commit install` prior to making any code changes.

## The Skin Cancer MNIST:HAM1000 Dataset
### Skin Lesion Classifications

The dataset includes the following skin lesions:

- Melanocytic nevi (nv) are benign (non-cancerous) neoplasms composed of melanocytes.
- Melanoma is a type of cancer that develops from melanocytes. It is the most dangerous type of skin cancer and is more common among men than women and high rates are found in Northern Europe and North America in areas mostly populated with white people.
- Benign keratosis-like lesions are non-cancerous skin growths. They are often found on the back or chest and they grow slowly either in groups or singly. It is very likely that everyone will develop one keratosis-like lesion in their lifetime.
- Basal cell carcinoma is a type of skin cancer that begins in the basal cells (which produce new skin cells) and occurs frequently in areas exposed to the sun (e.g., head or neck).
- Actinic keratoses is a non-cancerous rough, scaly patch on your skin that develops from years of exposure to the sun. It's most commonly found on your face, lips, ears, back of your hands, forearms, scalp or neck.
- Vascular lesions include acquired lesions (eg, pyogenic granuloma) and those that are present at birth or arise shortly after birth (vascular birthmarks).
- Dermatofibroma is a common cutaneous nodule of unknown etiology that occurs more often in women. It develops frequently in the extremities (i.e., lower legs) and is usually asymptomatic.


In total, there are 7 different classifications and of these 7, there are 2 cancerous conditions.

![cancer](https://user-images.githubusercontent.com/44474067/71536906-d89b5580-28e2-11ea-8ba3-24860f49fced.png)

The distributionof the 7 different classifications in the dataset is visualized as follows:

![cancer 5](https://user-images.githubusercontent.com/44474067/71536999-e9000000-28e3-11ea-8a59-32a9152900bd.png)

There is an issue regarding overfitting of Melanocytic nevi, which will be addressed using image augmentation during the model development process.

### Analysis: Age & Gender

Canadian statistics on melanoma related to melanoma:
- Women age 49 are more likely to develop melanoma than any other cancer excluding breast and thyroid cancer
- Men age 49 are more likely to develop melanoma than any other cancer
- Men >= 50 are more likely to develop melanoma than women

The following is an analysis of the influence of age and gender on melanoma based on this current dataset:

![cancer 4](https://user-images.githubusercontent.com/44474067/71536965-90306780-28e3-11ea-9a28-d01aac4f1de4.png)

As mentioned, women are more likely to develop melanoma age 49 than any other form of cancer (except breast and thryoid). Though there is no comparison to other cancers we can see that women age 40 and lower have a significantly higher chance of developing melanoma. However, after age 50 it is men that have a higher chance of developing melanoma. In fact at certain ages their chance at developing melanoma is nearly double that of women.ollowing exploratory analysis 

