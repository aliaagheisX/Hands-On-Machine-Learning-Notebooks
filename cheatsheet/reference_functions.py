import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle.api_client import ApiClient

def fetch_titanic_dataset():
    titanic_path = "datasets/titanic"
    dataset_path = Path(titanic_path)
    # download dataset
    if not dataset_path.exists():
        api = KaggleApi(ApiClient())
        api.authenticate()
        
        Path("datasets").mkdir(parents=True, exist_ok=True)
        api.competition_download_files("titanic", path="datasets")

        with zipfile.ZipFile(f"{titanic_path}.zip", "r") as zip_file:
            zip_file.extractall(titanic_path)
        
    return pd.read_csv(f"{titanic_path}/train.csv"), pd.read_csv(f"{titanic_path}/test.csv")


def add_tag_name(X, y=None):
    X = X.copy(deep=True)
    unique_tag_names   = ["Don", "Mme",  "Ms",   "Major", "Lady", "Sir", "Mlle", "Col", "Capt", "the Countess", "Jonkheer"]
    replaces_tag_names = ["Mr",  "Miss", "Miss", "Mr",    "Miss", "Mr",  "Miss", "Mr",  "Mr",   "Mrs",          "Mr"]
    X["TagName"] = X["Name"].apply(lambda x :  x[x.find(',')+2: x.find('.')])
    X["TagName"] = X["TagName"].replace(dict(zip(unique_tag_names, replaces_tag_names)))
    return X