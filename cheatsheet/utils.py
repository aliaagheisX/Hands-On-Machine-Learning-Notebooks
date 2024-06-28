import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import zipfile
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix

"""
series_groupby = train_data[["TagName", "AgeBucket", "Embarked"]].groupby("TagName").agg(pd.Series.mode)
"""
def fill_na_(X, series_groupby):
    X = X.copy(deep=True)
    index_colname = series_groupby.index.name
    
    for index_value in series_groupby.index:
        for column_name in series_groupby.columns:
            msk = X[index_colname] == index_value
            fill_val = series_groupby.loc[index_value, column_name]
            X.loc[msk, column_name] = X.loc[msk, column_name].fillna(fill_val)
    return X

def get_mertric_result(y_true, y_pred):
    result = {}
    metric_names = ["Accuracy", "Precision", "Recall", "F1Score", "ROC_AUC", "ConfusionMatrix"]
    metric_funcs = [accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix]


    for i, name in enumerate(metric_names):
        result[name] = metric_funcs[i](y_true, y_pred)
    return result


def save_model(model, model_name, X_, y_):
    save_model.lst_model_index +=  1
    # get result csv to save
    result = get_mertric_result(y_, model.predict(X_))
    result['ModelName'] = model_name
    result = pd.DataFrame([result])
    # create directory to save file
    model_path = f'models/{save_model.lst_model_index:04}_{model_name}'
    Path(model_path).mkdir(parents=True, exist_ok=True)
    # save model and result
    result.to_csv(model_path + "/result.csv")
    joblib.dump(model, model_path + "/model.joblib")
    # return result
    return result, model_path
save_model.lst_model_index = 0
    
def read_model(model_path):
    model = joblib.load(model_path + "/model.joblib")
    model_result = pd.read_csv(model_path + "/result.csv")

    return model, model_result