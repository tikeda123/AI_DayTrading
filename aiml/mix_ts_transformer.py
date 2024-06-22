import os,sys
from typing import Tuple
import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split,ParameterGrid
from tqdm import tqdm
from typing import Tuple
import numpy as np
from imblearn.over_sampling import SMOTE

# b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得

# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)

from common.constants import *
from mongodb.data_loader_mongo import MongoDataLoader

from aiml.transformer_prediction_ts_model import TransformerPredictionTSModel

def main():
    model = TransformerPredictionTSModel("mix_lower_mlts")


    start_data = "2020-01-01"
    end_data = "2024-01-01"
    db = MongoDataLoader()
    db.set_collection("BTCUSDT_5_market_data_mlts_lower")
    df_root = db.load_data_from_datetime_period(start_data,end_data)

    db.set_collection("BTCUSDT_15_market_data_mlts_lower")
    df_next = db.load_data_from_datetime_period(start_data,end_data)

    df_root = pd.concat([df_root, df_next], ignore_index=True)

    db.set_collection("BTCUSDT_30_market_data_mlts_lower")
    df_next = db.load_data_from_datetime_period(start_data,end_data)

    df_root = pd.concat([df_root, df_next], ignore_index=True)

    db.set_collection("BTCUSDT_60_market_data_mlts_lower")
    df_next = db.load_data_from_datetime_period(start_data,end_data)

    df_root = pd.concat([df_root, df_next], ignore_index=True)
    print(df_root)

    x_train, x_test, y_train, y_test = model.load_and_prepare_data_time_series_mix(df_root)

    cv_scores = model.train_with_cross_validation(
        np.concatenate((x_train, x_test), axis=0),
        np.concatenate((y_train, y_test), axis=0)
    )
    # クロスバリデーションの結果を表示
    for i, score in enumerate(cv_scores):
        print(f'Fold {i+1}: Accuracy = {score[1]}')
    # モデルの評価
    accuracy, report, conf_matrix = model.evaluate(x_test, y_test)
    print(f'Accuracy: {accuracy}')
    print(report)
    print(conf_matrix)
    model.save_model()

    #model.load_model()
    x_train, x_test, y_train, y_test = model.load_and_prepare_data_time_series( '2024-01-01 00:00:00',
                                                                                '2024-06-01 00:00:00',
                                                                                MARKET_DATA_ML_LOWER,
                                                                                test_size=0.9,
                                                                                random_state=None,
                                                                                oversample=False)
    accuracy, report, conf_matrix = model.evaluate(x_test, y_test)
    print(f'Accuracy: {accuracy}')
    print(report)
    print(conf_matrix)






if __name__ == '__main__':
    main()
