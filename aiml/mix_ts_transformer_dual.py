import os
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Tuple

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow version:", tf.__version__)
print("Is GPU available: ", tf.test.is_gpu_available())
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# GPUの設定
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is available and configured")
    except RuntimeError as e:
        print(e)
else:
    print("GPU is not available, using CPU")

# b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得

# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)

from common.constants import *
from mongodb.data_loader_mongo import MongoDataLoader
from aiml.transformer_prediction_ts_model import TransformerPredictionTSModel

def create_mirrored_strategy():
    # 利用可能なすべてのGPUとCPUを使用する分散戦略を作成
    devices = tf.config.list_logical_devices('GPU') + tf.config.list_logical_devices('CPU')
    return tf.distribute.MirroredStrategy(devices=devices)

def main():
    # 分散戦略を作成
    strategy = create_mirrored_strategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    model = TransformerPredictionTSModel("mix_upper_mlts")

    start_data = "2020-01-01"
    end_data = "2024-01-01"
    db = MongoDataLoader()
    db.set_collection("BTCUSDT_5_market_data_mlts_upper")
    df_root = db.load_data_from_datetime_period(start_data, end_data)

    collections = [
        "BTCUSDT_15_market_data_mlts_upper",
        "BTCUSDT_30_market_data_mlts_upper",
        "BTCUSDT_60_market_data_mlts_upper",
        "BTCUSDT_120_market_data_mlts_upper",
        "BTCUSDT_240_market_data_mlts_upper",
        "BTCUSDT_720_market_data_mlts_upper"
    ]

    for collection in collections:
        db.set_collection(collection)
        df_next = db.load_data_from_datetime_period(start_data, end_data)
        df_root = pd.concat([df_root, df_next], ignore_index=True)

    print(df_root)

    # 分散戦略のスコープ内でデータの準備とモデルのトレーニングを行う
    with strategy.scope():
        x_train, x_test, y_train, y_test = model.load_and_prepare_data_time_series_mix(df_root)

        # 目標精度を設定
        target_accuracy = 0.70
        max_iterations = 20  # 最大反復回数
        current_iteration = 0

        while current_iteration < max_iterations:
            print(f"Training iteration {current_iteration + 1}")

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

            # 新しいデータでの評価
            x_new, y_new = model.load_and_prepare_data_time_series(
                '2024-01-01 00:00:00',
                '2024-06-01 00:00:00',
                MARKET_DATA_ML_LOWER,
                test_size=1.0,  # すべてのデータをテストデータとして使用
                random_state=None,
                oversample=False
            )
            new_accuracy, new_report, new_conf_matrix = model.evaluate(x_new, y_new)
            print(f'New data accuracy: {new_accuracy}')
            print(new_report)

            if new_accuracy >= target_accuracy:
                print(f"Target accuracy {target_accuracy} achieved. Stopping training.")
                break

            current_iteration += 1

        if current_iteration == max_iterations:
            print(f"Maximum iterations {max_iterations} reached without achieving target accuracy.")

    model.save_model()

if __name__ == "__main__":
    main()
