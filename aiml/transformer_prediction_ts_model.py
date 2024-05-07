
import os,sys
from typing import Tuple
import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split


# b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得

# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)

from common.constants import *
from mongodb.data_loader_mongo import MongoDataLoader

from aiml.transformer_prediction_rolling_model import TransformerPredictionRollingModel

# ハイパーパラメータの設定
PARAM_LEARNING_RATE = 0.001
PARAM_EPOCHS = 1100

N_SPLITS=4
POSITIVE_THRESHOLD = 100

class TransformerPredictionTSModel(TransformerPredictionRollingModel):
    """Transformerモデルによる時系列データの予測を行うクラスです。

    Attributes:
        d_model (int): 埋め込みの次元数。
        num_heads (int): アテンションヘッド数。
        dff (int): フィードフォワードネットワークの次元数。
        rate (float): ドロップアウト率。
        l2_reg (float): L2正則化の係数。
        model (Sequential): Transformerモデル。
        optimizer (Adam): 最適化アルゴリズム。
        loss_object (SparseCategoricalCrossentropy): 損失関数。
        train_loss (Mean): 訓練時の損失。
        train_accuracy (SparseCategoricalAccuracy): 訓練時の精度。
        test_loss (Mean): テスト時の損失。
        test_accuracy (SparseCategoricalAccuracy): テスト時の精度。
    """

    def __init__(self):
        """モデルの実行を行います。

        Args:
            d_model (int): 埋め込みの次元数。
            num_heads (int): アテンションヘッド数。
            dff (int): フィードフォワードネットワークの次元数。
            rate (float): ドロップアウト率。
            l2_reg (float): L2正則化の係数。
        """
        super().__init__()
        self._dataloader = MongoDataLoader()

    def load_and_prepare_data_time_series(self,
                                          start_date,
                                          end_date,
                                          coll_type,
                                          test_size=0.2,
                                          random_state=None):
        data = self._dataloader.load_data_from_datetime_period(start_date, end_date, coll_type)
        scaled_sequences, targets = self._prepare_sequences_time_series(data, TIME_SERIES_PERIOD-1, self.feature_columns)
        return train_test_split(scaled_sequences,
                                targets,
                                test_size=test_size,
                                random_state=random_state)

    def _prepare_sequences_time_series(self, data,ftime_steps,feature_columns) -> Tuple[np.ndarray, np.ndarray]:
        #  TIME_SERIES_PERIOD-1, self.feature_columns
        filtered_data = data[(data[COLUMN_BB_DIRECTION].isin([BB_DIRECTION_UPPER, BB_DIRECTION_LOWER])) & (data[COLUMN_BB_PROFIT] != 0)]
        # シーケンスとターゲットの生成
        sequences, targets = [], []

        for i in range(len(filtered_data)):
            end_index = filtered_data.index[i]
            start_index = end_index - ftime_steps
            if end_index > len(data):
                break

            sequence = data.loc[start_index:end_index, feature_columns].values
            target = data.loc[end_index, COLUMN_BB_PROFIT] > POSITIVE_THRESHOLD
            sequences.append(sequence)
            targets.append(target)

        # スケーリング
        sequences = np.array(sequences)
        targets = np.array(targets)
        scaled_sequences = np.array([self.scaler.fit_transform(seq) for seq in sequences])
        return scaled_sequences, targets


def main():
    """
    メインの実行関数。
    """
    # ログ情報の初期化


    # モデルの初期化
    model = TransformerPredictionTSModel()

    x_train, x_test, y_train, y_test = model.load_and_prepare_data_time_series(
                                                                    '2021-01-04 00:00:00',
                                                                    '2024-01-01 00:00:00',
                                                                    MARKET_DATA_ML_UPPER)
    # データのロードと前処理
    #x_train, x_test, y_train, y_test = model.load_and_prepare_data('2021-01-01 00:00:00', '2022-02-01 00:00:00')

    # モデルの訓練
        # モデルをクロスバリデーションで訓練します
    cv_scores = model.train_with_cross_validation(
        np.concatenate((x_train, x_test), axis=0),
        np.concatenate((y_train, y_test), axis=0)
    )
    # クロスバリデーションの結果を表示
    for i, score in enumerate(cv_scores):
        print(f'Fold {i+1}: Accuracy = {score[1]}')
    # モデルの評価
    accuracy, report, conf_matrix = model.evaluate(x_test, y_test)
    print(f"Accuracy: {accuracy}")
    print(report)
    print(conf_matrix)


    x_train, x_test, y_train, y_test = model.load_and_prepare_data_time_series(
                                                            '2024-01-01 00:00:00',
                                                            '2024-06-01 00:00:00',
                                                            MARKET_DATA_ML_UPPER,
                                                            test_size=0.95, random_state=None)
    # データのロードと前処理
    #x_train, x_test, y_train, y_test = model.load_and_prepare_data('2021-01-01 00:00:00', '2022-02-01 00:00:00')

    # モデルの訓練
    #model.train(x_train, y_train)

    # モデルの評価
    accuracy, report, conf_matrix = model.evaluate(x_test, y_test)
    print(f"Accuracy: {accuracy}")
    print(report)
    print(conf_matrix)

if __name__ == '__main__':
    main()
