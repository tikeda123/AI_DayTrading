import os
import sys
from abc import ABC, abstractmethod
from typing import Any,Tuple

import numpy as np
import pandas as pd

# 親ディレクトリのパスを設定
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)



PREDICTION_TIMEPERIOD = 7


class PredictionModel(ABC):
    """
    予測モデルの抽象クラス。
    """

    @abstractmethod
    def load_and_prepare_data(self, start_datetime: str, end_datetime: str, test_size: float = 0.5, random_state: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        指定された期間のデータを読み込み、前処理を行い、訓練データとテストデータに分割する。
        """
        pass

    @abstractmethod
    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        """
        モデルを訓練する。
        """
        pass

    @abstractmethod
    def train_with_cross_validation(self, x_data: np.ndarray, y_data: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> Any:
        """
        クロスバリデーションを使用してモデルを訓練する。
        """
        pass

    @abstractmethod
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, str, np.ndarray]:
        """
        モデルを評価する。
        """
        pass

    @abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        指定されたデータに対して予測を行う。
        """
        pass

    @abstractmethod
    def predict_single(self, data_point: np.ndarray) -> int:
        """
        単一のデータポイントに対して予測を行う。
        """
        pass

    @abstractmethod
    def save_model(self):
        """
        訓練済みモデルを保存する。
        """
        pass

    @abstractmethod
    def load_model(self):
        """
        保存されたモデルを読み込む。
        """
        pass

    @abstractmethod
    def get_data_loader(self):
        """
        データローダーを取得する。
        """
        pass

    @abstractmethod
    def get_feature_columns(self) -> list:
        """
        使用する特徴量カラムを取得する。
        """
        pass

    @abstractmethod
    def get_data_period(self, date: str, period: int) -> np.ndarray:
        """
        指定された期間のデータを取得する。
        """
        pass

