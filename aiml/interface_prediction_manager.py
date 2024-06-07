import os
import sys
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import operator

# 親ディレクトリのパスを設定
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common.utils import get_config
from common.trading_logger import TradingLogger
from common.constants import *
from common.utils import get_config

from aiml.prediction_model import PredictionModel

PREDICTION_TIMEPERIOD = 7


class InterfacePredictionRollingManager:
    """
    時系列データに対するローリング予測を管理するクラス。

    設定管理、ロギング、予測モデルの初期化、データの読み込みと準備、
    モデルの訓練、クロスバリデーション、予測、評価、モデルの保存と読み込みを行う。
    """

    def __init__(self,symbol=None,interval=None):
        """
        InferencePredictionRollingManagerの初期化を行う。

        Args:
            config (Dict[str, Any]): 設定値のディクショナリ。
        """
        config = get_config("AIML_ROLLING")

        if symbol is None:
            self.symbol = get_config('SYMBOL')
        else:
            self.symbol = symbol

        if interval is None:
            self.interval = get_config('INTERVAL')
        else:
            self.interval = interval


        self.table_name = f'{self.symbol}_{self.interval}_market_data'
        self.table_name_tech = f'{self.symbol}_{self.interval}_market_data_tech'
        self.logger = TradingLogger()
        self.prediction_model: PredictionModel = None

    def initialize_model(self, id, model_class):
        """
        指定されたモデルクラスを初期化する。

        Args:
            model_class (type): 初期化するモデルクラス。
        """
        self.prediction_model = model_class(id, self.symbol, self.interval)

    def load_and_prepare_data(self,
                              start_datetime: str,
                              end_datetime: str,
                              coll_type: str,
                              test_size: float = 0.5,
                              random_state: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        指定された期間のデータを読み込み、前処理を行い、訓練データとテストデータに分割する。

        Args:
            start_datetime (str): データ読み込みの開始日時。
            end_datetime (str): データ読み込みの終了日時。
            test_size (float): テストデータの割合。
            random_state (int, optional): 分割時の乱数シード。

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 訓練データ、テストデータ、訓練ラベル、テストラベル。
        """
        self.rl_x_train, self.rl_x_test, self.rl_y_train, self.rl_y_test = self.prediction_model.load_and_prepare_data(start_datetime, end_datetime, coll_type,test_size=test_size, random_state=random_state)
        return self.rl_x_train, self.rl_x_test, self.rl_y_train, self.rl_y_test

    def load_and_prepare_data_time_series(self,
                                          start_datetime,
                                          end_datetime,
                                          coll_type,
                                          test_size=0.2,
                                          random_state=None):
        self.rl_x_train, self.rl_x_test, self.rl_y_train, self.rl_y_test = self.prediction_model.load_and_prepare_data_time_series(start_datetime, end_datetime, coll_type,test_size=test_size, random_state=random_state)
        return self.rl_x_train, self.rl_x_test, self.rl_y_train, self.rl_y_test

    def train_models(self, x_train: np.ndarray=None, y_train: np.ndarray=None):
        """
        モデルを訓練する。

        Args:
            x_train (np.ndarray): 訓練データ。
            y_train (np.ndarray): 訓練データのラベル。
        """
        if x_train is None or y_train is None:
            x_train = self.rl_x_train
            y_train = self.rl_y_train
        self.prediction_model.train(x_train, y_train)

    def train_with_cross_validation(self,
                                    x_train: np.ndarray=None,
                                    y_train: np.ndarray=None,
                                    x_test: np.ndarray=None,
                                    y_test: np.ndarray=None):
        """
        クロスバリデーションを使用してモデルを訓練し、結果を表示する。

        Args:
            x_data (np.ndarray): 訓練データ。
            y_data (np.ndarray): 訓練データのラベル。
        """
        if x_train is None or y_train is None:
            x_train = self.rl_x_train
            y_train = self.rl_y_train

        if x_test is None or y_test is None:
            x_test = self.rl_x_test
            y_test = self.rl_y_test


        # クロスバリデーションを実行
        cv_scores =  self.prediction_model.train_with_cross_validation(
            np.concatenate((x_train, x_test), axis=0),
            np.concatenate((y_train, y_test), axis=0)
        )

        # クロスバリデーションの結果を表示
        for i, score in enumerate(cv_scores):
            print(f'Fold {i+1}: Accuracy = {score}')

    def predict_period_model(self, date: str) -> np.ndarray:
        """
        指定された期間のデータに対して予測を行う。

        Args:
            date (str): 予測を行う期間の開始日時。

        Returns:
            np.ndarray: 予測結果。
        """
        data = self.prediction_model.get_data_period(date, PREDICTION_TIMEPERIOD)
        return self.prediction_model.predict(data)

    def predict_rolling_model_date(self, feature_date: str) -> int:
        """
        指定された日時のデータに対してローリング予測を行う。

        Args:
            feature_date (str): 予測を行うデータの日時。

        Returns:
            int: 予測結果（1が上昇、0が下降）。
        """
        data_loader = self.prediction_model.get_data_loader()
        df = data_loader.filter(COLUMN_START_AT, operator.eq, feature_date)

        if df.empty:
            self.logger.log_warning("No data found")
            return 0

        data_frame = data_loader.get_df_fromto(df.index[0] - (TIME_SERIES_PERIOD - 1), df.index[0])
        target_df = self.create_time_series_data(data_frame)

        prediction = self.predict_model(target_df)
        return prediction

    def predict_model(self, data_point: np.ndarray) -> int:
        """
        単一のデータポイントに対して予測を行う。

        Args:
            data_point (np.ndarray): 予測するデータポイント。

        Returns:
            int: 予測結果（1が上昇、0が下降）。
        """
        return self.prediction_model.predict_single(data_point)

    def evaluate_models(self, x_test: np.ndarray=None, y_test: np.ndarray=None):
        """
        モデルを評価し、結果をロギングする。

        Args:
            x_test (np.ndarray): テストデータ。
            y_test (np.ndarray): テストデータのラベル。
        """
        if x_test is None or y_test is None:
            x_test = self.rl_x_test
            y_test = self.rl_y_test

        accuracy, report, conf_matrix = self.prediction_model.evaluate(x_test, y_test)
        self.logger.log_debug_message(f"Rolling_model Model, Accuracy: {accuracy}")
        self.logger.log_debug_message(report)
        self.logger.log_debug_message(conf_matrix)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        予測を行う。

        Args:
            x (np.ndarray): 予測するデータ。

        Returns:
            np.ndarray: 予測結果。
        """
        return self.prediction_model.predict(x)

    def save_model(self):
        """
        訓練済みモデルを保存する。
        """
        self.prediction_model.save_model()

    def load_model(self):
        """
        保存されたモデルを読み込む。
        """
        self.prediction_model.load_model()

    def create_time_series_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        DataFrameから時系列データを生成する。

        Args:
            df (pd.DataFrame): 時系列データを含むDataFrame。

        Returns:
            np.ndarray: 生成された時系列データ。
        """
        feature_columns = self.prediction_model.get_feature_columns()
        sequence = df.filter(items=feature_columns)
        return sequence.to_numpy()


def init_inference_prediction_rolling_manager(id,
                                              model_class,
                                              symbol=None,
                                              interval=None) -> InterfacePredictionRollingManager:
    """
    InferencePredictionRollingManagerインスタンスを初期化し、指定されたモデルクラスを初期化する。

    Args:
        model_class (type): 初期化するモデルクラス。

    Returns:
        InferencePredictionRollingManager: 初期化されたInferencePredictionRollingManagerインスタンス。
    """
    manager = InterfacePredictionRollingManager(symbol,interval)
    manager.initialize_model(id,model_class)
    return manager

def main():

    #from aiml.transformer_prediction_ts_model import TransformerPredictionTSModel
    from aiml.transformer_prediction_rolling_model import TransformerPredictionRollingModel

    manager = init_inference_prediction_rolling_manager("rolling",TransformerPredictionRollingModel)
    manager.load_and_prepare_data("2023-01-01 00:00:00", "2024-01-01 00:00:00",MARKET_DATA_TECH,test_size=0.2, random_state=None)
    #manager.train_models()
    manager.train_with_cross_validation()
    manager.save_model()

    #manager.load_model()
    manager.load_and_prepare_data("2024-01-01 00:00:00", "2024-06-01 00:00:00",MARKET_DATA_TECH,test_size=0.9, random_state=None)

    manager.evaluate_models()

    # テストデータを取得

if __name__ == '__main__':
    main()