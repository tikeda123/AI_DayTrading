
import os,sys
from typing import Tuple
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import joblib


# b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得

# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)

from common.trading_logger import TradingLogger
from mongodb.data_loader_mongo import MongoDataLoader
from common.utils import get_config
from common.constants import *

from aiml.prediction_model import PredictionModel
from aiml.transformerblock import TransformerBlock

# ハイパーパラメータの設定
PARAM_LEARNING_RATE = 0.001
PARAM_EPOCHS = 1100

N_SPLITS=4
POSITIVE_THRESHOLD = 100


class TransformerPredictionRollingModel(PredictionModel):
    """
    Transformerを使用してローリング予測を行うモデルクラス。

    Args:
        config (Dict[str, Any]): モデルの設定値のディクショナリ。

    Attributes:
        __logger (TradingLogger): ログ情報を扱うインスタンス。
        __data_loader (DataLoaderDB): データロードを扱うインスタンス。
        __config (Dict[str, Any]): モデル設定値のディクショナリ。
        __datapath (str): データパス。
        __feature_columns (list): 特徴量カラム。
        __symbol (str): シンボル名。
        __interval (str): データの間隔。
        __filename (str): ファイル名。
        __target_column (str): ターゲットカラム。
        __scaler (MinMaxScaler): スケーリングに使用するインスタンス。
        __table_name (str): データテーブル名。
        __model (tf.keras.Model): Transformerモデル。
    """

    def __init__(self,symbol=None,interval=None):
        """
        TransformerPredictionRollingModelの初期化を行う。

        Args:
            config (Dict[str, Any]): モデルの設定値のディクショナリ。
        """
        self.logger = TradingLogger()
        self.data_loader = MongoDataLoader()
        self.config = get_config("AIML_ROLLING")

        if symbol is None:
            self.symbol =  get_config("SYMBOL")
        else:
            self.symbol = symbol

        if interval is None:
            self.interval =  get_config("INTERVAL")
        else:
            self.interval = interval


        self.initialize_paths()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.create_table_name()
        self.model = None

    def initialize_paths(self):
        """
        パス関連の初期化を行う。
        """
        self.datapath = parent_dir + '/' + self.config['DATAPATH']
        self.feature_columns = self.config['FEATURE_COLUMNS']
        self.target_column = self.config["TARGET_COLUMN"]
        self.filename = f'{self.symbol}_{self.interval}_{self.target_column}_model'

    def get_data_loader(self) -> MongoDataLoader:
        """
        データローダーを取得する。

        Returns:
            DataLoaderDB: データローダーのインスタンス。
        """
        return self.data_loader

    def get_feature_columns(self) -> list:
        """
        使用する特徴量カラムを取得する。

        Returns:
            list: 特徴量カラムのリスト。
        """
        return self.feature_columns

    def create_table_name(self) -> str:
        """
        テーブル名を作成する。

        Returns:
            str: 作成されたテーブル名。
        """
        self.table_name = f'{self.symbol}_{self.interval}_market_data_tech'
        return self.table_name

    def load_data_from_csv(self, learning_datafile: str):
        """
        指定されたCSVファイルからデータを読み込み、pandas DataFrameとして返します。

        このメソッドは、学習データファイルのパスを組み立て、pandasを使用してCSVファイルを読み込みます。
        ファイルが存在しない場合、FileNotFoundErrorをキャッチし、ログに記録した後に例外を再発生させます。

        Args:
            learning_datafile (str): 読み込む学習データのCSVファイル名。このファイル名は、内部で保持されているデータパスに追加されます。

        Returns:
            pd.DataFrame: 読み込まれたデータを含むDataFrame。

        Raises:
            FileNotFoundError: 指定されたCSVファイルが見つからない場合に発生します。
        """
        data_path = self.datapath + learning_datafile
        try:
            df = pd.read_csv(data_path)
        except FileNotFoundError as e:
            self.logger.log_system_message(f"FileNotFoundError: {e}")
            raise
        return df


    def load_and_prepare_data_from_csv(self, csv_file_path, test_size=0.5, random_state=None):
       data = self.load_data_from_csv(csv_file_path)
       scaled_sequences, targets = self._prepare_sequences(data)
       return train_test_split(scaled_sequences, targets, test_size=test_size, random_state=random_state)

    def load_and_prepare_data(self,
                              start_datetime,
                              end_datetime,
                              test_size=0.5,
                              random_state=None):

       data = self.data_loader.load_data_from_datetime_period(
                                               start_datetime,
                                            end_datetime,
                                            self.table_name)
       scaled_sequences, targets = self._prepare_sequences(data)

       return train_test_split(scaled_sequences,
                                targets,
                                test_size=test_size,
                                random_state=random_state)

    def _prepare_sequences(self, data):
       """データからシーケンスとターゲットを準備します。

       Args:
           data (pd.DataFrame): 入力データのDataFrame。

       Returns:
           tuple: シーケンスとターゲットのNumPy配列のタプル。
       """
       sequences = []
       targets = []

       for i in range(len(data) - (TIME_SERIES_PERIOD + 1)):
           sequence = data.iloc[i:i+TIME_SERIES_PERIOD, data.columns.get_indexer(self.feature_columns)].values
           target = int(data.iloc[i+TIME_SERIES_PERIOD][self.target_column[0]] > data.iloc[i+TIME_SERIES_PERIOD-1][self.target_column[1]])
           sequences.append(sequence)
           targets.append(target)

       sequences = np.array(sequences)
       targets = np.array(targets)

       # 特徴量のスケーリング
       scaled_sequences = np.array([self.scaler.fit_transform(seq) for seq in sequences])
       return scaled_sequences, targets


    def create_transformer_model(self, input_shape, num_heads=8, dff=256, rate=0.1, l2_reg=0.01):
        """
        Transformerモデルを作成する。

        Args:
            input_shape (tuple): 入力データの形状。
            num_heads (int): アテンション機構のヘッド数。
            dff (int): フィードフォワードネットワークの次元数。
            rate (float): ドロップアウト率。
            l2_reg (float): L2正則化の係数。

        Returns:
            tf.keras.Model: 作成されたモデル。
        """
        inputs = tf.keras.Input(shape=input_shape)
        x = TransformerBlock(input_shape[1], num_heads, dff, rate, l2_reg=l2_reg)(inputs)
        x = Dropout(0.2)(x)
        x = Dense(20, activation='relu', kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(0.2)(x)
        x = Dense(10, activation='relu', kernel_regularizer=l2(l2_reg))(x)
        x = Flatten()(x)
        outputs = Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_reg))(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def train(self, x_train, y_train, epochs=1200, batch_size=32):
        """モデルを訓練します。

        Args:
            x_train (np.array): 訓練データ。
            y_train (np.array): 訓練データのラベル。
            epochs (int): エポック数。
            batch_size (int): バッチサイズ。
        """
        # モデルのトレーニング
        self.model = self.create_transformer_model((x_train.shape[1], x_train.shape[2]))
        self.model.compile(optimizer=Adam(learning_rate=PARAM_LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)


    def train_with_cross_validation(self,
                                    data: np.ndarray,
                                    targets: np.ndarray,
                                    epochs: int = PARAM_EPOCHS,
                                    batch_size: int = 32,
                                    n_splits: int = 2) -> list:
        """
        K-Foldクロスバリデーションを用いてモデルを訓練する。

        Args:
            data (np.ndarray): 訓練に使用するデータセット。
            targets (np.ndarray): データセットに対応するターゲット値。
            epochs (int): エポック数。
            batch_size (int): バッチサイズ。
            n_splits (int): クロスバリデーションの分割数。

        Returns:
            list: 各分割でのテストデータに対するモデルの性能評価結果。
        """
        # K-Foldクロスバリデーションを初期化
        kfold = KFold(n_splits=n_splits, shuffle=True)

        # 各フォールドでのスコアを記録するリスト
        fold_no = 1
        scores = []

        for train, test in kfold.split(data, targets):
            # モデルを生成
            self.model = self.create_transformer_model((data.shape[1], data.shape[2]))
            # モデルをコンパイル
            self.model.compile(optimizer=Adam(learning_rate=PARAM_LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
            # モデルをトレーニング
            self.logger.log_system_message(f'Training for fold {fold_no} ...')
            self.model.fit(data[train], targets[train], epochs=epochs, batch_size=batch_size)

            # モデルの性能を評価
            scores.append(self.model.evaluate(data[test], targets[test], verbose=0))

            fold_no += 1

        return scores

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, str, np.ndarray]:
        """
        モデルをテストデータセットで評価する。

        Args:
            x_test (np.ndarray): テストデータセット。
            y_test (np.ndarray): テストデータセットの正解ラベル。

        Returns:
            Tuple[float, str, np.ndarray]: モデルの正解率、分類レポート、混同行列。
        """
        # モデルの評価
        y_pred = (self.model.predict(x_test) > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        return accuracy, report, conf_matrix

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        指定されたデータに対して予測を行う。

        Args:
            data (np.ndarray): 予測対象のデータ。

        Returns:
            np.ndarray: 予測結果。
        """

        return self.model.predict(data)

    def predict_single(self, data_point: np.ndarray) -> int:
        """
        単一のデータポイントに対して予測を行う。

        Args:
            data_point (np.ndarray): 予測対象のデータポイント。

        Returns:
            int: 予測されたクラスラベル。
        """
        # 単一データポイントの予測
        # 予測用スケーラーを使用
        #self.scaler.scaler = MinMaxScaler(feature_range=(0, 1))
        data_point = data_point[self.feature_columns].values
        scaled_data_point = self.scaler.fit_transform(data_point)
        # モデルによる予測
        prediction = self.model.predict(scaled_data_point.reshape(1, -1, len(self.feature_columns)))
        prediction = (prediction > 0.5).astype(int)
        return prediction[0][0]

    def save_model(self):
        """
        訓練済みのモデルとスケーラーをファイルに保存する。
        """
        # モデルの保存
        model_file_name = self.filename + '.keras'
        model_path = os.path.join(self.datapath, model_file_name)
        self.logger.log_system_message(f"Saving model to {model_path}")
        self.model.save(model_path)

        # スケーラーの保存
        model_scaler_file = self.filename + '.scaler'
        model_scaler_path = os.path.join(self.datapath, model_scaler_file)
        self.logger.log_system_message(f"Saving scaler to {model_scaler_path}")
        joblib.dump(self.scaler, model_scaler_path)

    def load_model(self):
        """
        保存されたモデルとスケーラーを読み込む。
        """
        # モデルの読み込み
        model_file_name = self.filename + '.keras'
        model_path = os.path.join(self.datapath, model_file_name)
        self.logger.log_system_message(f"Loading model from {model_path}")
        self.model = tf.keras.models.load_model(model_path, custom_objects={'TransformerBlock': TransformerBlock})

        # スケーラーの読み込み
        model_scaler_file = self.filename + '.scaler'
        model_scaler_path = os.path.join(self.datapath, model_scaler_file)
        self.logger.log_system_message(f"Loading scaler from {model_scaler_path}")
        self.scaler = joblib.load(model_scaler_path)

    def get_data_period(self, date: str, period: int) -> np.ndarray:
        """
        指定された期間のデータを取得する。

        Args:
            date (str): 期間の開始日時。
            period (int): 期間の長さ。

        Returns:
            np.ndarray: 指定された期間のデータ。
        """
        data = self.data_loader.load_data_from_datetime_period(date, period, self.table_name)
        return data.filter(items=self.feature_columns).to_numpy()


def main():
    """
    メインの実行関数。
    """
    # ログ情報の初期化


    # モデルの初期化
    model = TransformerPredictionRollingModel()
    #learning_datafile = 'BTCUSDT_20210101000_20230901000_60_price_upper_mlts.csv'
    learning_datafile = 'BTCUSDT_20200101_20230101_60_price_lower_mlts.csv'
    x_train, x_test, y_train, y_test = model.load_and_prepare_data_time_series_from_csv(learning_datafile)
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


    test_datafile = 'BTCUSDT_20230901000_20231201000_60_price_upper_mlts.csv'
    x_train, x_test, y_train, y_test = model.load_and_prepare_data_time_series_from_csv(learning_datafile)
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
