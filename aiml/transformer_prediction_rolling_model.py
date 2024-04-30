
import os,sys
from typing import Dict, Tuple
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.decomposition import PCA
import joblib



# b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得

# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)

from common.trading_logger import TradingLogger
from common.data_loader_db import DataLoaderDB
from common.utils import get_config
from common.constants import *

from aiml.prediction_model import PredictionModel


# ハイパーパラメータの設定
PARAM_LEARNING_RATE = 0.001
PARAM_EPOCHS = 100

POSITIVE_THRESHOLD = 0

def step_decay(epoch):
    """学習率を段階的に減少させる関数です。

    Args:
        epoch (int): 現在のエポック数。

    Returns:
        float: 新しい学習率。
    """
    initial_lr = 0.001
    drop = 0.5
    epochs_drop = 20.0
    lr = initial_lr * (drop ** np.floor((1+epoch)/epochs_drop))
    return lr

class TransformerBlock(tf.keras.layers.Layer):
    """Transformerモデルのブロックを表すクラスです。

    Args:
        d_model (int): 埋め込みの次元数。
        num_heads (int): アテンション機構のヘッド数。
        dff (int): フィードフォワードネットワークの次元数。
        rate (float): ドロップアウト率。
        l2_reg (float): L2正則化の係数。

    Attributes:
        mha (MultiHeadAttention): マルチヘッドアテンション層。
        ffn (Sequential): フィードフォワードネットワーク層。
        layernorm1 (LayerNormalization): 最初のレイヤー正規化層。
        layernorm2 (LayerNormalization): 二番目のレイヤー正規化層。
        dropout1 (Dropout): 最初のドロップアウト層。
        dropout2 (Dropout): 二番目のドロップアウト層。
    """
    def __init__(self, d_model, num_heads, dff, rate=0.1, l2_reg=0.01):
        """モデルの実行を行います。

        Args:
            x (Tensor): 入力テンソル。
            training (bool): トレーニングモードかどうか。

        Returns:
            Tensor: 出力テンソル。
        """
        super().__init__()
        self.mha = MultiHeadAttention(key_dim=d_model, num_heads=num_heads)
        # L2正規化をDense層に適用
        self.ffn = Sequential([
            Dense(dff, activation='relu', kernel_regularizer=l2(l2_reg)),
            Dense(d_model, kernel_regularizer=l2(l2_reg))
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training=False):
        """モデルの実行を行います。

        Args:
            x (Tensor): 入力テンソル。
            training (bool): トレーニングモードかどうか。

        Returns:
            Tensor: 出力テンソル。
        """
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

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
        self.__logger = TradingLogger()
        self.__data_loader = DataLoaderDB()
        self.__config = get_config("AIML_ROLLING")

        if symbol is None:
            self.__symbol =  get_config("SYMBOL")
        else:
            self.__symbol = symbol

        if interval is None:
            self.__interval =  get_config("INTERVAL")
        else:
            self.__interval = interval


        self.__initialize_paths()
        self.__scaler = MinMaxScaler(feature_range=(0, 1))
        self.create_table_name()
        self.__model = None

    def __initialize_paths(self):
        """
        パス関連の初期化を行う。
        """
        self.__datapath = parent_dir + '/' + self.__config['DATAPATH']
        self.__feature_columns = self.__config['FEATURE_COLUMNS']
        self.__target_column = self.__config["TARGET_COLUMN"]
        self.__filename = f'{self.__symbol}_{self.__interval}_{self.__target_column}_model'

    def get_data_loader(self) -> DataLoaderDB:
        """
        データローダーを取得する。

        Returns:
            DataLoaderDB: データローダーのインスタンス。
        """
        return self.__data_loader

    def get_feature_columns(self) -> list:
        """
        使用する特徴量カラムを取得する。

        Returns:
            list: 特徴量カラムのリスト。
        """
        return self.__feature_columns

    def create_table_name(self) -> str:
        """
        テーブル名を作成する。

        Returns:
            str: 作成されたテーブル名。
        """
        self.__table_name = f'{self.__symbol}_{self.__interval}_market_data_tech'
        return self.__table_name

    def load_and_prepare_data(self, start_datetime, end_datetime, test_size=0.5, random_state=None):
        """指定された期間のデータをロードし、前処理を行った後に訓練データとテストデータに分割します。

        Args:
            start_datetime (str): データの開始日時 (YYYY-MM-DD HH:MM:SS形式)。
            end_datetime (str): データの終了日時 (YYYY-MM-DD HH:MM:SS形式)。
            test_size (float): テストデータセットの割合。
            random_state (int, optional): データ分割時のランダムシード。

        Returns:
            tuple: 訓練用データセットとテスト用データセット。
        """
        #start_datetime (str): データの開始日時 (YYYY-MM-DD HH:MM:SS形式)
        #end_datetime (str): データの終了日時 (YYYY-MM-DD HH:MM:SS形式)
        data = self.__data_loader.load_data_from_datetime_period(start_datetime, end_datetime,self.__table_name)
        sequences = []
        targets = []

        for i in range(len(data) - (TIME_SERIES_PERIOD +1)):
            sequence = data.loc[i:i+TIME_SERIES_PERIOD-1, self.__feature_columns].values  # 7日間のデータ
            target = int(data.loc[i+TIME_SERIES_PERIOD, self.__target_column[0]] > data.loc[i+TIME_SERIES_PERIOD-1, self.__target_column[1]])  # 8日目のcloseが7日目より高ければ1、そうでなければ0
            sequences.append(sequence)
            targets.append(target)

        sequences = np.array(sequences)
        targets = np.array(targets)

        # 特徴量のスケーリング
        scaled_sequences = np.array([self.__scaler.fit_transform(seq) for seq in sequences])
        return train_test_split(scaled_sequences, targets, test_size=test_size, random_state=random_state)



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

    def train(self, x_train, y_train, epochs=200, batch_size=32):
        """モデルを訓練します。

        Args:
            x_train (np.array): 訓練データ。
            y_train (np.array): 訓練データのラベル。
            epochs (int): エポック数。
            batch_size (int): バッチサイズ。
        """
        # モデルのトレーニング
        self.__model = self.create_transformer_model((x_train.shape[1], x_train.shape[2]))
        self.__model.compile(optimizer=Adam(learning_rate=PARAM_LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
        self.__model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)


    def train_with_cross_validation(self, data: np.ndarray, targets: np.ndarray, epochs: int = PARAM_EPOCHS, batch_size: int = 32, n_splits: int = 2) -> list:
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
            self.__model = self.create_transformer_model((data.shape[1], data.shape[2]))
            # モデルをコンパイル
            self.__model.compile(optimizer=Adam(learning_rate=PARAM_LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
            # モデルをトレーニング
            self.__logger.log_system_message(f'Training for fold {fold_no} ...')
            self.__model.fit(data[train], targets[train], epochs=epochs, batch_size=batch_size)

            # モデルの性能を評価
            scores.append(self.__model.evaluate(data[test], targets[test], verbose=0))

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
        y_pred = (self.__model.predict(x_test) > 0.5).astype(int)
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

        return self.__model.predict(data)

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
        #self.__scaler.scaler = MinMaxScaler(feature_range=(0, 1))
        data_point = data_point[self.__feature_columns].values
        scaled_data_point = self.__scaler.fit_transform(data_point)
        # モデルによる予測
        prediction = self.__model.predict(scaled_data_point.reshape(1, -1, len(self.__feature_columns)))
        prediction = (prediction > 0.5).astype(int)
        return prediction[0][0]

    def save_model(self):
        """
        訓練済みのモデルとスケーラーをファイルに保存する。
        """
        # モデルの保存
        model_file_name = self.__filename + '.keras'
        model_path = os.path.join(self.__datapath, model_file_name)
        self.__logger.log_system_message(f"Saving model to {model_path}")
        self.__model.save(model_path)

        # スケーラーの保存
        model_scaler_file = self.__filename + '.scaler'
        model_scaler_path = os.path.join(self.__datapath, model_scaler_file)
        self.__logger.log_system_message(f"Saving scaler to {model_scaler_path}")
        joblib.dump(self.__scaler, model_scaler_path)

    def load_model(self):
        """
        保存されたモデルとスケーラーを読み込む。
        """
        # モデルの読み込み
        model_file_name = self.__filename + '.keras'
        model_path = os.path.join(self.__datapath, model_file_name)
        self.__logger.log_system_message(f"Loading model from {model_path}")
        self.__model = tf.keras.models.load_model(model_path, custom_objects={'TransformerBlock': TransformerBlock})

        # スケーラーの読み込み
        model_scaler_file = self.__filename + '.scaler'
        model_scaler_path = os.path.join(self.__datapath, model_scaler_file)
        self.__logger.log_system_message(f"Loading scaler from {model_scaler_path}")
        self.__scaler = joblib.load(model_scaler_path)

    def get_data_period(self, date: str, period: int) -> np.ndarray:
        """
        指定された期間のデータを取得する。

        Args:
            date (str): 期間の開始日時。
            period (int): 期間の長さ。

        Returns:
            np.ndarray: 指定された期間のデータ。
        """
        data = self.__data_loader.load_data_from_datetime_period(date, period, self.__table_name)
        return data.filter(items=self.__feature_columns).to_numpy()
