
import os,sys
from typing import Tuple
import numpy as np
import tensorflow as tf
import pandas as pd
import shap

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, GlobalAveragePooling1D
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import matplotlib.pyplot as plt

import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Flatten, Dense, Dropout, GlobalAveragePooling1D, Input,
    MultiHeadAttention, LayerNormalization
)

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
PARAM_LEARNING_RATE = 0.0001
PARAM_EPOCHS = 600

N_SPLITS=4

POSITIVE_THRESHOLD = 0


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

    def __init__(self,id: str,symbol=None,interval=None):
        """
        TransformerPredictionRollingModelの初期化を行う。

        Args:
            config (Dict[str, Any]): モデルの設定値のディクショナリ。
        """
        self.id = id
        self.logger = TradingLogger()
        self.data_loader = MongoDataLoader()
        if id == 'upper_mlts':
            self.config = get_config("AIML_ROLLING_UPPER")
        elif id == 'lower_mlts':
            self.config = get_config("AIML_ROLLING_LOWER")
        else:
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
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.create_table_name()
        self.model = None

    def initialize_paths(self):
        """
        パス関連の初期化を行う。
        """
        self.datapath = parent_dir + '/' + self.config['DATAPATH']
        self.feature_columns = self.config['FEATURE_COLUMNS']
        self.target_column = self.config["TARGET_COLUMN"]
        self.filename = f'{self.id}_{self.symbol}_{self.interval}_model'

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
       return train_test_split(scaled_sequences, targets, test_size=test_size, random_state=random_state, shuffle=False)

    def load_and_prepare_data(self,
                              start_datetime,
                              end_datetime,
                              coll_type,
                              test_size=0.5,
                              random_state=None):

       data = self.data_loader.load_data_from_datetime_period(
                                               start_datetime,
                                                end_datetime,
                                                coll_type)
       scaled_sequences, targets = self._prepare_sequences(data)

       return train_test_split(scaled_sequences,
                                targets,
                                test_size=test_size,
                                random_state=random_state,shuffle=False)

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


    def create_cnn_transformer_model(
        self, input_shape, num_heads=32, dff=256, rate=0.1, l2_reg=0.01, num_transformer_blocks=4,
        num_filters=64, kernel_size=3, pool_size=2
    ):
        """
        CNNとTransformerを組み合わせたモデルを作成する。
        """

        # CNN部分
        inputs = Input(shape=input_shape)
        x = Conv1D(num_filters, kernel_size, activation='relu', padding='same')(inputs)
        x = MaxPooling1D(pool_size)(x)
        x = Conv1D(num_filters * 2, kernel_size, activation='relu', padding='same')(x)
        x = MaxPooling1D(pool_size)(x)

        # Transformer部分
        x = LayerNormalization(epsilon=1e-6)(x)  # LayerNormalizationを追加
        for _ in range(num_transformer_blocks):
            x = TransformerBlock(x.shape[-1], num_heads, dff, rate, l2_reg=l2_reg)(x)

        # 出力層
        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.2)(x)
        x = Dense(160, activation='relu', kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(0.2)(x)
        x = Dense(80, activation='relu', kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(0.2)(x)
        x = Dense(40, activation='relu', kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu', kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation='relu', kernel_regularizer=l2(l2_reg))(x)
        outputs = Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_reg))(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def create_transformer_model(self, input_shape, num_heads=16, dff=256, rate=0.1, l2_reg=0.01,num_transformer_blocks=3):
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
        x = inputs
        for _ in range(num_transformer_blocks):
            x = TransformerBlock(input_shape[1], num_heads, dff, rate, l2_reg=l2_reg)(x)
        x = GlobalAveragePooling1D()(x)

        x = Dropout(0.2)(x)
        x = Dense(80, activation='relu', kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(0.2)(x)
        x = Dense(40, activation='relu', kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu', kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation='relu', kernel_regularizer=l2(l2_reg))(x)
        x = Flatten()(x)

        outputs = Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_reg))(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def train(self, x_train, y_train, epochs=PARAM_EPOCHS, batch_size=32):
        """モデルを訓練します。

        Args:
            x_train (np.array): 訓練データ。
            y_train (np.array): 訓練データのラベル。
            epochs (int): エポック数。
            batch_size (int): バッチサイズ。
        """
        # モデルのトレーニング
        #self.model = self.create_transformer_model((x_train.shape[1], x_train.shape[2]))
        self.model = self.create_cnn_transformer_model((x_train.shape[1], x_train.shape[2]))
        self.model.compile(optimizer=Adam(learning_rate=PARAM_LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)


    def train_with_cross_validation(self,
                                    data: np.ndarray,
                                    targets: np.ndarray,
                                    epochs: int = PARAM_EPOCHS,
                                    batch_size: int = 32,
                                    n_splits: int = N_SPLITS) -> list:
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
            #self.model = self.create_transformer_model((data.shape[1], data.shape[2]))
            self.model = self.create_cnn_transformer_model((data.shape[1], data.shape[2]))
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
        print("Predict function input shape:", data.shape)
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

        #data_point = data_point[self.feature_columns].values
        #print(data_point)

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
        '''
        # モデルの読み込み
        model_file_name = self.filename + '.keras'
        model_path = os.path.join(self.datapath, model_file_name)
        self.logger.log_system_message(f"Loading model from {model_path}")
        self.model = tf.keras.models.load_model(model_path, custom_objects={'TransformerBlock': TransformerBlock})
        '''
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

    def compute_gradcam(self, data, layer_name='transformer_block_4', pred_index=None):
        grad_model = tf.keras.models.Model([self.model.inputs], [self.model.get_layer(layer_name).output, self.model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(data)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def display_gradcam(self, data, heatmap, alpha=0.4):
        plt.figure(figsize=(10, 6))
        plt.plot(data.flatten(), label='Input Data')
        plt.imshow(np.expand_dims(heatmap, axis=0), aspect='auto', cmap='viridis', alpha=alpha)
        plt.colorbar(label='Importance')
        plt.xlabel('Time Step')
        plt.ylabel('Feature Index')
        plt.title('Grad-CAM Visualization')
        plt.legend()
        plt.show()

def permutation_feature_importance(model, X, y, metric, n_repeats=10):
    baseline_score = metric(y, (model.predict(X) > 0.5).astype(int).flatten())
    feature_importances = []

    for column in range(X.shape[2]):
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, :, column])
            score = metric(y, (model.predict(X_permuted) > 0.5).astype(int).flatten())
            scores.append(score)
        feature_importances.append(baseline_score - np.mean(scores))

    return np.array(feature_importances)

def drop_column_feature_importance(model, X, y, metric):
    baseline_score = metric(y, (model.predict(X) > 0.5).astype(int).flatten())
    feature_importances = []

    for column in range(X.shape[2]):
        X_dropped = X.copy()
        X_dropped[:, :, column] = 0
        score = metric(y, (model.predict(X_dropped) > 0.5).astype(int).flatten())
        feature_importances.append(baseline_score - score)

    return np.array(feature_importances)

def main():

    model = TransformerPredictionRollingModel("rolling")
    x_train, x_test, y_train, y_test = model.load_and_prepare_data(
                                                                    '2020-01-01 00:00:00',
                                                                    '2024-01-01 00:00:00',
                                                                    MARKET_DATA_TECH)
    # データのロードと前処理
    #x_train, x_test, y_train, y_test = model.load_and_prepare_data('2021-01-01 00:00:00', '2022-02-01 00:00:00')
    #model.load_model(0
    # モデルの訓練
        # モデルをクロスバリデーションで訓練します
    """
    param_grid = {
        'num_heads': [4, 8, 16],
        'dff': [128, 256, 512],
        'rate': [0.1, 0.2, 0.3],
        'l2_reg': [0.001, 0.01, 0.1]
    }

    # ハイパーパラメータの探索とモデルの訓練
    best_params = model.train_with_hyperparameter_tuning(
        np.concatenate((x_train, x_test), axis=0),
        np.concatenate((y_train, y_test), axis=0),
        param_grid
    )

    print(f"Best hyperparameters: {best_params['best_params']}")
    print(f"Best validation loss: {best_params['best_val_loss']}")
    """


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
        # モデルのレイヤー名を取得する
    #layer_names = [layer.name for layer in model.model.layers]
    #print("Available layers:", layer_names)

    #x_train, x_test, y_train, y_test = model.load_and_prepare_data(
    #                                                        '2023-01-01 00:00:00',
    #                                                        '2024-01-01 00:00:00',
    #                                                        MARKET_DATA_TECH,
    #                                                        test_size=0.9, random_state=None)
    # データのロードと前処理
    #x_train, x_test, y_train, y_test = model.load_and_prepare_data('2021-01-01 00:00:00', '2022-02-01 00:00:00')

    # モデルの訓練
    #model.train(x_train, y_train)
    #model.load_model()
    # モデルの評価
    #accuracy, report, conf_matrix = model.evaluate(x_test, y_test)
    #print(f"Accuracy: {accuracy}")
    #print(report)
    #print(conf_matrix)



    #model.load_model()

    x_train, x_test, y_train, y_test = model.load_and_prepare_data('2024-01-01 00:00:00', '2024-06-01 00:00:00', MARKET_DATA_TECH, test_size=0.9, random_state=None)
    accuracy, report, conf_matrix = model.evaluate(x_test, y_test)
    print(f"Accuracy: {accuracy}")
    print(report)
    print(conf_matrix)


        #print(f"Prediction for data point {i}: {model.predict_single(x_test[i])}")

    """
        # 異なるレイヤー名を試す
    for layer_name in layer_names:
        print(f"Grad-CAM for layer: {layer_name}")
        try:
            heatmap = model.compute_gradcam(x_test, layer_name=layer_name)
            model.display_gradcam(x_test[0], heatmap)
            grads = check_gradients(model.model, x_test, layer_name)
            mean_grad = tf.reduce_mean(tf.abs(grads))
            print(f"Checking gradients for layer: {layer_name}")
            print(f"Mean gradient for {layer_name}: {mean_grad}")
        except Exception as e:
            print(f"Could not compute Grad-CAM for layer {layer_name}: {e}")




    # Permutation Feature Importanceの計算
    permutation_importances = permutation_feature_importance(model.model, x_test, y_test, accuracy_score)
     # 特徴量重要度の可視化
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(permutation_importances)), permutation_importances)
    plt.xlabel('Feature Index')
    plt.ylabel('Permutation Feature Importance')
    plt.title('Permutation Feature Importance')
    plt.tight_layout()
    plt.show()

    # Permutation Feature Importanceの計算
    permutation_importances = drop_column_feature_importance(model, x_test, y_test, accuracy_score)
    # 特徴量重要度の可視化
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(permutation_importances)), permutation_importances)
    plt.xlabel('Feature Index')
    plt.ylabel('Permutation Feature Importance')
    plt.title('Permutation Feature Importance')
    plt.tight_layout()
    plt.show()


def main():



    # モデルの初期化
    model = TransformerPredictionRollingModel("rolling")
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
"""
if __name__ == '__main__':
    main()


