import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common.constants import *
from common.trading_logger import TradingLogger
from common.data_loader_db import DataLoaderDB
from common.utils import get_config
from aiml.prediction_model import PredictionModel


class NaiveBayesPredictionModel(PredictionModel):
    def __init__(self):
        self.__logger = TradingLogger()
        self.__data_loader = DataLoaderDB()
        self.__initialize_paths()
        self.create_table_name()
        self.__model = None

    def __initialize_paths(self):
        self.__config  = get_config('AIML_ROLLING')
        self.__feature_columns = self.__config['FEATURE_COLUMNS']
        self.__symbol = self.__config["SYMBOL"]
        self.__interval = self.__config["INTERVAL"]
        self.__learn_start = self.__config["LEARN_START"]
        self.__learn_end = self.__config["LEARN_END"]
        self.__datapath = os.path.join(os.path.dirname(__file__), self.__config['DATAPATH'])
        self.__filename = f'{self.__symbol}_{self.__interval}_model'

    def get_feature_columns(self):
        return self.__feature_columns

    def create_table_name(self)->str:
        self.__table_name = f'{self.__symbol}_{self.__interval}_market_data_tech'
        return self.__table_name

    def get_data_loader(self):
        return self.__data_loader

    def load_and_prepare_data(self, start_datetime, end_datetime, test_size=0.5, random_state=None):
        df = self.__data_loader.load_data_from_datetime_period(start_datetime, end_datetime, self.__table_name)
        features = df[self.__feature_columns]
        target = np.where(df[COLUMN_CLOSE].shift(-1) > df[COLUMN_EMA], 1, 0)
        return train_test_split(features, target, test_size=test_size, random_state=random_state)

    def load_data_from_db(self, start_datetime, end_datetime, table_name=None):
        """指定された期間のデータをデータベースからロードします。

        Args:
            start_datetime (str): データの開始日時 (YYYY-MM-DD HH:MM:SS形式)。
            end_datetime (str): データの終了日時 (YYYY-MM-DD HH:MM:SS形式)。
            table_name (str): データをロードするテーブル名。

        Returns:
            pd.DataFrame: ロードされたデータ。
        """
        if table_name is None:
            table_name = self.__table_name
        return self.__data_loader.load_data_from_datetime_period(start_datetime, end_datetime, table_name)

    def train(self, x_train, y_train):
        self.__model = GaussianNB()
        self.__model.fit(x_train, y_train)

    def train_with_cross_validation(self, data, targets, n_splits=3):
        kfold = KFold(n_splits=n_splits, shuffle=True)
        scores = []

        for train, test in kfold.split(data, targets):
            self.__model = GaussianNB()
            self.__model.fit(data[train], targets[train])
            scores.append(self.__model.score(data[test], targets[test]))

        return scores

    def evaluate(self, x_test, y_test):
        y_pred = self.__model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        return accuracy, report, conf_matrix

    def predict(self, data):
        return self.__model.predict(data)

    def predict_single(self, data_point):
        #return self.__model.predict(data_point.reshape(1, -1))[0]

        # 特徴量を取得
        # 特徴量を取得
        features = data_point[self.__feature_columns].to_frame().T
        #features = row[self.__feature_columns]
    #    予測を実行
        prediction = self.__model.predict(features)
        return prediction[0]

    def save_model(self):
        model_path = os.path.join(self.__datapath, self.__filename + '.pkl')
        joblib.dump(self.__model, model_path)

    def load_model(self):
        model_path = os.path.join(self.__datapath, self.__filename + '.pkl')
        self.__model = joblib.load(model_path)

    def get_data_period(self, date, period):
        data = self.__data_loader.load_data_from_datetime_period(date, period, self.__table_name)
        return data[self.__feature_columns].to_numpy()



def main():


    model = NaiveBayesPredictionModel()
    x_train, x_test, y_train, y_test  = model.load_and_prepare_data('2020-01-10 00:00:00','2023-01-01 00:00:00')
    model.train(x_train, y_train)


    x_train, x_test, y_train, y_test = model.load_and_prepare_data('2023-01-10 00:00:00',
                                                                        '2024-01-01 00:00:00',
                                                                          test_size=0.2, random_state=None)

    for i in range(0, 10):
        pred = model.predict_single(x_test.iloc[i])
        print(pred)



if __name__ == '__main__':
    main()