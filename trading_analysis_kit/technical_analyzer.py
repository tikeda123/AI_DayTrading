import pandas as pd
import numpy as np
import talib as ta
import os, sys
from talib import MA_Type


# b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得

# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)

from common.trading_logger_db import TradingLoggerDB
from common.config_manager import ConfigManager
from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import *


class TechnicalAnalyzer:
    """
    株式や為替などの金融データに対してテクニカル分析を行うクラスです。
    RSI、ボリンジャーバンド、MACD、DMIなどの指標を計算し、分析結果をデータフレームに追加します。

    Attributes:
        data_loader (DataLoaderDB): データをロードするためのデータローダー。
        config_manager (ConfigManager): 設定情報を管理するコンフィグマネージャー。
        logger (TradingLogger): ログ情報を記録するトレーディングロガー。
        data (pd.DataFrame): 分析対象のデータフレーム。
    """

    def __init__(self,df=None):
        """
        クラスの初期化メソッドです。

        Args:
            df (pd.DataFrame, optional): 分析対象のデータフレーム。デフォルトはNoneです。
            data_loader (DataLoaderDB): データロードに使用するDataLoaderDBのインスタンス。
            config_manager (ConfigManager): 設定管理に使用するConfigManagerのインスタンス。
            trading_logger (TradingLogger): ログ記録に使用するTradingLoggerのインスタンス。
        """
        #self.__data_loader = DataLoaderDB()
        self.__data_loader = MongoDataLoader()
        self.__config_manager = ConfigManager()
        #self.__logger = TradingLoggerDB()


        if df is not None:
            self.__data = df
        else:
            self.__data = None #self.__data_loader.get_raw()

    def load_data_from_datetime_period(self, start_datetime, end_datetime, table_name=None)->pd.DataFrame:
        """
        指定した期間とテーブル名からデータをロードし、データフレームを更新します。

        Args:
            start_datetime (datetime): データロードの開始日時。
            end_datetime (datetime): データロードの終了日時。
            table_name (str, optional): データをロードするテーブルの名前。デフォルトはNoneです。

        Returns:
            pd.DataFrame: ロードしたデータフレーム。
        """
        table_name = self.__data_loader.make_table_name(table_name)
        self.__data_loader.load_data_from_datetime_period(start_datetime, end_datetime, table_name)
        self.__data = self.__data_loader.get_raw()
        return self.__data
    def load_recent_data_from_db(self, table_name=None)->pd.DataFrame:
        """
        データベースから最新のデータをロードし、データフレームを更新します。

        Args:
            table_name (str, optional): データをロードするテーブルの名前。デフォルトはNoneです。

        Returns:
            pd.DataFrame: ロードしたデータフレーム。
        """
        table_name = self.__data_loader.make_table_name(table_name)
        self.__data_loader.load_recent_data_from_db(table_name)
        self.__data = self.__data_loader.get_raw()
        return self.__data

    def load_data_from_db(self, table_name=None)->pd.DataFrame:
        """
        データベースから指定されたテーブル名のデータをロードし、データフレームを更新します。

        Args:
            table_name (str, optional): データをロードするテーブルの名前。デフォルトはNoneです。

        Returns:
            pd.DataFrame: ロードしたデータフレーム。
        """
        """
        table_name = self.__data_loader.make_table_name(table_name)
        self.__data_loader.load_data_from_db(table_name)
        self.__data = self.__data_loader.get_raw()
        """
        self.__data_loader.load_data(MARKET_DATA)
        self.__data = self.__data_loader.get_df_raw()
        return self.__data

    def load_data_from_tech_db(self, table_name=None)->pd.DataFrame:
        """
        テクニカル分析のデータをデータベースからロードし、データフレームを更新します。

        Args:
            table_name (str, optional): データをロードするテーブルの名前。デフォルトはNoneです。

        Returns:
            pd.DataFrame: ロードしたデータフレーム。
        """
        table_name = self.__data_loader.make_table_name_tech()
        #self.__logger.log_verbose_message(table_name)
        self.__data_loader.load_data_from_db(table_name)
        self.__data = self.__data_loader.get_raw()
        return self.__data

    def import_to_db(self,table_name=None):
        """
        分析結果をデータベースにインポートします。

        Args:
            table_name (str, optional): データをインポートするテーブルの名前。デフォルトはNoneです。
        """
        table_name = self.__data_loader.make_table_name_tech(table_name)
        self.__data_loader.import_to_db(dataframe=self.__data,table_name=table_name)

    def analyze(self):
        """
        テクニカル分析を実行し、結果をデータフレームに追加します。

        Returns:
            pd.DataFrame: 分析結果が追加されたデータフレーム。
        """
        self.calculate_rsi()
        self.calculate_bollinger_bands()
        self.calculate_macd()
        self.calculate_dmi()
        self.calculate_volume_moving_average()
        self.calculate_differences()
        self.calculate_sma()
        self.calculate_ema()
        self.calculate_differences_for_time_series()
        self.finalize_analysis()
        return self.__data

    def finalize_analysis(self):
        """
        最終的なデータフレームの整理を行う関数。欠損値の削除やインデックスのリセットなど、
        分析の最後に必要な処理をここで行います。
        """
        self.__data.dropna(inplace=True)
        #nan_rows = self.__data[self.__data.isnull().any(axis=1)]



        self.__data.reset_index(drop=True, inplace=True)
        return self.__data

    def calculate_rsi(self):
        """
        RSIを計算し、データフレームに追加します。
        """
        timeperiod = int(self.__config_manager.get('TECHNICAL', 'RSI', 'TIMEPERIOD'))
        self.__data[COLUMN_RSI] = ta.RSI(self.__data[COLUMN_CLOSE], timeperiod)
        self._truncate_and_add_to_df(self.__data[COLUMN_RSI], COLUMN_RSI)
        #self.__data.dropna(inplace=True)
        #self.__data.reset_index(drop=True, inplace=True)

    def calculate_bollinger_bands(self):
        """
        ボリンジャーバンドを計算し、データフレームに追加します。

        ボリンジャーバンドは、指定された期間にわたる移動平均（中央バンド）、
        およびこの移動平均からの標準偏差の上下の乖離（上部バンドと下部バンド）を計算します。
        このメソッドでは、複数の標準偏差乖離を計算し、それぞれ異なるバンドとしてデータフレームに追加します。
        """

        timeperiod = int(self.__config_manager.get('TECHNICAL', 'BB', 'TIMEPERIOD'))

        for nbdev_multiplier in range(1, 4):
            upper_band, middle_band, lower_band = ta.BBANDS(
                self.__data[COLUMN_CLOSE],
                timeperiod,
                nbdevup=nbdev_multiplier,
                nbdevdn=nbdev_multiplier,
                matype=MA_Type.EMA
            )
            self._truncate_and_add_to_df(upper_band, f'{COLUMN_UPPER_BAND}{nbdev_multiplier}')
            self._truncate_and_add_to_df(lower_band, f'{COLUMN_LOWER_BAND}{nbdev_multiplier}')
            if nbdev_multiplier == 1:
                self._truncate_and_add_to_df(middle_band, COLUMN_MIDDLE_BAND)

        #self.__data.dropna(inplace=True)
        #self.__data.reset_index(drop=True, inplace=True)

    def calculate_macd(self):
        """
        MACDを計算し、データフレームに追加します。
        """
        fastperiod, slowperiod, signalperiod = int(self.__config_manager.get('TECHNICAL', 'MACD', 'FASTPERIOD')), int(self.__config_manager.get('TECHNICAL', 'MACD', 'SLOWPERIOD')), int(self.__config_manager.get('TECHNICAL', 'MACD', 'SIGNALPERIOD'))
        macd, macdsignal, macdhist = ta.MACD(self.__data[COLUMN_CLOSE], fastperiod, slowperiod, signalperiod)
        self._truncate_and_add_to_df(macd, COLUMN_MACD)
        self._truncate_and_add_to_df(macdsignal, COLUMN_MACDSIGNAL)
        self._truncate_and_add_to_df(macdhist, COLUMN_MACDHIST)
        #self.__data.dropna(inplace=True)
        #self.__data.reset_index(drop=True, inplace=True)

    def calculate_dmi(self):
        """
        DMIを計算し、データフレームに追加します。
        """
        timeperiod = int(self.__config_manager.get('TECHNICAL', 'DMI', 'TIMEPERIOD'))
        p_di = ta.PLUS_DI(self.__data[COLUMN_HIGH], self.__data[COLUMN_LOW], self.__data[COLUMN_CLOSE], timeperiod)
        m_di = ta.MINUS_DI(self.__data[COLUMN_HIGH], self.__data[COLUMN_LOW], self.__data[COLUMN_CLOSE], timeperiod)
        adx = ta.ADX(self.__data[COLUMN_HIGH], self.__data[COLUMN_LOW], self.__data[COLUMN_CLOSE], timeperiod)
        adxr = ta.ADXR(self.__data[COLUMN_HIGH], self.__data[COLUMN_LOW], self.__data[COLUMN_CLOSE], timeperiod)
        self._truncate_and_add_to_df(p_di, COLUMN_P_DI)
        self._truncate_and_add_to_df(m_di, COLUMN_M_DI)
        self._truncate_and_add_to_df(adx, COLUMN_ADX)
        self._truncate_and_add_to_df(adxr, COLUMN_ADXR)
        #self.__data.dropna(inplace=True)
        #self.__data.reset_index(drop=True, inplace=True)

    def calculate_volume_moving_average(self):
        """
        出来高の移動平均とその差分を計算し、データフレームに追加します。
        """
        timeperiod = int(self.__config_manager.get('TECHNICAL', 'VOLUME_MA', 'TIMEPERIOD'))
        self.__data[COLUMN_VOLUME_MA] = self.__data[COLUMN_VOLUME].rolling(window=timeperiod).mean()
        self.__data[COLUMN_VOLUME_MA_DIFF] = self.__data[COLUMN_VOLUME_MA].diff()
        self._truncate_and_add_to_df(self.__data[COLUMN_VOLUME_MA_DIFF], COLUMN_VOLUME_MA_DIFF)
        #self.__data.dropna(inplace=True)
        #self.__data.reset_index(drop=True, inplace=True)

    def _truncate_and_add_to_df(self, series, column_name):
        """
        指定されたシリーズの値を小数点以下2桁に丸め、データフレームに新しい列として追加します。

        Args:
            series (pd.Series): 値を丸める対象のシリーズ。
            column_name (str): 追加する列の名前。

        このメソッドは内部で使用され、各種指標をデータフレームに追加する際に値を適切にフォーマットします。
        """
        self.__data[column_name] = np.trunc(series * 100) / 100


    def calculate_differences_for_time_series(self):
        """
        時系列データのための各種指標の差分を計算し、データフレームに追加します。
        """
        self.__data[COLUMN_UPPER_DIFF] = self.__data[COLUMN_UPPER_BAND2] - self.__data[COLUMN_CLOSE]
        self.__data[COLUMN_LOWER_DIFF] = self.__data[COLUMN_LOWER_BAND2] - self.__data[COLUMN_CLOSE]
        self.__data[COLUMN_MIDDLE_DIFF] = self.__data[COLUMN_MIDDLE_BAND] - self.__data[COLUMN_CLOSE]
        self.__data[COLUMN_EMA_DIFF] = self.__data[COLUMN_EMA] - self.__data[COLUMN_CLOSE]
        self.__data[COLUMN_SMA_DIFF] = self.__data[COLUMN_SMA] - self.__data[COLUMN_CLOSE]
        #self.__data[COLUMN_ENTRY_DIFF] = self.__data[COLUMN_ENTRY_PRICE] - self.__data[COLUMN_CLOSE]

        self.__data[COLUMN_RSI_SELL] = self.__data[COLUMN_RSI] - 70
        self.__data[COLUMN_RSI_BUY] = self.__data[COLUMN_RSI] - 30
        self.__data[COLUMN_DMI_DIFF] = self.__data[COLUMN_P_DI] - self.__data[COLUMN_M_DI]
        self.__data[COLUMN_MACD_DIFF] = self.__data[COLUMN_MACD] - self.__data[COLUMN_MACDSIGNAL]

        self.__data[COLUMN_BOL_DIFF] = self.__data[COLUMN_UPPER_BAND2] - self.__data[COLUMN_LOWER_BAND2]

    def calculate_differences(self):
        """
        ボリンジャーバンド、DMI、およびその他の指標の差分を計算し、データフレームに追加します。
        """
        difference = int(self.__config_manager.get('TECHNICAL', 'DIFFERENCE', 'TIMEPERIOD'))
        self.__data[COLUMN_MIDDLE_DIFF] = self.__data[COLUMN_MIDDLE_BAND] - self.__data[COLUMN_MIDDLE_BAND].shift(difference)
        self.__data[COLUMN_BAND_DIFF] = (self.__data[COLUMN_UPPER_BAND2] - self.__data[COLUMN_LOWER_BAND2]) - ((self.__data[COLUMN_UPPER_BAND2].shift(difference) - self.__data[COLUMN_LOWER_BAND2]).shift(difference))
        self.__data[COLUMN_DI_DIFF] = (self.__data[COLUMN_P_DI] - self.__data[COLUMN_M_DI]) - ((self.__data[COLUMN_P_DI].shift(difference) - self.__data[COLUMN_M_DI]).shift(difference))

        self._truncate_and_add_to_df(self.__data[COLUMN_MIDDLE_DIFF], COLUMN_MIDDLE_DIFF)
        self._truncate_and_add_to_df(self.__data[COLUMN_BAND_DIFF], COLUMN_BAND_DIFF)
        self._truncate_and_add_to_df(self.__data[COLUMN_DI_DIFF], COLUMN_DI_DIFF)

    def calculate_sma(self):
        """
        単純移動平均(SMA)を計算し、データフレームに追加します。
        """
        timeperiod = int(self.__config_manager.get('TECHNICAL', 'SMA', 'TIMEPERIOD'))
        self.__data[COLUMN_SMA] = ta.SMA(self.__data[COLUMN_CLOSE], timeperiod)
        self._truncate_and_add_to_df(self.__data[COLUMN_SMA], COLUMN_SMA)
        #self.__data.dropna(inplace=True)
        #self.__data.reset_index(drop=True, inplace=True)

    def calculate_ema(self):
        """
        指数移動平均（EMA）を計算し、データフレームに追加します。
        """
        timeperiod = int(self.__config_manager.get('TECHNICAL', 'EMA', 'TIMEPERIOD'))
        self.__data[COLUMN_EMA] = ta.EMA(self.__data[COLUMN_CLOSE], timeperiod)
        self._truncate_and_add_to_df(self.__data[COLUMN_EMA], COLUMN_EMA)
        #self.__data.dropna(inplace=True)
        #self.__data.reset_index(drop=True, inplace=True)

    def _truncate_and_add_to_df(self, series, column_name):
        """
        指定されたシリーズの値を小数点以下2桁に丸め、データフレームに新しい列として追加します。

        Args:
            series (pd.Series): 値を丸める対象のシリーズ。
            column_name (str): 追加する列の名前。

        このメソッドは内部で使用され、各種指標をデータフレームに追加する際に値を適切にフォーマットします。
        """
        self.__data[column_name] = np.trunc(series * 100) / 100

    def get_data_loader(self):
        """
        使用中のデータローダーを取得します。

        Returns:
            DataLoaderDB: 使用中のデータローダー。
        """
        return self.__data_loader

    def get_raw(self):
        """
        現在のデータフレームを取得します。

        Returns:
            pd.DataFrame: 現在のデータフレーム。
        """
        return self.__data

def main():

    analyzer = TechnicalAnalyzer()
    table_name = 'BTCUSDT_240_market_data'
    analyzer.load_data_from_db(table_name)
    result = analyzer.analize()
    analyzer.import_to_db()
    print(result)

if __name__ == '__main__':
    main()

