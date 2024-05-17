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

from trading_analysis_kit.trading_data import TradingStateData
from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import *

class TradingDataManager:
    def __init__(self):
        self.dataloader = MongoDataLoader()
        self.state_data = TradingStateData()

        self.reset_index()

    def get_df_fromto(self, start_index: int, end_index: int) -> pd.DataFrame:
        """
        指定されたインデックスの範囲のデータフレームを取得します。

        Args:
            start_index (int): 開始インデックス。
            end_index (int): 終了インデックス。

        Returns:
            pd.DataFrame: 指定されたインデックスの範囲のデータフレーム。
        """
        return self.dataloader.get_df_fromto(start_index, end_index)

    def set_df_fromto(self,start_index: int, end_index: int, col,value):
        """
        指定されたインデックスの範囲のデータフレームを設定します。

        Args:
            df (pd.DataFrame): 設定するデータフレーム。
            start_index (int): 開始インデックス。
            end_index (int): 終了インデックス。
        """
        self.dataloader.set_df_fromto(start_index, end_index, col,value)

    def is_first_column_less_than_second(self, column1: str, column2: str, index: int = None) -> bool:
        """
        指定された2つのカラムの値を比較し、最初のカラムの値が2番目のカラムの値よりも小さいかどうかを判定します。

        Args:
            column1 (str): 比較する最初のカラムの名前。
            column2 (str): 比較する2番目のカラムの名前。
            index (int): インデックス番号。

        Returns:
            bool: 最初のカラムの値が2番目のカラムの値よりも小さいかどうかの真偽値。
        """
        if index is None:
            index = self.state_data.current_index
        return self.dataloader.get_df(index, column1) < self.dataloader.get_df(index, column2)

    def is_first_column_greater_than_second(self, column1: str, column2: str, index: int = None) -> bool:
        """
        指定された2つのカラムの値を比較し、最初のカラムの値が2番目のカラムの値よりも大きいかどうかを判定します。

        Args:
            column1 (str): 比較する最初のカラムの名前。
            column2 (str): 比較する2番目のカラムの名前。
            index (int): インデックス番号。

        Returns:
            bool: 最初のカラムの値が2番目のカラムの値よりも大きいかどうかの真偽値。
        """
        if index is None:
            index = self.state_data.current_index
        return self.dataloader.get_df(index, column1) > self.dataloader.get_df(index, column2)

    def load_data_from_datetime_period(self, symbol: str, start_date: str, end_date: str):
        self.dataloader.load_data_from_datetime_period(symbol, start_date, end_date)
        self.add_data_columns()

    def get_current_index(self) -> int:
        """
        現在のインデックスを取得します。

        Returns:
            int: 現在のインデックス番号。
        """
        return self.state_data.get_current_index()

    def set_current_index(self, index: int):
        """
        現在のデータインデックスを設定します。

        Args:
            index (int): 設定するインデックスの値。
        """
        self.state_data.set_current_index(index)


    def get_order_id(self) -> int:
        """
        注文IDを取得します。

        Returns:
            int: 注文ID。
        """
        return self.state_data.get_order_id()

    def set_order_id(self, id):
        """
        注文IDを設定します。

        Args:
            order_id (int): 注文ID。
        """
        self.state_data.set_order_id(id)

    def get_entry_counter(self) -> int:
        """
        エントリーのカウンターを取得します。

        Returns:
            int: 現在のエントリーのカウンター値。
        """
        return self.state_data.get_entry_counter()

    def set_entry_counter(self, counter: int):
        """
        エントリーのカウンターを設定します。

        Args:
            counter (int): 設定するカウンターの値。
        """
        self.state_data.set_entry_counter(counter)

    def get_entry_index(self) -> int:
        """
        エントリーしたトレードのインデックスを取得します。

        Returns:
            int: トレードのエントリーインデックス。
        """
        return self.state_data.get_entry_index()

    def set_entry_index(self, index: int):
        """
        エントリーしたトレードのインデックスを設定します。

        Args:
            index (int): 設定するインデックスの値。
        """
        self.state_data.set_entry_index(index)

    def get_exit_index(self) -> int:
        """
        エグジットしたトレードのインデックスを取得します。

        Returns:
            int: トレードのエグジットインデックス。
        """
        return self.state_data.get_exit_index()

    def set_exit_index(self, index: int):
        """
        エグジットしたトレードのインデックスを設定します。

        Args:
            index (int): 設定するインデックスの値。
        """
        self.state_data.set_exit_index(index)

    def get_fx_serial(self) -> int:
        """
        FX取引のシリアル番号を取得します。

        Returns:
            int: FX取引のシリアル番号。
        """
        return self.state_data.get_fx_serial()

    def set_fx_serial(self, serial: int):
        """
        FX取引のシリアル番号を設定します。

        Args:
            serial (int): 設定するシリアル番号。
        """
        self.state_data.set_fx_serial(serial)

    def add_data_columns(self):
        """
        データフレームに取引分析用の新しいカラムを追加します。
        """

        self.dataloader.df_new_column(COLUMN_PANDL, 0.0,float)
        self.dataloader.df_new_column(COLUMN_STATE, None,str)
        self.dataloader.df_new_column(COLUMN_BB_DIRECTION, None,str)
        self.dataloader.df_new_column(COLUMN_ENTRY_PRICE, 0.0,float)
        self.dataloader.df_new_column(COLUMN_EXIT_PRICE,0.0,float)
        self.dataloader.df_new_column(COLUMN_CURRENT_PROFIT, 0.0,float)
        self.dataloader.df_new_column(COLUMN_BB_PROFIT, 0.0,float)
        self.dataloader.df_new_column(COLUMN_PREDICTION, 0,int)
        self.dataloader.df_new_column(COLUMN_PROFIT_MA, 0.0,float)
        self.dataloader.df_new_column(COLUMN_ENTRY_TYPE,  None,str)
        self.dataloader.df_new_column(COLUMN_MAX_PANDL, 0.0,float)
        self.dataloader.df_new_column(COLUMN_MIN_PANDL, 0.0,float)
        self.dataloader.df_new_column(COLUMN_EXIT_REASON, None,str)




    def get_raw_data(self) -> pd.DataFrame:
        """
        トレーディングデータの生データフレームを取得します。

        Returns:
            pd.DataFrame: トレーディングデータの生データ。
        """
        return self.dataloader.get_df_raw()

    def get_value_by_column(self, column_name: str, index: int=None) -> float:
        if index is None:
            index = self.state_data.get_current_index()
        return self.dataloader.get_df(index, column_name)

    def set_value_by_column(self, column_name: str, value, index: int=None):
        if index is None:
            index = self.state_data.get_current_index()
        return self.dataloader.set_df(index, column_name, value)
        #self.dataloader.set_value_by_column(column_name, value, index)

    def get_max_pandl(self, index: int = None) -> float:
        return self.get_value_by_column(COLUMN_MAX_PANDL,index)

    def set_max_pandl(self, price: float, index: int = None):
        self.set_value_by_column(COLUMN_MAX_PANDL,price, index)

    def get_min_pandl(self, index: int = None) -> float:
        return self.get_value_by_column(COLUMN_MIN_PANDL,index)

    def set_min_pandl(self, price: float, index: int = None):
        self.set_value_by_column(COLUMN_MIN_PANDL,price, index)

    def get_bb_direction(self, index: int = None) -> str:
        if index is None:
            return self.state_data.get_bb_direction()
        return self.get_value_by_column(COLUMN_BB_DIRECTION, index)

    def set_bb_direction(self, direction: str, index: int = None):
        self.state_data.set_bb_direction(direction)
        self.set_value_by_column(COLUMN_BB_DIRECTION, direction, index)

    def get_bb_profit(self, index: int = None) -> float:
        return self.get_value_by_column(COLUMN_BB_PROFIT,index)

    def set_bb_profit(self, profit: float, index: int = None):
        self.set_value_by_column(COLUMN_BB_PROFIT,profit, index)

    def get_current_profit(self, index: int = None) -> float:
        return self.get_value_by_column(COLUMN_CURRENT_PROFIT,index)

    def set_current_profit(self, profit: float, index: int = None):
        self.set_value_by_column(COLUMN_CURRENT_PROFIT,profit, index)

    def get_pandl(self, index: int = None) -> float:
        return self.get_value_by_column(COLUMN_PANDL,index)

    def set_pandl(self, pandl: float, index: int = None):
        self.set_value_by_column(COLUMN_PANDL,pandl, index)

    def get_high_price(self, index: int = None) -> float:
        return self.get_value_by_column(COLUMN_HIGH,index)

    def get_low_price(self, index: int = None) -> float:
        return self.get_value_by_column(COLUMN_LOW,index)

    def get_lower2_price(self, index: int = None) -> float:
        return self.get_value_by_column(COLUMN_LOWER_BAND2,index)

    def get_upper2_price(self, index: int = None) -> float:
        return self.get_value_by_column(COLUMN_UPPER_BAND2,index)

    def get_open_price(self, index: int = None) -> float:
        return self.get_value_by_column(COLUMN_OPEN,index)

    def get_middle_price(self, index: int = None) -> float:
        return self.get_value_by_column(COLUMN_MIDDLE_BAND,index)

    def get_ema_price(self, index: int = None) -> float:
        return self.get_value_by_column(COLUMN_EMA,index)

    def get_entry_type(self, index: int = None) -> str:
        if index is None:
            return self.state_data.get_entry_type()
        return self.get_value_by_column(COLUMN_ENTRY_TYPE,index)

    def set_entry_type(self, entry_type: str, index: int = None):
        self.state_data.set_entry_type(entry_type)
        self.set_value_by_column(COLUMN_ENTRY_TYPE,entry_type, index)

    def get_prediction(self, index: int = None) -> int:
        if index is None:
            return self.state_data.get_prediction()
        return self.get_value_by_column(COLUMN_PREDICTION,index)

    def set_prediction(self, prediction: int, index: int = None):
        self.state_data.set_prediction(prediction)
        self.set_value_by_column(COLUMN_PREDICTION,prediction, index)

    def get_close_price(self, index: int = None) -> float:
        return self.get_value_by_column(COLUMN_CLOSE,index)

    def get_current_date(self, index: int = None) -> str:
        return self.get_value_by_column(COLUMN_DATE,index)

    def get_entry_price(self, index: int = None) -> float:
        if index is None:
            return self.state_data.get_entry_price()
        return self.get_value_by_column(COLUMN_ENTRY_PRICE,index)

    def set_entry_price(self, price: float, index: int = None):
        self.state_data.set_entry_price(price)
        self.set_value_by_column(COLUMN_ENTRY_PRICE,price, index)

    def get_exit_price(self, index: int = None) -> float:
        return self.get_value_by_column(COLUMN_EXIT_PRICE,index)

    def set_exit_price(self, price: float, index: int = None):
        self.set_value_by_column(COLUMN_EXIT_PRICE,price, index)

    def read_state(self, index: int = None) -> str:
        return self.get_value_by_column(COLUMN_STATE,index)

    def record_state(self, state: str, index: int = None):
        self.set_value_by_column(COLUMN_STATE,state, index)

    def set_exit_reason(self, reason: str, index: int = None):
        self.set_value_by_column(COLUMN_EXIT_REASON,reason, index)

    def reset_index(self):
        self.state_data.reset_index()

