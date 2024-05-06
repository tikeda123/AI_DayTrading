import pandas as pd
import os,sys



#from common.data_loader_db import DataLoaderDB
from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import *

class TradingStateData:
    def __init__(self) -> None:
        self._entry_index = 0
        self._exit_index = 0
        self._fx_serial = 0
        self._order_id = 0
        self._bb_direction = None
        self._entry_price = 0.0

    def reset_index(self):
        """
        現在のインデックス、エントリーインデックス、エグジットインデックス、およびFXシリアル番号をリセットします。
        """
        self._entry_index = 0
        self._exit_index = 0
        self._fx_serial = 0
        self._order_id = 0
        self._bb_direction = None
        self._entry_price = 0.0

    @property
    def current_index(self)->int:
        """
        現在のインデックスを取得します。

        Returns:
            int: 現在のインデックス番号。
        """
        return self._current_index

    @current_index.setter
    def current_index(self, index: int):
        """
        現在のデータインデックスを設定します。

        Args:
            index (int): 設定するインデックスの値。
        """
        self._current_index = index

    @property
    def bb_direction(self) -> str:
        """
        Bollinger Bandの方向を取得します。

        Returns:
            str: Bollinger Bandの方向。
        """
        return self._bb_direction

    @bb_direction.setter
    def bb_direction(self, direction: str):
        """
        Bollinger Bandの方向を設定します。

        Args:
            direction (str): Bollinger Bandの方向。
        """
        self._bb_direction = direction

    @property
    def order_id(self) -> int:
        """
        注文IDを取得します。

        Returns:
            int: 注文ID。
        """
        return self._order_id

    @order_id.setter
    def order_id(self, id):
        """
        注文IDを設定します。

        Args:
            order_id (int): 注文ID。
        """
        self._order_id = id

    @property
    def current_index(self) -> int:
        """
        現在のインデックスを取得します。

        Returns:
            int: 現在のインデックス番号。
        """
        return self._current_index

    @current_index.setter
    def current_index(self, index: int):
        """
        現在のデータインデックスを設定します。

        Args:
            index (int): 設定するインデックスの値。
        """
        self._current_index = index

    @property
    def entry_counter(self) -> int:
        """
        エントリーのカウンターを取得します。

        Returns:
            int: 現在のエントリーのカウンター値。
        """
        return self._entry_counter

    @entry_counter.setter
    def entry_counter(self, counter: int):
        """
        エントリーのカウンターを設定します。

        Args:
            counter (int): 設定するカウンターの値。
        """
        self._entry_counter = counter

    @property
    def entry_index(self) -> int:
        """
        エントリーしたトレードのインデックスを取得します。

        Returns:
            int: トレードのエントリーインデックス。
        """
        return self._entry_index

    @entry_index.setter
    def entry_index(self, index: int):
        """
        エントリーしたトレードのインデックスを設定します。

        Args:
            index (int): 設定するインデックスの値。
        """
        self._entry_index = index

    @property
    def exit_index(self) -> int:
        """
        エグジットしたトレードのインデックスを取得します。

        Returns:
            int: トレードのエグジットインデックス。
        """
        return self._exit_index

    @exit_index.setter
    def exit_index(self, index: int):
        """
        エグジットしたトレードのインデックスを設定します。

        Args:
            index (int): 設定するインデックスの値。
        """
        self._exit_index = index

    @property
    def fx_serial(self) -> int:
        """
        FX取引のシリアル番号を取得します。

        Returns:
            int: FX取引のシリアル番号。
        """
        return self._fx_serial

    @fx_serial.setter
    def fx_serial(self, serial: int):
        """
        FX取引のシリアル番号を設定します。

        Args:
            serial (int): 設定するシリアル番号。
        """
        self._fx_serial = serial


class TradingData(MongoDataLoader):
    def __init__(self) -> None:
        super().__init__()
        #self.add_data_columns()
        self.ts = TradingStateData()
        self._current_index = 0

    def get_current_index(self) -> int:
        """
        現在のインデックスを取得します。

        Returns:
            int: 現在のインデックス。
        """
        return self._current_index

    def set_current_index(self, index: int):
        """
        現在のインデックスを設定します。

        Args:
            index (int): 設定するインデックス。
        """
        self._current_index = index

    def add_data_columns(self):
        """
        データフレームに取引分析用の新しいカラムを追加します。
        """

        self.df_new_column(COLUMN_PANDL, 0.0,float)
        self.df_new_column(COLUMN_STATE, None,str)
        self.df_new_column(COLUMN_BB_DIRECTION, None,str)
        self.df_new_column(COLUMN_ENTRY_PRICE, 0.0,float)
        self.df_new_column(COLUMN_EXIT_PRICE,0.0,float)
        self.df_new_column(COLUMN_CURRENT_PROFIT, 0.0,float)
        self.df_new_column(COLUMN_BB_PROFIT, 0.0,float)
        self.df_new_column(COLUMN_PREDICTION, 0,int)
        self.df_new_column(COLUMN_PROFIT_MA, 0.0,float)
        self.df_new_column(COLUMN_ENTRY_TYPE,  None,str)

    def get_value_by_column(self, column_name, index=None) -> float:
        """
        指定されたカラム名とインデックスに対応する価格を取得します。

        Args:
            column_name (str): 価格を取得するカラム名。
            index (int): 価格を取得するインデックス。指定されていない場合は、現在のインデックスが使用されます。

        Returns:
            float: 指定されたカラム名とインデックスに対応する価格。
        """
        if index is None:
            index = self._current_index
        return self.get_df(index, column_name)

    def set_value_by_column(self, column_name, value, index=None):
        """
        指定されたカラム名とインデックスに対応する値を設定します。

        Args:
            column_name (str): 値を設定するカラム名。
            value: 設定する値。
            index (int): 値を設定するインデックス。指定されていない場合は、現在のインデックスが使用されます。
        """
        if index is None:
            index = self._current_index
        self.set_df(index, column_name, value)

    def get_bb_direction(self,index=None) -> str:
        """
        Bollinger Bandsの方向を取得します。

        Returns:
            str: Bollinger Bandsの方向。
        """
        if index is None:
            return self.ts.bb_direction
        return self.get_value_by_column(COLUMN_BB_DIRECTION,index)

    def set_bb_direction(self, direction: str,index=None):
        """
        Bollinger Bandsの方向を設定します。

        Args:
            direction (str): Bollinger Bandsの方向。
        """
        self.ts.bb_direction = direction
        self.set_value_by_column(COLUMN_BB_DIRECTION, direction,index)

    def get_bb_profit(self,index=None) -> float:
        """
        Bollinger Bandsの利益を取得します。

        Returns:
            float: Bollinger Bandsの利益。
        """
        return self.get_value_by_column(COLUMN_BB_PROFIT,index)


    def set_bb_profit(self, profit: float,index=None):
        """
        Bollinger Bandsの利益を設定します。

        Args:
            profit (float): 設定する利益。
        """
        self.set_value_by_column(COLUMN_BB_PROFIT, profit, index)


    def get_current_profit(self,index=None) -> float:
        """
        現在の利益を取得します。

        Returns:
            float: 現在の利益。
        """
        return self.get_value_by_column(COLUMN_CURRENT_PROFIT,index)

    def set_current_profit(self, profit: float,index=None):
        """
        現在の利益を設定します。

        Args:
            profit (float): 設定する利益。
        """
        self.set_value_by_column(COLUMN_CURRENT_PROFIT, profit,index)


    def get_pandl(self,index=None) -> float:
        """
        現在のインデックスでの損益を取得します。

        Returns:
            float: 現在のインデックスでの損益値。
        """
        return self.get_value_by_column(COLUMN_PANDL,index)


    def set_pandl(self, pandl: float,index=None):
        """
        現在のインデックスでの損益を設定します。

        Args:
            pandl (float): 設定する損益値。
        """
        self.set_value_by_column(COLUMN_PANDL, pandl,index)

    def get_high_price(self,index=None) -> float:
        """
        指定されたインデックスでの高値を取得します。

        Args:
            index (int): 高値を取得するインデックス。

        Returns:
            float: 指定されたインデックスでの高値。
        """
        return self.get_value_by_column(COLUMN_HIGH,index)


    def get_low_price(self,index=None) -> float:
        """
        指定されたインデックスでの安値を取得します。

        Args:
            index (int): 安値を取得するインデックス。

        Returns:
            float: 指定されたインデックスでの安値。
        """
        return self.get_value_by_column(COLUMN_LOW,index)


    def get_lower2_price(self,index=None) -> float:
        """
        指定されたインデックスでの2σの価格を取得します。

        Args:
            index (int): 2σの価格を取得するインデックス。

        Returns:
            float: 指定されたインデックスでの2σの価格。
        """
        return self.get_value_by_column(COLUMN_LOWER_BAND2,index)


    def get_upper2_price(self, index=None) -> float:
        """
        指定されたインデックスでの2σの価格を取得します。

        Args:
            index (int): 2σの価格を取得するインデックス。

        Returns:
            float: 指定されたインデックスでの2σの価格。
        """
        return self.get_value_by_column(COLUMN_UPPER_BAND2,index)


    def get_open_price(self,index=None) -> float:
        """
        指定されたインデックスでの始値を取得します。

        Returns:
            float: 指定されたインデックスでの始値。
        """
        return self.get_value_by_column(COLUMN_OPEN,index)

    def get_ema_price(self, index=None) -> float:
        """
        指定されたインデックスでのEMAを取得します。

        Args:
            index (int): EMAを取得するインデックス。

        Returns:
            float: 指定されたインデックスでのEMA。
        """
        return self.get_value_by_column(COLUMN_EMA,index)



    def get_entry_type(self,index=None) -> str:
        """
        指定されたインデックス、または現在のエントリーインデックスのエントリータイプを取得します。

        Args:
            index (int, optional): エントリータイプを取得するインデックス。指定されていない場合は、現在のエントリーインデックスが使用されます。

        Returns:
            str: エントリータイプ。
        """
        return self.get_value_by_column(COLUMN_ENTRY_TYPE,index)

    def set_entry_type(self, entry_type: str,index=None):
        """
        エントリーのタイプを設定します。

        Args:
            entry_type (str): 設定するエントリータイプ。
        """
        self.set_value_by_column(COLUMN_ENTRY_TYPE, entry_type,index)

    def get_prediction(self,index=None) -> int:
        """
        指定されたインデックス、または現在のエントリーインデックスの予測値を取得します。

        Args:
            index (int, optional): 予測値を取得するインデックス。指定されていない場合は、現在のエントリーインデックスが使用されます。

        Returns:
            int: 予測値。
        """
        return self.get_value_by_column(COLUMN_PREDICTION,index)


    def set_prediction(self, prediction: int, index=None):
        """
        予測値を設定します。

        Args:
            prediction (int): 設定する予測値。
        """
        self.set_value_by_column(COLUMN_PREDICTION, prediction,index)

    def get_current_profit(self,index=None) -> float:
        """
        現在の利益を取得します。指定されたインデックス、または現在のデータインデックスの一つ前の利益が返されます。

        Args:
            index (int, optional): 利益を取得するインデックス。指定されていない場合は、現在のデータインデックスが使用されます。

        Returns:
            float: 現在の利益。
        """
        return self.get_value_by_column(COLUMN_CURRENT_PROFIT,index)


    def set_current_profit(self, profit: float, index=None):
        """
        現在の利益を設定します。

        Args:
            profit (float): 設定する利益。
        """
        self.set_value_by_column(COLUMN_CURRENT_PROFIT, profit,index)


    def get_close_price(self ,index=None) -> float:
        """
        指定したインデックスでの終値を取得します。

        Args:
            index (int): 終値を取得するインデックス。

        Returns:
            float: 指定されたインデックスでの終値。
        """
        return self.get_value_by_column(COLUMN_CLOSE ,index)



    def get_current_date(self, index=None) -> str:
        """
        現在のインデックスでの日付を取得します。

        Returns:
            str: 現在のインデックスでの日付。
        """
        return self.get_value_by_column(COLUMN_DATE)


    def get_raw_data(self) -> pd.DataFrame:
        """
        トレーディングデータの生データフレームを取得します。

        Returns:
            pd.DataFrame: トレーディングデータの生データ。
        """
        return self._df


    def get_entry_price(self,index=None) -> float:
        """
        設定されたエントリー価格を取得します。

        Returns:
            float: エントリー価格。
        """
        if index is None:
            return self.ts.entry_price
        return self.get_value_by_column(COLUMN_ENTRY_PRICE,index)

    def set_entry_price(self, price: float,index=None):
        """
        エントリー価格を設定します。

        Args:
            price (float): エントリーする価格。
        """
        self.ts.entry_price = price
        self.set_value_by_column(COLUMN_ENTRY_PRICE, price,index)

    def get_exit_price(self,index=None) -> float:
        """
        設定されたエグジット価格を取得します。

        Returns:
            float: エグジット価格。
        """
        return self.get_value_by_column(COLUMN_EXIT_PRICE,index)

    def set_exit_price(self, price: float,index=None):
        """
        エグジット価格を設定します。

        Args:
            price (float): エグジットする価格。
        """
        self.set_value_by_column(COLUMN_EXIT_PRICE, price)

    def read_state(self, index=None) -> str:
        """
        トレードの状態を記録します。

        Args:
            state (str): トレードの状態。
        """
        return self.get_value_by_column(COLUMN_STATE,index)

    def record_state(self, state: str,index=None):
        """
        トレードの状態を記録します。

        Args:
            state (str): トレードの状態。
        """
        self.set_value_by_column(COLUMN_STATE, state,index)







