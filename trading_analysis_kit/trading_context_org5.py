import pandas as pd
import os,sys


from common.trading_logger_db import TradingLoggerDB
from common.config_manager import ConfigManager
from common.data_loader_db import DataLoaderDB
from common.constants import *

from trading_analysis_kit.trading_state import IdleState


class TradingContext:
    """
    トレーディングのコンテキストを管理するクラスです。取引に関連する各種コンポーネントとの連携を担います。

    Attributes:
        _state (IdleState): 現在の取引状態。
        config_manager (ConfigManager): 設定管理コンポーネント。
        trading_logger (TradingLogger): 取引ログ記録用コンポーネント。
        dataloader (DataLoaderDB): データロード用コンポーネント。
        strategy (Any): 取引戦略コンテキスト。

    """
    def __init__(self, strategy_context):

        self._state = IdleState()
        self.config_manager = ConfigManager()
        self.trading_logger = TradingLoggerDB()
        self.dataloader = DataLoaderDB()
        self.strategy = strategy_context
        self.reset_index()
        #self.add_data_columns()


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


    def get_state(self):
        """
        トレーディングの状態を設定します。

        Args:
            new_state (State): 新しい状態オブジェクト。
        """
        return self._state

    def set_state(self,state):
        """
        トレーディングの状態を取得します。

        Returns:
            State: 現在の状態オブジェクト。
        """
        self._state = state

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
        self.set_value_by_column(COLUMN_BB_DIRECTION, direction)

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
            index = self.current_index
        return self.dataloader.get_df(index, column_name)

    def set_value_by_column(self, column_name, value, index=None):
        """
        指定されたカラム名とインデックスに対応する値を設定します。

        Args:
            column_name (str): 値を設定するカラム名。
            value: 設定する値。
            index (int): 値を設定するインデックス。指定されていない場合は、現在のインデックスが使用されます。
        """
        if index is None:
            index = self.current_index
        self.dataloader.set_df(index, column_name, value)


    @property
    def bb_profit(self) -> float:
        """
        Bollinger Bandsの利益を取得します。

        Returns:
            float: Bollinger Bandsの利益。
        """
        return self.get_value_by_column(COLUMN_BB_PROFIT)

    @bb_profit.setter
    def bb_profit(self, profit: float):
        """
        Bollinger Bandsの利益を設定します。

        Args:
            profit (float): 設定する利益。
        """
        self.set_value_by_column(COLUMN_BB_PROFIT, profit)

    @property
    def current_profit(self) -> float:
        """
        現在の利益を取得します。

        Returns:
            float: 現在の利益。
        """
        return self.get_value_by_column(COLUMN_CURRENT_PROFIT)

    @current_profit.setter
    def current_profit(self, profit: float):
        """
        現在の利益を設定します。

        Args:
            profit (float): 設定する利益。
        """
        self.set_value_by_column(COLUMN_CURRENT_PROFIT, profit)

    @property
    def pandl(self) -> float:
        """
        現在のインデックスでの損益を取得します。

        Returns:
            float: 現在のインデックスでの損益値。
        """
        return self.get_value_by_column(COLUMN_PANDL)

    @pandl.setter
    def pandl(self, pandl: float):
        """
        現在のインデックスでの損益を設定します。

        Args:
            pandl (float): 設定する損益値。
        """
        self.set_value_by_column(COLUMN_PANDL, pandl)

    @property
    def high_price(self) -> float:
        """
        指定されたインデックスでの高値を取得します。

        Args:
            index (int): 高値を取得するインデックス。

        Returns:
            float: 指定されたインデックスでの高値。
        """
        return self.get_value_by_column(COLUMN_HIGH)


    @property
    def low_price(self) -> float:
        """
        指定されたインデックスでの安値を取得します。

        Args:
            index (int): 安値を取得するインデックス。

        Returns:
            float: 指定されたインデックスでの安値。
        """
        return self.get_value_by_column(COLUMN_LOW)

    @property
    def lower2_price(self) -> float:
        """
        指定されたインデックスでの2σの価格を取得します。

        Args:
            index (int): 2σの価格を取得するインデックス。

        Returns:
            float: 指定されたインデックスでの2σの価格。
        """
        return self.get_value_by_column(COLUMN_LOWER_BAND2)

    @property
    def upper2_price(self, index=None) -> float:
        """
        指定されたインデックスでの2σの価格を取得します。

        Args:
            index (int): 2σの価格を取得するインデックス。

        Returns:
            float: 指定されたインデックスでの2σの価格。
        """
        return self.get_value_by_column(COLUMN_UPPER_BAND2)

    @property
    def open_price(self) -> float:
        """
        指定されたインデックスでの始値を取得します。

        Returns:
            float: 指定されたインデックスでの始値。
        """
        return self.get_value_by_column(COLUMN_OPEN)

    def get_ema_price(self, index=None) -> float:
        """
        指定されたインデックスでのEMAを取得します。

        Args:
            index (int): EMAを取得するインデックス。

        Returns:
            float: 指定されたインデックスでのEMA。
        """
        return self.get_value_by_column(COLUMN_EMA,index)

    @property
    def ema_price(self) -> float:
        """
        指定されたインデックスでのEMAを取得します。

        Returns:
            float: 指定されたインデックスでのEMA。
        """
        return self.get_value_by_column(COLUMN_EMA)


    @property
    def entry_type(self) -> str:
        """
        指定されたインデックス、または現在のエントリーインデックスのエントリータイプを取得します。

        Args:
            index (int, optional): エントリータイプを取得するインデックス。指定されていない場合は、現在のエントリーインデックスが使用されます。

        Returns:
            str: エントリータイプ。
        """
        return self.get_value_by_column(COLUMN_ENTRY_TYPE)

    @entry_type.setter
    def entry_type(self, entry_type: str):
        """
        エントリーのタイプを設定します。

        Args:
            entry_type (str): 設定するエントリータイプ。
        """
        self.set_value_by_column(COLUMN_ENTRY_TYPE, entry_type)


    @property
    def prediction(self) -> int:
        """
        指定されたインデックス、または現在のエントリーインデックスの予測値を取得します。

        Args:
            index (int, optional): 予測値を取得するインデックス。指定されていない場合は、現在のエントリーインデックスが使用されます。

        Returns:
            int: 予測値。
        """
        return self.get_value_by_column(COLUMN_PREDICTION)

    @prediction.setter
    def prediction(self, prediction: int):
        """
        予測値を設定します。

        Args:
            prediction (int): 設定する予測値。
        """
        self.set_value_by_column(COLUMN_PREDICTION, prediction)

    @property
    def current_profit(self) -> float:
        """
        現在の利益を取得します。指定されたインデックス、または現在のデータインデックスの一つ前の利益が返されます。

        Args:
            index (int, optional): 利益を取得するインデックス。指定されていない場合は、現在のデータインデックスが使用されます。

        Returns:
            float: 現在の利益。
        """
        return self.get_value_by_column(COLUMN_CURRENT_PROFIT)

    @current_profit.setter
    def current_profit(self, profit: float):
        """
        現在の利益を設定します。

        Args:
            profit (float): 設定する利益。
        """
        self.set_value_by_column(COLUMN_CURRENT_PROFIT, profit)


    @property
    def close_price(self) -> float:
        """
        指定したインデックスでの終値を取得します。

        Args:
            index (int): 終値を取得するインデックス。

        Returns:
            float: 指定されたインデックスでの終値。
        """
        return self.get_value_by_column(COLUMN_CLOSE)


    @property
    def current_date(self) -> str:
        """
        現在のインデックスでの日付を取得します。

        Returns:
            str: 現在のインデックスでの日付。
        """
        return self.get_value_by_column(COLUMN_DATE)

    @property
    def raw_data(self) -> pd.DataFrame:
        """
        トレーディングデータの生データフレームを取得します。

        Returns:
            pd.DataFrame: トレーディングデータの生データ。
        """
        return self.dataloader.get_raw()


    @property
    def entry_price(self) -> float:
        """
        設定されたエントリー価格を取得します。

        Returns:
            float: エントリー価格。
        """
        if self._entry_index != 0:
            return self._entry_price
        return self.get_value_by_column(COLUMN_ENTRY_PRICE)

    @entry_price.setter
    def entry_price(self, price: float):
        """
        エントリー価格を設定します。

        Args:
            price (float): エントリーする価格。
        """
        self._entry_price = price
        self.set_value_by_column(COLUMN_ENTRY_PRICE, price)
        price = self.get_value_by_column(COLUMN_ENTRY_PRICE)
        print("entry_price",price)

    @property
    def exit_price(self) -> float:
        """
        設定されたエグジット価格を取得します。

        Returns:
            float: エグジット価格。
        """
        return self.get_value_by_column(COLUMN_EXIT_PRICE)

    @exit_price.setter
    def exit_price(self, price: float):
        """
        エグジット価格を設定します。

        Args:
            price (float): エグジットする価格。
        """
        self.set_value_by_column(COLUMN_EXIT_PRICE, price)


    @property
    def record_state(self) -> str:
        """
        トレードの状態を記録します。

        Args:
            state (str): トレードの状態。
        """
        return self.get_value_by_column(COLUMN_STATE)

    @record_state.setter
    def record_state(self, state: str):
        """
        トレードの状態を記録します。

        Args:
            state (str): トレードの状態。
        """
        self.set_value_by_column(COLUMN_STATE, state)

    def save_data(self):
        """
        トレードデータをCSVファイルに保存します。
        """
        self.data.to_csv(FILENAME_RESULT_CSV)

    def is_first_column_less_than_second(self, index, col1, col2) -> bool:
        """
        指定されたインデックスの2つのカラム値を比較し、最初のカラムの値が2番目のカラムの値より小さいかどうかを判断します。

        Args:
            index (int): データフレームのインデックス。
            col1 (str): 最初のカラム名。
            col2 (str): 2番目のカラム名。

        Returns:
            bool: 最初のカラムの値が2番目のカラムの値より小さい場合はTrue、そうでない場合はFalse。
        """
        return self.dataloader.is_first_column_less_than_second(index, col1, col2)

    def is_first_column_greater_than_second(self, index, col1, col2) -> bool:
        """
        指定されたインデックスの2つのカラム値を比較し、最初のカラムの値が2番目のカラムの値より大きいかどうかを判断します。

        Args:
            index (int): データフレームのインデックス。
            col1 (str): 最初のカラム名。
            col2 (str): 2番目のカラム名。

        Returns:
            bool: 最初のカラムの値が2番目のカラムの値より大きい場合はTrue、そうでない場合はFalse。
        """
        return self.dataloader.is_first_column_greater_than_second(index, col1, col2)



    def log_transaction(self, message):
        """
        トランザクションログを記録します。

        Args:
            message (str): ログに記録するメッセージ。
        """
        date = self.current_date
        self.trading_logger.log_transaction(date, message)

    def event_handle(self, index: int):
        """
        指定されたインデックスに基づいてイベントを処理します。

        Args:
            index (int): イベントを処理するデータのインデックス。
        """
        self.current_index = index
        self._state.event_handle(self, index)

    def run_trading(self,context):
        """
        取引プロセスを実行します。指定されたコンテキストに基づき、データを分析し、取引イベントを生成します。

        Args:
            context (TradingContext): 実行する取引コンテキスト。
        """
        data = context.raw_data
        for index in range(len(data)):
            context.event_handle(index)


    def load_data_from_datetime_period(self, start_datetime, end_datetime, table_name=None):
        """
        指定された期間とテーブル名に基づいてデータをロードします。

        Args:
            start_datetime (datetime): データロードの開始日時。
            end_datetime (datetime): データロードの終了日時。
            table_name (str, optional): データをロードするテーブル名。デフォルトはNoneです。
        """
        table_name = self.dataloader.make_table_name_tech(table_name)
        self.dataloader.load_data_from_datetime_period(start_datetime, end_datetime, table_name)
        self.reset_index()
        self.add_data_columns()

    def load_recent_data_from_db(self, table_name=None)->pd.DataFrame:
        """
        最新のデータをデータベースからロードします。

        Args:
            table_name (str, optional): データをロードするテーブル名。デフォルトはNoneです。

        Returns:
            pd.DataFrame: ロードされたデータ。
        """
        table_name = self.dataloader.make_table_name_tech(table_name)
        self.dataloader.load_recent_data_from_db(table_name)
        self.reset_index()
        self.add_data_columns()
        return self.dataloader.get_raw()

    def get_latest_data(self, table_nam=None):
        """
        最新のデータを取得します。

        Args:
            table_name (str): データを取得するテーブル名。

        Returns:
            pd.DataFrame: 最新のデータ。
        """
        if table_nam is None:
            table_name = self.dataloader.make_table_name_tech(table_nam)
        df = self.dataloader.get_latest_data(table_name)
        self.set_current_index(len(self.dataloader.get_raw())-1)
        return df