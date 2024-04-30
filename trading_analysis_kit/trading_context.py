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
        #self.add_data_columns()

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


    def reset_index(self):
        """
        現在のインデックス、エントリーインデックス、エグジットインデックス、およびFXシリアル番号をリセットします。
        """
        self.entry_index = 0
        self.exit_index = 0
        self.fx_serial = 0

    def add_data_columns(self):
        """
        データフレームに取引分析用の新しいカラムを追加します。
        """
        self.dataloader.df_new_column(COLUMN_STATE, None,str)
        self.dataloader.df_new_column(COLUMN_ENTRY_PRICE, 0.0,float)
        self.dataloader.df_new_column(COLUMN_EXIT_PRICE,0.0,float)
        self.dataloader.df_new_column(COLUMN_PANDL, 0.0,float)
        self.dataloader.df_new_column(COLUMN_PREDICTION, 0,int)
        self.dataloader.df_new_column(COLUMN_ENTRY_TYPE,  None,str)

    def set_pandl(self, pandl: float):
        """
        損益を設定します。

        Args:
            pandl (float): 設定する損益値。
        """
        self.dataloader.set_df(self.current_index, COLUMN_PANDL, pandl)

    def get_current_index(self) -> int:
        """
        現在のインデックスを取得します。

        Returns:
            int: 現在のインデックス番号。
        """
        return self.current_index

    def set_current_index(self, index: int):
        """
        現在のデータインデックスを設定します。

        Args:
            index (int): 設定するインデックスの値。
        """
        self.current_index = index

    def get_entry_counter(self) -> int:
        """
        エントリーのカウンターを取得します。

        Returns:
            int: 現在のエントリーのカウンター値。
        """
        return self.entry_counter

    def set_entry_counter(self, counter: int):
        """
        エントリーのカウンターを設定します。

        Args:
            counter (int): 設定するカウンターの値。
        """
        self.entry_counter = counter

    def event_handle(self, index: int):
        """
        指定されたインデックスに基づいてイベントを処理します。

        Args:
            index (int): イベントを処理するデータのインデックス。
        """
        self.current_index = index
        self._state.event_handle(self, index)

    def set_state(self, new_state):
        """
        トレーディングの状態を設定します。

        Args:
            new_state (State): 新しい状態オブジェクト。
        """
        self._state = new_state

    def get_high_price(self, index=None) -> float:
        """
        指定されたインデックスでの高値を取得します。

        Args:
            index (int): 高値を取得するインデックス。

        Returns:
            float: 指定されたインデックスでの高値。
        """
        if index is None:
            index = self.get_current_index()

        return self.dataloader.get_df(index, COLUMN_HIGH)

    def get_low_price(self, index=None) -> float:
        """
        指定されたインデックスでの安値を取得します。

        Args:
            index (int): 安値を取得するインデックス。

        Returns:
            float: 指定されたインデックスでの安値。
        """
        if index is None:
            index = self.get_current_index()
        return self.dataloader.get_df(index, COLUMN_LOW)

    def get_lower2_price(self, index=None) -> float:
        """
        指定されたインデックスでの2σの価格を取得します。

        Args:
            index (int): 2σの価格を取得するインデックス。

        Returns:
            float: 指定されたインデックスでの2σの価格。
        """
        if index is None:
            index = self.get_current_index()
        return self.dataloader.get_df(index, COLUMN_LOWER_BAND2)

    def get_upper2_price(self, index=None) -> float:
        """
        指定されたインデックスでの2σの価格を取得します。

        Args:
            index (int): 2σの価格を取得するインデックス。

        Returns:
            float: 指定されたインデックスでの2σの価格。
        """
        if index is None:
            index = self.get_current_index()
        return self.dataloader.get_df(index, COLUMN_UPPER_BAND2)

    def get_open_price(self, index=None) -> float:
        """
        指定されたインデックスでの始値を取得します。

        Args:
            index (int): 始値を取得するインデックス。

        Returns:
            float: 指定されたインデックスでの始値。
        """
        if index is None:
            index = self.get_current_index()
        return self.dataloader.get_df(index, COLUMN_OPEN)

    def get_ema_price(self, index=None) -> float:
        """
        指定されたインデックスでのEMAを取得します。

        Args:
            index (int): EMAを取得するインデックス。

        Returns:
            float: 指定されたインデックスでのEMA。
        """
        if index is None:
            index = self.get_current_index()
        return self.dataloader.get_df(index, COLUMN_EMA)

    def set_entry_type(self, entry_type: str):
        """
        エントリーのタイプを設定します。

        Args:
            entry_type (str): 設定するエントリータイプ。
        """
        self.dataloader.set_df(self.current_index, COLUMN_ENTRY_TYPE, entry_type)

    def get_entry_type(self,index=None) -> str:
        """
        指定されたインデックス、または現在のエントリーインデックスのエントリータイプを取得します。

        Args:
            index (int, optional): エントリータイプを取得するインデックス。指定されていない場合は、現在のエントリーインデックスが使用されます。

        Returns:
            str: エントリータイプ。
        """
        if index is None:
            index = self.get_entry_index()
        return self.dataloader.get_df(index, COLUMN_ENTRY_TYPE)

    def set_prediction(self, prediction: int):
        """
        予測値を設定します。

        Args:
            prediction (int): 設定する予測値。
        """
        self.dataloader.set_df(self.current_index, COLUMN_PREDICTION, prediction)

    def get_prediction(self, index=None) -> int:
        """
        指定されたインデックス、または現在のエントリーインデックスの予測値を取得します。

        Args:
            index (int, optional): 予測値を取得するインデックス。指定されていない場合は、現在のエントリーインデックスが使用されます。

        Returns:
            int: 予測値。
        """
        if index is None:
            index = self.get_entry_index()
        return self.dataloader.get_df(index, COLUMN_PREDICTION)

    def get_current_profit(self,index=None) -> float:
        """
        現在の利益を取得します。指定されたインデックス、または現在のデータインデックスの一つ前の利益が返されます。

        Args:
            index (int, optional): 利益を取得するインデックス。指定されていない場合は、現在のデータインデックスが使用されます。

        Returns:
            float: 現在の利益。
        """
        if index is None:
            index = self.get_current_index()
        return self.dataloader.get_df(index-1, COLUMN_CURRENT_PROFIT)

    def get_sell_price(self) -> float:
        """
        現在のインデックスでの売却価格を取得します。

        Returns:
            float: 現在のインデックスでの売却価格。
        """
        return self.dataloader.get_close(self.current_index)

    def log_transaction(self, message):
        """
        トランザクションログを記録します。

        Args:
            message (str): ログに記録するメッセージ。
        """
        date = self.get_current_date()
        self.trading_logger.log_transaction(date, message)

    def get_close(self, index) -> float:
        """
        指定したインデックスでの終値を取得します。

        Args:
            index (int): 終値を取得するインデックス。

        Returns:
            float: 指定されたインデックスでの終値。
        """
        return self.dataloader.get_df(index, COLUMN_CLOSE)

    def get_current_price(self) -> float:
        """
        現在のインデックスでの終値を取得します。

        Returns:
            float: 現在のインデックスでの終値。
        """
        return self.dataloader.get_df(self.current_index, COLUMN_CLOSE)

    def get_current_date(self) -> str:
        """
        現在のインデックスでの日付を取得します。

        Returns:
            str: 現在のインデックスでの日付。
        """
        return self.dataloader.get_df(self.current_index, COLUMN_DATE)

    def set_entry_index(self, index: int):
        """
        エントリーしたトレードのインデックスを取得します。

        Returns:
            int: トレードのエントリーインデックス。
        """
        self.entry_index = index

    def set_exit_index(self, index: int):
        """
        エグジットしたトレードのインデックスを取得します。

        Returns:
            int: トレードのエグジットインデックス。
        """
        self.exit_index = index

    def get_entry_index(self) -> int:
        """
        エントリーしたトレードのインデックスを取得します。

        Returns:
            int: トレードのエントリーインデックス。
        """
        return self.entry_index

    def get_data(self) -> pd.DataFrame:
        """
        トレーディングデータの生データフレームを取得します。

        Returns:
            pd.DataFrame: トレーディングデータの生データ。
        """
        return self.dataloader.get_raw()

    def set_entry_price(self, price: float):
        """
        エントリー価格を設定します。

        Args:
            price (float): エントリーする価格。
        """
        self.entry_index = self.current_index
        self.dataloader.set_df(self.current_index, COLUMN_ENTRY_PRICE, price)

    def get_entry_price(self,index=None) -> float:
        """
        設定されたエントリー価格を取得します。

        Returns:
            float: エントリー価格。
        """
        if index is None:
            index = self.entry_index
        return self.dataloader.get_df(index, COLUMN_ENTRY_PRICE)

    def set_exit_price(self, price: float):
        """
        エグジット価格を設定します。

        Args:
            price (float): エグジットする価格。
        """
        self.exit_index = self.current_index
        self.dataloader.set_df(self.exit_index, COLUMN_EXIT_PRICE, price)

    def get_exit_price(self,index=None) -> float:
        """
        設定されたエグジット価格を取得します。

        Returns:
            float: エグジット価格。
        """
        if index is None:
            index = self.exit_index
        return self.dataloader.get_df(index, COLUMN_EXIT_PRICE)

    def record_state(self, state: str):
        """
        トレードの状態を記録します。

        Args:
            state (str): トレードの状態。
        """
        self.dataloader.set_df(self.current_index, COLUMN_STATE, state)

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

    def get_fx_serial(self) -> int:
        """
        FX取引のシリアル番号を取得します。

        Returns:
            int: FX取引のシリアル番号。
        """
        return self.fx_serial

    def set_fx_serial(self, serial: int):
        """
        FX取引のシリアル番号を設定します。

        Args:
            serial (int): 設定するシリアル番号。
        """
        self.fx_serial = serial

    def run_trading(self,context):
        """
        取引プロセスを実行します。指定されたコンテキストに基づき、データを分析し、取引イベントを生成します。

        Args:
            context (TradingContext): 実行する取引コンテキスト。
        """
        data = context.get_data()
        for index in range(len(data)):
            context.event_handle(index)

