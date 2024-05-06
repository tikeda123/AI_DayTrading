import pandas as pd
import os,sys


from common.trading_logger_db import TradingLoggerDB
from common.config_manager import ConfigManager
from common.constants import *

from trading_analysis_kit.trading_state import IdleState
from trading_analysis_kit.trading_data import TradingData,TradingStateData


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
        self.dataloader = TradingData()
        self.state_data = TradingStateData()
        self.strategy = strategy_context

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
        date = self.dataloader.get_current_date()
        self.trading_logger.log_transaction(date, message)

    def event_handle(self, index: int):
        """
        指定されたインデックスに基づいてイベントを処理します。

        Args:
            index (int): イベントを処理するデータのインデックス。
        """
        self.dataloader.set_current_index(index)
        self._state.event_handle(self, index)

    def run_trading(self,context):
        """
        取引プロセスを実行します。指定されたコンテキストに基づき、データを分析し、取引イベントを生成します。

        Args:
            context (TradingContext): 実行する取引コンテキスト。
        """
        data = context.dataloader.get_raw_data()
        for index in range(len(data)):
            context.event_handle(index)


    def load_data_from_datetime_period(self, start_datetime, end_datetime):
        """
        指定された期間とテーブル名に基づいてデータをロードします。

        Args:
            start_datetime (datetime): データロードの開始日時。
            end_datetime (datetime): データロードの終了日時。
            table_name (str, optional): データをロードするテーブル名。デフォルトはNoneです。
        """
        self.dataloader.load_data_from_datetime_period(start_datetime, end_datetime,MARKET_DATA_TECH)
        self.dataloader.ts.reset_index()
        self.dataloader.add_data_columns()
        self.dataloader.reset_index()


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