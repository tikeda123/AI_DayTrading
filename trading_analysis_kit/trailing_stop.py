import sys
from pathlib import Path
import pandas as pd
import numpy as np


from common.trading_logger import TradingLogger
from common.config_manager import ConfigManager
from common.constants import *


class TrailingStopCalculator:
    """
    トレーリングストップ機能を計算するためのクラスです。
    このクラスは、指定されたエントリー価格、トレーリングストップのパーセンテージ、およびポジションの種類（ロングまたはショート）に基づいて、
    アクティベーション価格を管理し、現在の市場価格に応じてトレードを実行するかどうかを判断します。

    Attributes:
        trailing_rate (float): トレーリングストップのパーセンテージ。
        trailing_stop_rate (float): トレーリングストップのパーセンテージ。
        entry_price (float): トレードのエントリー価格。
        activation_price (float): トレーリングストップのアクティベーション価格。
        current_best_price (float): ポジションの最高価格。
    """
    def __init__(self):
        config = ConfigManager()
        self.trailing_rate = config.get('ACCOUNT', 'TRAILING_RATE')
        self.trailing_stop_rate = None
        self.entry_price = None
        self.activation_price = None
        self.current_best_price = None
        self.entry_price = None

    def set_entry_conditions(self, entry_price, start_trailing_price, trade_type):
        """
        トレーリングストップの初期条件を設定します。

        Args:
            entry_price (float): トレードのエントリー価格。
            start_trailing_price (float): トレーリングストップの開始価格。
            trade_type (str): トレードの種類（ENTRY_TYPE_LONGまたはENTRY_TYPE_SHORT）。
        """
    def set_entry_conditions(self, entry_price, start_trailing_price, trade_type):
        self.entry_price = entry_price
        self.trailing_stop_rate = (abs(start_trailing_price - entry_price) * self.trailing_rate) / entry_price
        self.start_trailing_price = start_trailing_price
        self.trade_tpye = trade_type

        if self.trade_tpye == ENTRY_TYPE_LONG:
            self.activation_price = self.start_trailing_price - (self.entry_price * self.trailing_stop_rate)
        else:
            self.activation_price = self.start_trailing_price + (self.entry_price * self.trailing_stop_rate)

        self.current_best_price = start_trailing_price

    def update_price(self, current_price):
        """
        資産の現在価格を更新し、新しいアクティベーション価格を計算し、トレードを実行するかどうかを確認します。
        戻り値として、トレードを実行するかどうかを示すブール値と、現在のアクティベーション価格のタプルを返します。

        Args:
            current_price (float): 資産の現在の市場価格。

        Returns:
            tuple: (トレードを実行するかどうかを示すブール値, 現在のアクティベーション価格)
        """
        trade_triggered = False

        if self.trade_tpye == ENTRY_TYPE_LONG:
                # For long positions, check if current price is higher than the best price
                if current_price > self.current_best_price:
                        self.current_best_price = current_price
                        self.activation_price = self.current_best_price - (self.entry_price * self.trailing_stop_rate)

                # Check if current price has fallen below the activation price
                if current_price <= self.activation_price:
                        trade_triggered = True
        else:
                # For short positions, check if current price is lower than the best price
                if current_price < self.current_best_price:
                        self.current_best_price = current_price
                        self.activation_price = self.current_best_price + (self.entry_price * self.trailing_stop_rate)

                # Check if current price has risen above the activation price
                if current_price >= self.activation_price:
                        trade_triggered = True

        return trade_triggered, self.activation_price


class TrailingStopAnalyzer:
    def __init__(self, config_manager: ConfigManager, trading_logger: TradingLogger):
        # コンフィギュレーションマネージャーとトレーディングロガーの初期化
        self.__config_manager = config_manager
        self.__logger = trading_logger
        # トレーリングストップの設定値をコンフィギュレーションマネージャーから取得
        self.__tailing_stop_duration = config_manager.get('ACCOUNT', 'TRAILING_STOP_DUR')
        self.__trailing_rate = config_manager.get('ACCOUNT', 'TRAILING_STOP_RATE')
        # 長期と短期のトレーリングストップ計算用のインスタンスを作成
        self.__long_trailing_stop = TrailingStopCalculator()
        self.__short_trailing_stop = TrailingStopCalculator()

    def process_trade(self, data, index, trade_type):
        # トレード処理を行うメソッド。指定されたトレードタイプ（長期または短期）に基づいて処理を行う
        trade_triggered = False
        entry_price = data.at[index, 'close']
        next_price = entry_price  # 初期値を設定

        # トレードタイプに基づいて使用するトレーリングストップ計算器を選択
        if trade_type == "long":
            trailing_stop_calculator = self.__long_trailing_stop
        else:
            trailing_stop_calculator = self.__short_trailing_stop

        # トレーリングストップの期間にわたってトレードを処理
        for i in range(1, self.__tailing_stop_duration + 1):
            if index + i < len(data):
                next_price = data.at[index + i, 'close']
                trade_triggered, exit_price = trailing_stop_calculator.update_price(next_price)
                # デバッグメッセージをログに記録
                self.__logger.log_debug_message(f'index:{index}, i:{i}, next_price:{next_price}, trade_triggered:{trade_triggered}, exit_price:{exit_price}, trade_type:{trade_type}')
                if trade_triggered:
                    break

        if not trade_triggered:
            exit_price = next_price

        return exit_price


    def apply_trailing_stop_strategy(self, data):
        """
        Apply the trailing stop strategy to each row in the DataFrame.

        :param data: The DataFrame containing the trade data.
        """
        for index in data.index:
            best_exit_price, is_long = self.apply_trailing_stop_to_row(data, index)

            # 計算された終値とポジションタイプをデータフレームに設定
            data.at[index, 'exit_price'] = best_exit_price
            data.at[index, 'islong'] = is_long


    def apply_trailing_stop_to_row(self, data, row_index):
        """
        Apply the trailing stop strategy to a specific row in the DataFrame.

        :param data: The DataFrame containing the trade data.
        :param row_index: The index of the row to which the strategy is to be applied.
        """
        #self.__logger.log_verbose_message(f'row_index:{row_index}, data.index:{data.index},size:{len(data)}')
        if row_index not in data.index:
            raise IndexError(f"Row index {row_index} is out of bounds.")

        # 長期および短期のトレーリングストップの条件を設定
        row = data.iloc[row_index]
        self.__long_trailing_stop.set_entry_conditions(row['close'], self.__trailing_rate, True)
        self.__short_trailing_stop.set_entry_conditions(row['close'], self.__trailing_rate, False)

        # 長期および短期トレードの終値を計算
        exit_price_long = self.process_trade(data, row_index, "long")
        exit_price_short = self.process_trade(data, row_index, "short")

        # 最も良い終値を決定
        long_diff = exit_price_long - row['close']
        short_diff = row['close'] - exit_price_short
        is_long = long_diff > short_diff
        best_exit_price = exit_price_long if is_long else exit_price_short

        # 計算された終値とポジションタイプを返す
        return best_exit_price, is_long




