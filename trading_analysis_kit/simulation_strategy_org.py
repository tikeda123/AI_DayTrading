import pandas as pd
import os,sys

from common.constants import *
from common.utils import get_config

from trading_analysis_kit.trading_state import *
from trading_analysis_kit.trading_strategy import TradingStrategy

class SimulationStrategy(TradingStrategy):
    """
    トレーディング戦略を実装するシミュレーションクラスです。
    トレードのエントリーとエグジットの判断、状態遷移の管理を行います。
    """
    def __init__(self):
        self.__config = get_config("ACCOUNT")
        self.__entry_rate= self.__config["ENTRY_RATE"]

    def Idel_event_execute(self, context):
        """
        Executes an event in the idle state by increasing the entry counter
        and transitioning to the entry preparation state.

        Args:
            context (TradingContext): The trading context object.
        """
        if context.get_current_index() < 10:
            return

        current_price = context.get_open_price()
        ema_price = context.get_ema_price(context.get_current_index()-1)
        trend_prediction = context.prediction_trend()

        # Determine the adjusted entry price based on trend prediction
        entry_price = self.calculate_adjusted_entry_price(ema_price, current_price, trend_prediction)

        # Check if the entry conditions are met
        if self.entry_conditions_met(context.get_high_price(), entry_price, context.get_low_price()):
            self.trade_entry(context, trend_prediction, entry_price)
            context.change_to_entrypreparation_state()
        else:
            context.log_transaction("Entry Preparation: No Entry")
            context.change_to_idle_state()

    def calculate_adjusted_entry_price(self, ema_price, current_price, trend_prediction):
        """
        Calculates the adjusted entry price based on the EMA and the current price.

        Args:
            ema_price (float): EMA price from the previous day.
            current_price (float): Today's current price.
            trend_prediction (int): Prediction of the market trend (1 for upward, other for downward).

        Returns:
            float: The adjusted entry price.
        """
        adjustment_factor = 1 - self.__entry_rate if trend_prediction == 1 else 1 + self.__entry_rate
        ideal_price = ema_price * adjustment_factor

        return current_price if (trend_prediction == 1 and ideal_price > current_price) or \
                                (trend_prediction != 1 and ideal_price < current_price) else ideal_price

    def entry_conditions_met(self, high_price, entry_price, low_price):
        """
        Checks if the entry conditions are met based on price thresholds.

        Args:
            high_price (float): Today's high price.
            entry_price (float): Calculated entry price.
            low_price (float): Today's low price.

        Returns:
            bool: True if conditions are met, False otherwise.
        """
        return high_price > entry_price > low_price


    def EntryPreparation_execute(self, context):
        """
        エントリー準備状態でカウンターが閾値を超えた場合のイベントを実行します。
        エントリー判断を行い、エントリーする場合はポジション状態に遷移し、トレードのエントリーと
        エグジットラインの決定を行います。エントリーしない場合はアイドル状態に戻ります。

        Args:
            context (TradingContext): トレーディングコンテキストオブジェクト。
        """


        # 損切りがトリガーされたかどうかを判断します。
        is_losscut_triggered, exit_price = self.is_losscut_triggered(context)

        if is_losscut_triggered:
            losscut_event = "losscut"
            context.log_transaction(f'losscut price: {exit_price}')
            self.trade_exit(context, exit_price, losscut=losscut_event)
            context.change_to_idle_state()
            return

        # コンテキストの状態をアイドルに変更します。
        context.change_to_position_state()
        return

    def PositionState_event_exit_execute(self, context):
        """
        ポジション状態でのエグジットイベントを実行します。ロスカットがトリガーされた場合は、
        ロスカット価格でポジションを終了し、そうでない場合は現在の価格でポジションを終了します。
        その後、状態をアイドル状態に変更します。

        Args:
            context (TradingContext): トレーディングコンテキストオブジェクト。
        """
        pandl = self.calculate_current_pandl(context)

       # 損切りがトリガーされたかどうかを判断します。
        is_losscut_triggered, exit_price = self.is_losscut_triggered(context)

        if is_losscut_triggered:
            losscut_event = "losscut"
            context.log_transaction(f'losscut price: {exit_price}')
            self.trade_exit(context, exit_price, losscut=losscut_event)
            context.change_to_idle_state()
            return

        if not self.should_hold_position(context):
            exit_price = context.get_current_price()
            self.trade_exit(context, exit_price)
            context.change_to_idle_state()
            return

        context.set_pandl(pandl)
        context.log_transaction(f'continue Position state pandl:{pandl}')
    def PositionState_event_continue_execute(self, context):
        """
        ポジション状態での継続イベントを実行します。ロスカットの判断を行い、必要に応じて
        ポジションを終了しアイドル状態に遷移します。

        Args:
            context (TradingContext): トレーディングコンテキストオブジェクト。
        """
        pass
        #ここで、継続するかどうか判断するロジックを実装する。もう一度、機械学習モデルを使って、予測を行い、
        #予測が正しければ、継続する。予測が間違っていれば、エグジットする。エントリーした価格が、一つ前の日のEMAよりも
        #高い場合は、ロングポジションを継続する。逆に、エントリーした価格が、一つ前の日のEMAよりも低い場合は、ショートポジションを継続する。

    def should_hold_position(self, context):
        """
        ポジションを保持すべきかどうかを判断します。

        Args:
            context (TradingContext): トレーディングコンテキストオブジェクト。

        Returns:
            bool: ポジションを保持すべきかどうかの真偽値。

        ポジションを保持すべきかどうかを判断します。
        """
        trend_prediction = context.prediction_trend()
        entry_type = context.get_entry_type()
        entry_price = context.get_entry_price()
        current_ema_price = context.get_ema_price()

        if trend_prediction == 1 and entry_type == ENTRY_TYPE_LONG and entry_price < current_ema_price:
            return True
        elif trend_prediction == 0 and entry_type == ENTRY_TYPE_SHORT and entry_price > current_ema_price:
            return True

        return False

    def trade_entry(self, context, pred,entry_price):
        """
        トレードのエントリーを実行します。

        Args:
            context (TradingContext): トレーディングコンテキストオブジェクト。
            pred (int): 予測結果（1または0）。

        ボリンジャーバンドの方向と予測結果に基づいてエントリータイプを決定し、
        トレードエントリーを実行します。
        """
        # 現在のBollinger Bandsの方向を取得
        if pred == 1:
            entry_type = ENTRY_TYPE_LONG
        else:
            entry_type = ENTRY_TYPE_SHORT

        date = context.get_current_date()

        context.set_entry_index(context.get_current_index())
        context.set_entry_price(entry_price)
        context.set_entry_type(entry_type)

        # トレードエントリーを実行し、トランザクションシリアル番号を取得
        serial = context.fx_transaction.trade_entry(entry_type, pred, entry_price, date, "upper")
        # 取得したトランザクションシリアル番号をコンテキストに設定
        context.set_fx_serial(serial)

    def trade_exit(self, context, exit_price,losscut=None):
        """
        トレードのエグジットを実行します。

        Args:
            context (TradingContext): トレーディングコンテキストオブジェクト。
            exit_price (float): エグジットする価格。

        Returns:
            float: 実行されたトレードの損益。

        指定された価格でトレードのエグジットを実行し、損益を計算します。
        """
        serial = context.get_fx_serial()
        date = context.get_current_date()
        context.set_exit_index(context.get_current_index())
        context.set_exit_price(exit_price)
        return context.fx_transaction.trade_exit(serial,exit_price, date, losscut=losscut)

    def is_losscut_triggered(self, context):
        """
        損切りがトリガーされたかどうかを判断します。

        Args:
            context (TradingContext): トレーディングコンテキストオブジェクト。

        Returns:
            bool, float: 損切りがトリガーされたかの真偽値と、損切りがトリガーされた場合の価格。

        現在の価格をもとに損切りがトリガーされたかどうかを判断し、
        トリガーされた場合はその価格を返します。
        """
        serial = context.get_fx_serial()
        entry_type = context.get_entry_type()
        losscut_price = None

        if entry_type == ENTRY_TYPE_LONG:
            losscut_price = context.get_low_price()
        else:
            losscut_price = context.get_high_price()

        return context.fx_transaction.is_losscut_triggered(serial, losscut_price)

    def calculate_current_pandl(self, context):
        """
        現在の損益を計算します。

        Args:
            context (TradingContext): トレーディングコンテキストオブジェクト。

        Returns:
            float: 現在の損益。

        現在の価格とエントリー価格をもとに損益を計算します。
        """
        serial = context.get_fx_serial()
        current_price = context.get_current_price()
        pandl = context.fx_transaction.get_pandl(serial, current_price)
        return pandl

    def show_win_lose(self,context):
        """
        勝敗の統計情報を表示します。

        Args:
            context (TradingContext): トレーディングコンテキストオブジェクト。

        トレードの勝敗に関する統計情報を表示します。
        """
        context.fx_transaction.display_all_win_rates()
        context.fx_transaction.plot_balance_over_time()



