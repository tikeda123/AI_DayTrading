import pandas as pd
import os,sys

from common.constants import *

from trading_analysis_kit.trading_state import *
from trading_analysis_kit.trading_strategy import TradingStrategy


def short_or_long(direction, pred):
    """
    ショートかロングかを判断する関数
    :param context: TradingContext
    :param pred: int
    :return: str
        ショートかロングか
    """
    if direction == BB_DIRECTION_UPPER and pred == 1:
        return ENTRY_TYPE_LONG
    elif direction == BB_DIRECTION_UPPER and pred == 0:
        return ENTRY_TYPE_SHORT
    elif direction == BB_DIRECTION_LOWER and pred == 1:
        return ENTRY_TYPE_SHORT
    else:
        return ENTRY_TYPE_LONG

class SimulationStrategy(TradingStrategy):
    """
    トレーディング戦略を実装するシミュレーションクラスです。
    トレードのエントリーとエグジットの判断、状態遷移の管理を行います。
    """


    def Idel_event_execute(self, context):
        """
        アイドル状態のイベントを実行します。エントリーカウンターを増やし、
        エントリー準備状態に遷移させます。

        Args:
            context (TradingContext): トレーディングコンテキストオブジェクト。
        """
        context.inc_entry_counter()
        context.change_to_other_state(STATE_ENTRY_PREPARATION)

    def EntryPreparation_event_over_counter_execute(self, context):
        """
        エントリー準備状態でカウンターが閾値を超えた場合のイベントを実行します。
        エントリー判断を行い、エントリーする場合はポジション状態に遷移し、トレードのエントリーと
        エグジットラインの決定を行います。エントリーしない場合はアイドル状態に戻ります。

        Args:
            context (TradingContext): トレーディングコンテキストオブジェクト。
        """

        direction = context.get_bb_direction()
        #is_entry,pred = self.entry_manager.entry_decision_maker(context)
        #is_entry, pred = self.entry_manager.entry_decision_maker_rolling(context)
        is_entry, pred = context.entry_manager.should_entry(context)

        if is_entry == False:
            context.record_state_and_transition(STATE_IDLE)
            context.reset_index()
            return

        context.change_to_position_state(direction)
        self.trade_entry(context, pred)
        self.decision_exit_line(context, pred)

    def EntryPreparation_event_under_counter_execute(self, context):
        """
        エントリー準備状態でカウンターが閾値未満の場合のイベントを実行します。
        エントリーカウンターを増やし、状態を記録します。

        Args:
            context (TradingContext): トレーディングコンテキストオブジェクト。
        """
        context.inc_entry_counter()
        context.record_state(STATE_ENTRY_PREPARATION)


    def PositionState_event_exit_execute(self, context):
        """
        ポジション状態でのエグジットイベントを実行します。ロスカットがトリガーされた場合は、
        ロスカット価格でポジションを終了し、そうでない場合は現在の価格でポジションを終了します。
        その後、状態をアイドル状態に変更します。

        Args:
            context (TradingContext): トレーディングコンテキストオブジェクト。
        """
        # 損切りがトリガーされたかどうかを判断します。
        is_losscut_triggered, exit_price = self.is_losscut_triggered(context)

        losscut_event =None
        if not is_losscut_triggered:
            exit_price = context.get_current_price()

        else:
            losscut_event = "losscut"
            context.log_transaction(f'losscut price: {exit_price}')

        # 取引を終了します。
        self.trade_exit(context, exit_price, losscut=losscut_event)
        # コンテキストの状態をアイドルに変更します。
        context.change_to_idle_state()



    def PositionState_event_continue_execute(self, context):
        """
        ポジション状態での継続イベントを実行します。ロスカットの判断を行い、必要に応じて
        ポジションを終了しアイドル状態に遷移します。

        Args:
            context (TradingContext): トレーディングコンテキストオブジェクト。
        """
        is_losscut, losscut_price = self.is_losscut_triggered(context)
        if is_losscut:
            context.log_transaction(f'losscut price: {losscut_price}')
            self.trade_exit(context, losscut_price, losscut="losscut")
            context.change_to_idle_state()
            return

        pandl = self.calculate_current_pandl(context)
        current_price = context.get_current_price()
        context.log_transaction(f'current price: {current_price}, pandl: {pandl}')

    def decide_on_position_exit(self, context, index):
        """
        出口ラインを取得するメソッド
        :param context: TradingContext
        :return: float
            出口ライン
        """
        return context.exit_manager.decide_on_position_exit(context,index)


    def decision_exit_line(self, context, pred):
        """
        トレードのエグジットラインを決定します。

        Args:
            context (TradingContext): トレーディングコンテキストオブジェクト。
            pred (int): 予測結果（1または0）。

        ボリンジャーバンドの方向と予測結果に基づいて、エグジットラインを設定します。
        """
        if context.get_bb_direction() == BB_DIRECTION_UPPER and pred == 1:
            context.set_exit_line(COLUMN_MIDDLE_BAND)

        elif context.get_bb_direction() == BB_DIRECTION_UPPER and pred == 0:
            context.set_exit_line(COLUMN_LOWER_BAND1)

        elif context.get_bb_direction() == BB_DIRECTION_LOWER and pred == 1:
            context.set_exit_line(COLUMN_MIDDLE_BAND)
        else:
            context.set_exit_line(COLUMN_UPPER_BAND1)

    def trade_entry(self, context, pred):
        """
        トレードのエントリーを実行します。

        Args:
            context (TradingContext): トレーディングコンテキストオブジェクト。
            pred (int): 予測結果（1または0）。

        ボリンジャーバンドの方向と予測結果に基づいてエントリータイプを決定し、
        トレードエントリーを実行します。
        """
        # 現在のBollinger Bandsの方向を取得
        direction = context.get_bb_direction()
        entry_type = short_or_long(direction, pred)

        #context.set_prediction(pred)
        # エントリー価格と現在の日付を取得
        context.set_entry_type(entry_type)
        entry_price = context.get_entry_price()
        date = context.get_current_date()
        # トレードエントリーを実行し、トランザクションシリアル番号を取得
        serial = context.fx_transaction.trade_entry(entry_type, pred, entry_price, date, direction)
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



