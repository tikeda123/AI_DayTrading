import pandas as pd
import os,sys

sys.path.append('/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0')

from trading_analysis_kit.trading_state import *
from trading_analysis_kit.trading_strategy import TradingStrategy

class MLDataCreationStrategy(TradingStrategy):
    """
    機械学習データを作成するための戦略を実装するクラス。
    トレーディング戦略の具体的な実装を提供し、特定の条件下でのトレード処理を自動化する。
    """

    def Idel_event_execute(self, context):
        """
        アイドル状態のイベントを実行します。

        Args:
            context: 現在のトレーディングコンテキスト。
        """
        context.inc_entry_counter()
        context.change_to_other_state(STATE_ENTRY_PREPARATION)

    def EntryPreparation_event_execute(self, context):
        """
        エントリー準備状態でカウンターがオーバーした場合のイベントを実行します。

        Args:
            context: 現在のトレーディングコンテキスト。
        """
        direction = context.get_bb_direction()
        context.change_to_position_state(direction)

    def PositionState_event_exit_execute(self, context):
        """
        ポジション状態での退出イベントを実行します。

        Args:
            context: 現在のトレーディングコンテキスト。
        """
        context.change_to_idle_state()

    def PositionState_event_continue_execute(self, context):
        """
        ポジション状態で継続する場合のイベントを実行します。

        Args:
            context: 現在のトレーディングコンテキスト。
        """
        context.calculate_current_profit()
        context.record_state(STATE_POSITION)

    def decide_on_position_exit(self, context, index):
        """
        ポジションの退出を決定します。

        Args:
            context: 現在のトレーディングコンテキスト。
            index: 現在のインデックス。

        Returns:
            str: 実行するイベントの名前。
        """
        bb_direction = context.get_bb_direction()
        if bb_direction == BB_DIRECTION_UPPER and \
                context.is_first_column_less_than_second(index, COLUMN_CLOSE, COLUMN_MIDDLE_BAND):
            return 'PositionState_event_exit_execute'
        elif bb_direction == BB_DIRECTION_LOWER and \
                context.is_first_column_greater_than_second(index, COLUMN_CLOSE, COLUMN_MIDDLE_BAND):
            return 'PositionState_event_exit_execute'
        else:
            return 'PositionState_event_continue_execute'