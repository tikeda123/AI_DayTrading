import pandas as pd
import os,sys

from common.constants import *

class State:
    """
    抽象状態クラスです。すべての具体的な状態クラスはこのクラスを継承する必要があります。

    このクラスは、状態に応じたイベントの処理方法を定義するための基盤を提供します。
    """
    def handle_request(self, context, event):
        """
        状態に応じてイベントを処理するメソッドです。このメソッドは各サブクラスで実装されるべきです。

        Args:
            context: 現在のコンテキストオブジェクト。
            event: 処理するイベント。

        Raises:
            NotImplementedError: このメソッドは抽象メソッドであるため、サブクラスでの実装が必須です。
        """
        raise NotImplementedError("Each state must implement 'handle_request' method.")

    def invalid_event(self, event):
        """
        無効なイベントが発生した際に呼び出されるメソッドです。

        Args:
            event: 無効なイベント。
        """
        print(f"エラー: 現在の状態では '{event}' イベントは無効です。")

class IdleState(State):
    """
    アイドル状態:
    この状態は、トレードの機会を探している状態を表します。ここでは、特定の条件が満たされるまで待機し、
    条件が満たされたら次の状態に遷移する処理を行います。
    """

    def handle_request(self, context, event):
        """
        イベントに応じて状態遷移を行うメソッドです。

        Args:
            context: 現在のコンテキストオブジェクト。
            event: 処理するイベント。

        特定のイベント(EVENT_ENTER_PREPARATION)が発生した場合、エントリー準備状態に遷移します。
        それ以外のイベントは無効として扱います。
        """
        if event == EVENT_POSITION:
            context.log_transaction( "エントリーイベントが発生。ポジション状態に遷移します。")
            context.set_state(PositionState())
        elif event == EVENT_ENTER_PREPARATION:
            context.log_transaction("エントリー準備状態に遷移します。")
            context.set_state(EntryPreparationState())
        elif event == EVENT_IDLE:
            context.log_transaction(LOG_TRANSITION_TO_IDLE_FROM_POSITION)
            context.set_state(IdleState())
        else:
            self.invalid_event(event)

    def event_handle(self, context, index: int):
        """
        アイドル状態において特定の条件を満たすイベントが発生した場合に処理を行います。

        Args:
            context: 現在のコンテキストオブジェクト。
            index: イベントが発生したデータのインデックス。

        対応するトレードの実行処理を行います。
        """
        if context.is_first_column_greater_than_second(context.get_current_index(), COLUMN_CLOSE, COLUMN_UPPER_BAND2):
            context.strategy.Idel_event_execute(context)
        #if index < TIME_SERIES_PERIOD:
        #    return None
        context.strategy.Idel_event_execute(context)

class EntryPreparationState(State):
    """
    エントリー準備状態:
    この状態は、トレードのエントリー準備が行われている段階を表します。
    トレードエントリーの条件が整うまで待機し、条件が満たされたらポジション状態に遷移する処理を行います。
    """

    def handle_request(self, context, event):
        """
        イベントに応じて状態遷移を行うメソッドです。

        Args:
            context: 現在のコンテキストオブジェクト。
            event: 処理するイベント。

        特定のイベント(EVENT_POSITION)が発生した場合、ポジション状態に遷移します。
        EVENT_IDLEイベントが発生した場合、アイドル状態に戻ります。
        それ以外のイベントは無効として扱います。
        """
        if event == EVENT_POSITION:
            context.log_transaction( "エントリーイベントが発生。ポジション状態に遷移します。")
            context.set_state(PositionState())
        elif event == EVENT_IDLE:
            context.log_transaction("アイドル状態に遷移します。")
            context.set_state(IdleState())
        else:
            self.invalid_event(event)

    def event_handle(self, context, index: int):
        """
        状態に応じたイベント処理メソッド:
        """
        context.strategy.EntryPreparation_event_execute(context)




class PositionState(State):
    """
    ポジション状態を表すクラスです。この状態では、トレードが実行され、ポジションを保有しています。
    """
    def handle_request(self, context, event):
        """
        イベントに応じて状態遷移を行うメソッドです。

        Args:
            context: 現在のコンテキストオブジェクト。
            event: 処理するイベント。

        EVENT_IDLEイベントが発生した場合、アイドル状態に遷移します。
        それ以外のイベントは無効として扱います。
        """
        if event == EVENT_IDLE:
            context.log_transaction(LOG_TRANSITION_TO_IDLE_FROM_POSITION)
            context.set_state(IdleState())
        else:
            self.invalid_event(event)

    def event_handle(self, context, index: int):
        """
        状態に応じたイベント処理メソッド:
        """
        context.strategy.PositionState_event_exit_execute(context)

