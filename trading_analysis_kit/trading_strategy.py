from abc import ABC, abstractmethod

# 戦略インターフェース
class TradingStrategy(ABC):
    """
    トレード戦略のための抽象基底クラスです。具体的なトレード戦略を実装するためには、
    このクラスを継承し、すべての抽象メソッドをオーバーライドする必要があります。
    各メソッドはトレードの異なる段階や状態での行動を定義します。
    """
    @abstractmethod
    def Idel_event_execute(self, context):
        """
        トレードがアイドル状態にある際に実行されるメソッドです。

        Args:
            context: トレードの実行コンテキストを提供するオブジェクト。
        """
        pass

    @abstractmethod
    def EntryPreparation_event_execute(self, context):
        """
        エントリー準備状態で、カウンターが設定閾値を超えた際に実行されるメソッドです。

        Args:
            context: トレードの実行コンテキストを提供するオブジェクト。
        """
        pass

    @abstractmethod
    def PositionState_event_exit_execute(self, context):
        """
        ポジションを保持している状態で、エグジット条件が満たされた際に実行されるメソッドです。

        Args:
            context: トレードの実行コンテキストを提供するオブジェクト。
        """
        pass

    @abstractmethod
    def PositionState_event_continue_execute(self, context):
        """
        ポジションを保持している状態で、ポジションを継続する条件が満たされた際に実行されるメソッドです。

        Args:
            context: トレードの実行コンテキストを提供するオブジェクト。
        """
        pass




