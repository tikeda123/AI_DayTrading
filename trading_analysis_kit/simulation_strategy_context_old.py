# 必要なライブラリをインポート
import pandas as pd
import os, sys


# b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得

# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)

# 共通の設定や定数を管理するモジュールをインポート
from common.constants import *

# 技術分析やシミュレーション戦略、取引コンテキストのクラスをインポート
from fxtransaction import FXTransaction
from trading_analysis_kit.simulation_strategy import SimulationStrategy
from trading_analysis_kit.trading_context import TradingContext
from trading_analysis_kit.simulation_entry_strategy import BollingerBand_EntryStrategy
from trading_analysis_kit.simulation_exit_strategy import BollingerBand_ExitStrategy

# TradingContextを継承したシミュレーション戦略コンテキストクラス
class SimulationStrategyContext(TradingContext):
    """
    シミュレーション戦略のコンテキストを管理するクラスです。

    このクラスは、特定のシミュレーション戦略に基づいて取引を実行するための環境を提供します。具体的には、
    エントリーとエグジット戦略の初期化、インデックスのリセット、ボリンジャーバンドに基づいた取引条件の管理などを行います。

    Attributes:
        fx_transaction (FXTransaction): FX取引を管理するクラスのインスタンス。
        entry_manager (BollingerBand_EntryStrategy): エントリー戦略を管理するクラスのインスタンス。
        exit_manager (BollingerBand_ExitStrategy): エグジット戦略を管理するクラスのインスタンス。
        bb_direction (str): ボリンジャーバンドの方向。
        exit_line (str): エグジットライン。
        entry_counter (int): エントリーの回数をカウントする変数。

    Args:
        strategy (SimulationStrategy): シミュレーション戦略を提供するクラスのインスタンス。
    """
    def __init__(self,strategy: SimulationStrategy):
        """
        クラスのコンストラクタです。親クラスのコンストラクタを呼び出し、インデックスをリセットし、
        FX取引クラスのインスタンスを初期化します。

        Args:
            strategy (SimulationStrategy): シミュレーション戦略を提供するクラスのインスタンス。
        """

        super().__init__(strategy)  # 親クラスのコンストラクタを呼び出す
        self.reset_index()  # インデックスをリセットするメソッドを呼び出し
        self.fx_transaction = FXTransaction()

    def init_manager(self):
        """
        エントリーとエグジット戦略の管理クラスを初期化します。
        """
        self.entry_manager = BollingerBand_EntryStrategy()
        self.exit_manager = BollingerBand_ExitStrategy()

    def reset_index(self):
        """
        インデックス、ボリンジャーバンドの方向、出口価格、エントリーカウンターをリセットします。
        """
        super().reset_index()  # 親クラスのreset_indexメソッドを呼び出し
        # ボリンジャーバンドの方向、出口価格、エントリーカウンターを初期化
        self.bb_direction = None
        self.exit_line = None
        self.entry_counter = 0

    def get_exit_line(self) -> float:
        """
        現在設定されているエグジットラインを取得します。

        Returns:
            float: エグジットライン。
        """
        return self.exit_line

    def set_exit_line(self, line: str):
        """
        エグジットラインを設定します。

        Args:
            line (float): エグジットする価格ライン。
        """
        self.exit_line = line

    def inc_entry_counter(self):
        """
        エントリーカウンターをインクリメントします。
        """
        self.entry_counter += 1  # エントリーカウンターをインクリメント

    def set_bb_direction(self, direction: str):
        """
        ボリンジャーバンドの方向を設定します。

        Args:
            direction (str): 設定するボリンジャーバンドの方向。
        """
        self.dataloader.set_df(self.current_index, COLUMN_BB_DIRECTION, direction)
        self.bb_direction = direction

    def get_bb_direction_df(self, index) -> str:
        """
        指定されたインデックスでのボリンジャーバンドの方向を取得します。

        Args:
            index (int): ボリンジャーバンドの方向を取得するインデックス。

        Returns:
            str: 指定されたインデックスでのボリンジャーバンドの方向。
        """
        return self.dataloader.get_df(index, COLUMN_BB_DIRECTION)

    def get_bb_direction(self) -> str:
        """
        現在のボリンジャーバンドの方向を取得します。

        Returns:
            str: 現在のボリンジャーバンドの方向。
        """
        return self.bb_direction

    def get_middle_band(self, index) -> float:
        """
        指定されたインデックスでのミドルバンドの価格を取得します。

        Args:
            index (int): ミドルバンドの価格を取得するインデックス。

        Returns:
            float: 指定されたインデックスでのミドルバンドの価格。
        """
        return self.dataloader.get_df(index, COLUMN_MIDDLE_BAND)

    def analyze_profit(self):
        """
        利益に関する統計情報を分析し、結果をデータローダーに保存します。
        """
        # 利益に関する統計情報を分析し、データローダーを通じて保存
        des = self.dataloader.describe(self.entry_index, self.exit_index, COLUMN_CURRENT_PROFIT)
        self.dataloader.set_df(self.entry_index, COLUMN_PROFIT_MAX, des['max'])
        self.dataloader.set_df(self.entry_index, COLUMN_PROFIT_MIN, des['min'])
        self.dataloader.set_df(self.entry_index, COLUMN_PROFIT_MEAN, des['mean'])

        # 移動平均の計算
        entry_ma = self.dataloader.get_df(self.entry_index, 'ema')  # 'ema'は外部定義ファイルで管理されている場合、置き換える
        exit_ma = self.dataloader.get_df(self.exit_index, 'ema')  # 同上

        # BB方向に基づいて利益を計算
        if self.get_bb_direction() == BB_DIRECTION_UPPER:
            self.dataloader.set_df(self.entry_index, COLUMN_PROFIT_MA, exit_ma - entry_ma)
        elif self.get_bb_direction() == BB_DIRECTION_LOWER:
            self.dataloader.set_df(self.entry_index, COLUMN_PROFIT_MA, entry_ma - exit_ma)

    def calculate_current_profit(self) -> float:
        """
        現在の利益を計算し、結果をデータローダーに保存します。

        Returns:
            float: 現在の利益。
        """
        # 現在の利益を計算し、データローダーを通じて保存
        direction = self.get_bb_direction()
        if direction == BB_DIRECTION_UPPER:
            profit = self.dataloader.get_df(self.current_index, COLUMN_CLOSE) - self.dataloader.get_df(self.entry_index, COLUMN_ENTRY_PRICE)
        elif direction == BB_DIRECTION_LOWER:
            profit = self.dataloader.get_df(self.entry_index, COLUMN_ENTRY_PRICE) - self.dataloader.get_df(self.current_index, COLUMN_CLOSE)

        self.dataloader.set_df(self.current_index, COLUMN_CURRENT_PROFIT, profit)
        return profit

    def record_entry_exit_price(self):
        """
        エントリー価格と出口価格を記録します。
        """
        # エントリー価格と出口価格を記録
        self.dataloader.set_df_fromto(self.entry_index, self.exit_index, COLUMN_ENTRY_PRICE, self.get_entry_price())
        self.dataloader.set_df_fromto(self.entry_index, self.exit_index, COLUMN_EXIT_PRICE, self.get_exit_price())
        self.dataloader.set_df_fromto(self.entry_index, self.exit_index, COLUMN_BB_DIRECTION, self.get_bb_direction())

    def change_to_idle_state(self):
        """
        アイドル状態に遷移します。ボリンジャーバンドの方向が設定されている場合、
        現在の利益を計算し、エントリーとエグジットの価格を記録した後、
        利益分析を行い、インデックスをリセットします。
        最後にアイドル状態への遷移を記録します。
        """
        # アイドル状態に遷移するロジック
        if self.get_bb_direction() is not None:
            self.set_exit_price(self.get_close(self.current_index))
            bb_profit = self.calculate_current_profit()
            self.dataloader.set_df(self.entry_index, COLUMN_BB_PROFIT, bb_profit)
            self.record_entry_exit_price()
            self.analyze_profit()
            self.reset_index()
        self.record_state_and_transition(STATE_IDLE)

    def change_to_position_state(self, direction: str):
        """
        ポジション状態に遷移します。指定された方向に基づいて、
        エントリー価格を設定し、現在のインデックスをエントリーインデックスとして記録します。
        最後にポジション状態への遷移を記録します。

        Args:
            direction (str): トレードの方向を示す文字列。
        """
        # ポジション状態に遷移するロジック
        self.set_bb_direction(direction)
        self.set_entry_price(self.get_close(self.current_index))
        self.set_entry_index(self.current_index)
        self.record_state_and_transition(STATE_POSITION)

    def change_to_other_state(self, state: str, direction: str = None):
        """
        他の任意の状態に遷移します。指定された状態に遷移を記録します。

        Args:
            state (str): 遷移する状態の名前。
            direction (str, optional): トレードの方向を示す文字列。デフォルトはNone。
        """
        # 他の状態に遷移するロジック
        self.record_state_and_transition(state)

    def record_state_and_transition(self, state: str):
        """
        状態の遷移を記録し、ログに記録します。状態管理オブジェクトを使用して、
        指定された状態への遷移を処理します。

        Args:
            state (str): 遷移する状態の名前。
        """
        # 状態と遷移を記録
        bb_direction = self.get_bb_direction()
        self.log_transaction(f'go to {state}: {bb_direction}')
        self.record_state(state)
        self._state.handle_request(self, state)

    def print_win_lose(self):
        """
        勝敗の統計情報を表示し、時間経過によるバランスの変化をプロットします。
        """
        self.fx_transaction.display_all_win_rates()
        self.fx_transaction.plot_balance_over_time()


def run_trading(context):
    """
    Run the trading analysis and manage the trading state using the given context.

    Args:
    - trading_analysis (TechnicalAnalyzer): The TechnicalAnalyzer instance with trading data.
    - context (TradingContext): The TradingContext instance to manage state transitions.
    """
    data = context.get_data()
    # Iterate over the analyzed data and generate events based on the data
    for index in range(len(data)):
        context.event_handle(index)



# テスト用のコード
def main():
        # 設定ファイルのパスを指定


    strategy_context = SimulationStrategy()

    context = SimulationStrategyContext(strategy_context)
    context.init_manager()
    table_name = 'BTCUSDT_60_market_data_tech'
    context.load_data_from_datetime_period('2023-01-01 00:00:00', '2024-01-01 00:00:00',table_name)
    context.run_trading(context)
    context.print_win_lose()

if __name__ == "__main__":
    main()