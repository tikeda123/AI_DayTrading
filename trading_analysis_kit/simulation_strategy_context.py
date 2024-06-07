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
from trading_analysis_kit.trading_context import TradingContext
from trading_analysis_kit.simulation_entry_strategy import BollingerBand_EntryStrategy

# TradingContextを継承したシミュレーション戦略コンテキストクラス
class SimulationStrategyContext(TradingContext):
    """
    シミュレーション戦略のコンテキストを管理するクラスです。

    このクラスは、特定のシミュレーション戦略に基づいて取引を実行するための環境を提供します。具体的には、
    エントリーとエグジット戦略の初期化、インデックスのリセット、ボリンジャーバンドに基づいた取引条件の管理などを行います。

    Attributes:
        fx_transaction (FXTransaction): FX取引を管理するクラスのインスタンス。
        entry_manager (BollingerBand_EntryStrategy): エントリー戦略を管理するクラスのインスタンス。

    Args:
        strategy (SimulationStrategy): シミュレーション戦略を提供するクラスのインスタンス。
    """
    def __init__(self,strategy):
        """
        クラスのコンストラクタです。親クラスのコンストラクタを呼び出し、インデックスをリセットし、
        FX取引クラスのインスタンスを初期化します。

        Args:
            strategy (SimulationStrategy): シミュレーション戦略を提供するクラスのインスタンス。
        """

        super().__init__(strategy)  # 親クラスのコンストラクタを呼び出す
        self.fx_transaction = FXTransaction()
        self.entry_manager = BollingerBand_EntryStrategy()
        self.losscut = self.config_manager.get("ACCOUNT", "LOSSCUT")
        self.leverage = self.config_manager.get("ACCOUNT", "LEVERAGE")
        self.init_amount = self.config_manager.get("ACCOUNT", "INIT_AMOUNT")
        self.ptc = self.config_manager.get("ACCOUNT", "PTC")
        self.make_filenname()

    def make_filenname(self):
        symbol = self.config_manager.get("SYMBOL")
        interval = self.config_manager.get("INTERVAL")
        self.simulation_result_filename = f'{symbol}_{interval}_simulation_result.csv'

    def save_simulation_result(self,context):
        """
        シミュレーション結果を保存します。
        Args:
            context (TradingContext): トレーディングコンテキストオブジェクト。
        """
        save_filepath = parent_dir + '/' + self.config_manager.get('AIML_ROLLING','DATAPATH') + self.simulation_result_filename
        df = context.dm.get_raw_data()
        print(df)
        df.to_csv(save_filepath, index=False)

    def record_max_min_pandl_to_entrypoint(self):
        """
        最大と最小の利益を記録します。
        """
        entry_index = self.dm.get_entry_index()
        exit_index = self.dm.get_exit_index()
        df = self.dm.get_df_fromto(entry_index+1, exit_index)
        min_pandl = df[COLUMN_MIN_PANDL].min()
        max_pandl = df[COLUMN_MAX_PANDL].max()

        self.dm.set_min_pandl(min_pandl, entry_index)
        self.dm.set_max_pandl(max_pandl, entry_index)
        #self.log_transaction(f'entrypoint max_pandl: {max_pandl},min_pandl: {min_pandl}')

    def record_entry_exit_price(self):
        """
        エントリー価格と出口価格を記録します。
        """
        # エントリー価格と出口価格を記録
        entry_index = self.dm.get_entry_index()
        exit_index = self.dm.get_exit_index()

        self.dm.set_df_fromto(entry_index, exit_index, COLUMN_ENTRY_PRICE, self.dm.get_entry_price())
        self.dm.set_df_fromto(entry_index, exit_index, COLUMN_EXIT_PRICE, self.dm.get_exit_price())
        self.dm.set_df_fromto(entry_index, exit_index, COLUMN_BB_DIRECTION, self.dm.get_bb_direction())


    def calculate_current_profit(self,current_price=None) -> float:
        """
        現在の利益を計算し、結果をデータローダーに保存します。
        ここでは、エントリー価格と現在の価格の差を利益を計算します。（Longを基本としています）

        Returns:
            float: 現在の利益。
        """
        if current_price is None:
            current_price = self.dm.get_close_price()

        entry_price = self.dm.get_entry_price()
        pred_type = self.dm.get_prediction()

        qty = self.init_amount * self.leverage / entry_price
        buymnt = (qty * entry_price)  # 買った時の金額
        selmnt = (qty * current_price)

        buy_fee = self.init_amount*self.ptc*self.leverage
        sel_fee = self.init_amount*self.ptc*self.leverage

        if pred_type == PRED_TYPE_LONG:  # LONGの場合の利益計算
            current_profit = selmnt - buymnt  - (buy_fee + sel_fee)# 収益P＆L
        else:  # SHORTの場合の利益計算
            current_profit = buymnt - selmnt  - (buy_fee + sel_fee)# 収益P＆L

        #self.log_transaction(f'current price: {current_price},entry price: {entry_price}')
        return current_profit

    def is_profit_triggered(self,triggered_profit)->(bool,float):
        """
        利益がトリガーされたかどうかを判断します。

        Returns:
            bool: 利益がトリガーされたかどうかの真偽値。

        現在の価格をもとに利益がトリガーされたかどうかを判断し、
        トリガーされた場合はTrueを返します。
        """
        entry_type = self.dm.get_entry_type()
        profit_price = None

        if entry_type == ENTRY_TYPE_LONG:
            profit_price = self.dm.get_high_price()
        else:
            profit_price = self.dm.get_low_price()

        profit = self.calculate_current_profit(profit_price)
        if profit > triggered_profit:
            calculate_profit_price = self.calculate_profit_triggered_price(triggered_profit)
            return True, calculate_profit_price
        return False, profit_price

    def calculate_profit_triggered_price(self, profit) -> float:
        """
        利益がトリガーされた場合の価格を計算します。

        Args:
            profit (float): 利益。

        Returns:
            float: 利益がトリガーされた場合の価格。
        """
        entry_price = self.dm.get_entry_price()
        pred_type = self.dm.get_prediction()
        qty = self.init_amount * self.leverage / entry_price
        buymnt = qty * entry_price
        buy_fee = self.init_amount * self.ptc * self.leverage
        sell_fee = (self.init_amount + profit) * self.ptc * self.leverage

        if pred_type == PRED_TYPE_LONG:
        # LONGの場合の利益計算
            profit_price = (buymnt + buy_fee + sell_fee + profit) / qty
        else:
            # SHORTの場合の利益計算
            profit_price = (buymnt - buy_fee - sell_fee - profit) / qty

        return profit_price


    def is_losscut_triggered(self)->(bool,float):
        """
        損切りがトリガーされたかどうかを判断します。

        Returns:
            bool: 損切りがトリガーされたかどうかの真偽値。

        現在の価格をもとに損切りがトリガーされたかどうかを判断し、
        トリガーされた場合はTrueを返します。
        """
        entry_type = self.dm.get_entry_type()
        losscut_price = None

        if entry_type == ENTRY_TYPE_LONG:
            losscut_price = self.dm.get_low_price()
        else:
            losscut_price = self.dm.get_high_price()

        loss_cut_pandl =  self.losscut*self.init_amount*-1
        pandl = self.calculate_current_profit(losscut_price)

        if pandl < loss_cut_pandl:
            return True, loss_cut_pandl
        return False, pandl

    def change_to_idle_state(self):
        """
        アイドル状態に遷移します。ボリンジャーバンドの方向が設定されている場合、
        現在の利益を計算し、エントリーとエグジットの価格を記録した後、
        利益分析を行い、インデックスをリセットします。
        最後にアイドル状態への遷移を記録します。
        """
        self.dm.reset_index()
        self.record_state_and_transition(STATE_IDLE)

    def change_to_position_state(self):
        """
        ポジション状態に遷移します。指定された方向に基づいて、
        エントリー価格を設定し、現在のインデックスをエントリーインデックスとして記録します。
        最後にポジション状態への遷移を記録します。

        Args:
            direction (str): トレードの方向を示す文字列。
        """
        entry_index = self.dm.get_current_index()
        entry_price = self.dm.get_close_price()
        self.dm.set_entry_index(entry_index)
        self.dm.set_entry_price(entry_price)
        self.record_state_and_transition(STATE_POSITION)

    def change_to_entrypreparation_state(self):
        """
        他の任意の状態に遷移します。指定された状態に遷移を記録します。

        Args:
            state (str): 遷移する状態の名前。
            direction (str, optional): トレードの方向を示す文字列。デフォルトはNone。
        """
        # 他の状態に遷移するロジック
        self.record_state_and_transition(STATE_ENTRY_PREPARATION)

    def record_state_and_transition(self, state: str):
        """
        状態の遷移を記録し、ログに記録します。状態管理オブジェクトを使用して、
        指定された状態への遷移を処理します。

        Args:
            state (str): 遷移する状態の名前。
        """
        # 状態と遷移を記録
        self.log_transaction(f'go to {state}')
        self._state.handle_request(self, state)

    def set_current_max_min_pandl(self):
        h_pandl = self.calculate_current_profit(self.dm.get_high_price())
        l_pandl = self.calculate_current_profit(self.dm.get_low_price())

        max_pandl = max(h_pandl,l_pandl)
        min_pandl = min(h_pandl,l_pandl)

        self.dm.set_max_pandl(max_pandl)
        self.dm.set_min_pandl(min_pandl)

        return max_pandl,min_pandl
        #self.log_transaction(f'current max_pandl: {max_pandl},min_pandl: {min_pandl}')


    def print_win_lose(self):
        """
        勝敗の統計情報を表示し、時間経過によるバランスの変化をプロットします。
        """
        self.fx_transaction.display_all_win_rates()
        self.fx_transaction.plot_balance_over_time()


# テスト用のコード
def main():
        # 設定ファイルのパスを指定

    from trading_analysis_kit.simulation_strategy import SimulationStrategy
    strategy_context = SimulationStrategy()

    context = SimulationStrategyContext(strategy_context)
    context.load_data_from_datetime_period('2024-01-01 00:00:00', '2024-06-01 00:00:00')
    context.run_trading(context)
    context.print_win_lose()
    context.save_simulation_result(context)

if __name__ == "__main__":
    main()