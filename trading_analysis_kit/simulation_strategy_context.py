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
from aiml.interface_prediction_manager import init_inference_prediction_rolling_manager
from aiml.transformer_prediction_rolling_model import TransformerPredictionRollingModel

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
        self.fx_transaction = FXTransaction()
        self.__model = init_inference_prediction_rolling_manager(TransformerPredictionRollingModel)
        self.init_model()
        self.reset_record()
        self.order_id = 0

    def init_model(self):
        self.__model.load_model()
        #self.__model.load_and_prepare_data("202-01-01 00:00:00", "2024-01-01 00:00:00",test_size=0.2, random_state=None)
        #self.__model.evaluate_models()



    def prediction_trend(self):
        """
        現在の市場の状況を分析してトレンドを予測します。

        Args:
            context (TradingContext): トレーディングコンテキストオブジェクト。

        Returns:
            int: トレンド予測結果。
        """
        """
        df = self.dataloader.get_df_fromto(self.current_index, self.current_index)       #target = df[self.prediction_manager.get_feature_columns()]
        target =  df.iloc[0]
        prediction = self.__model.predict_price_direction(target)
        self.set_prediction(prediction)
        self.log_transaction(f"Prediction: {prediction}")
        return prediction
        """
        df = self.dataloader.get_df_fromto(self.current_index-7, self.current_index)       #target = df[self.prediction_manager.get_feature_columns()]
        prediction = self.__model.predict_rolling_model(df)
        self.set_prediction(prediction)
        self.log_transaction(f"Prediction: {prediction}")
        return prediction


    def reset_record(self):
        """
        レコードをリセットします。
        """
        self.order_id = 0
        self.reset_index()

    def set_order_id(self, order_id):
        """
        注文IDを設定します。

        Args:
            order_id (int): 注文ID。
        """
        self.order_id = order_id

    def get_order_id(self) -> int:
        """
        注文IDを取得します。

        Returns:
            int: 注文ID。
        """
        return self.order_id

    def calculate_current_profit(self) -> float:
        """
        現在の利益を計算し、結果をデータローダーに保存します。

        Returns:
            float: 現在の利益。
        """
        pass

    def change_to_idle_state(self):
        """
        アイドル状態に遷移します。ボリンジャーバンドの方向が設定されている場合、
        現在の利益を計算し、エントリーとエグジットの価格を記録した後、
        利益分析を行い、インデックスをリセットします。
        最後にアイドル状態への遷移を記録します。
        """
        self.reset_record()
        self.record_state_and_transition(STATE_IDLE)

    def change_to_position_state(self):
        """
        ポジション状態に遷移します。指定された方向に基づいて、
        エントリー価格を設定し、現在のインデックスをエントリーインデックスとして記録します。
        最後にポジション状態への遷移を記録します。

        Args:
            direction (str): トレードの方向を示す文字列。
        """
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
        self.record_state(state)
        self._state.handle_request(self, state)


    def print_win_lose(self):
        """
        勝敗の統計情報を表示し、時間経過によるバランスの変化をプロットします。
        """
        self.fx_transaction.display_all_win_rates()
        self.fx_transaction.plot_balance_over_time()


# テスト用のコード
def main():
        # 設定ファイルのパスを指定


    strategy_context = SimulationStrategy()

    context = SimulationStrategyContext(strategy_context)
    context.load_data_from_datetime_period('2024-01-01 00:00:00', '2024-01-31 00:00:00')
    context.run_trading(context)
    context.print_win_lose()

if __name__ == "__main__":
    main()