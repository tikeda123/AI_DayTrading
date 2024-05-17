import pandas as pd
import os,sys
from datetime import datetime
import operator
import numpy as np
# b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得

# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)

from common.constants import COLUMN_START_AT
from trading_analysis_kit.trading_state import *
from aiml.interface_prediction_manager import init_inference_prediction_rolling_manager
from aiml.transformer_prediction_ts_model import TransformerPredictionTSModel
from aiml.transformer_prediction_rolling_model import TransformerPredictionRollingModel



class BollingerBand_EntryStrategy():
    """
    ボリンジャーバンドに基づくエントリー戦略を実装するクラスです。
    トレンド予測モデルを用いて、現在の市場状況がエントリーに適しているかを判断します。

    Attributes:
        prediction_manager (InferencePredictionManager): トレンド予測を行うための予測マネージャー。
    """
    def __init__(self):
        """
        インスタンスの初期化メソッドです。トレンド予測マネージャーをロードします。
        """
        from common.utils import get_config
        self.conf = get_config("ENTRY")
        self.manager_upper = init_inference_prediction_rolling_manager("upper_mlts",TransformerPredictionTSModel)
        self.manager_lower = init_inference_prediction_rolling_manager("lower_mlts",TransformerPredictionTSModel)
        self.manager_rolling = init_inference_prediction_rolling_manager("rolling",TransformerPredictionRollingModel)
        self.init_model()

    def init_model(self):
        """
        トレンド予測モデルを初期化します。
        """
        self.manager_upper.load_model()
        self.manager_lower.load_model()
        self.manager_rolling.load_model()

    def should_entry(self, context)->bool:
        """
        現在の市場状況がエントリーに適しているかを判断します。

        Args:
            context (TradingContext): トレーディングコンテキストオブジェクト。

        Returns:
            bool,int: エントリーが適切かどうかの真偽値。
        """

        diff = self.conf["DIFFERENCE"]
        current_price = context.dm.get_close_price()
        bb_direction = context.dm.get_bb_direction()
        #pred = PRED_TYPE_LONG
        pred = self.trend_prediction(context)
        #pred = PRED_TYPE_LONG
        rolling_pred = self.predict_trend_rolling(context)
        #print(f'rolling_pred:{rolling_pred}')
        #if pred != PRED_TYPE_SHORT:
        #    return False
        #    return FalsE_SHORT:
        #    return False
        #if pred == PRED_TYPE_SHORT:
        #    return False
        #if pred != rolling_pred:
        #    return False
        if bb_direction == BB_DIRECTION_UPPER and pred == PRED_TYPE_LONG:
            if rolling_pred != pred:
                return False

        if bb_direction == BB_DIRECTION_LOWER and pred == PRED_TYPE_SHORT:
            #if rolling_pred != pred:
                return False

        """
        if bb_direction == BB_DIRECTION_UPPER:
            if pred == PRED_TYPE_LONG and rolling_pred != pred:
                return False

        if bb_direction == BB_DIRECTION_LOWER:
            if pred == PRED_TYPE_SHORT and rolling_pred != pred:
                return False
        """

        if bb_direction == BB_DIRECTION_UPPER:
            should_entry = self._check_upper_entry(context, current_price, diff, pred)
        elif bb_direction == BB_DIRECTION_LOWER:
            should_entry = self._check_lower_entry(context, current_price, diff, pred)
        else:
            should_entry = True

        return should_entry

    def _check_upper_entry(self, context, current_price, diff, pred):
        price_diff = current_price - context.dm.get_middle_price()
        return price_diff >= diff or pred != PRED_TYPE_SHORT

    def _check_lower_entry(self, context, current_price, diff, pred):
        price_diff = context.dm.get_middle_price() - current_price
        return price_diff >= diff or pred != PRED_TYPE_LONG

    def trend_prediction(self, context):
        """
        現在の市場の状況を分析してトレンドを予測します。

        Args:
            context (TradingContext): トレーディングコンテキストオブジェクト。

        Returns:
            int: トレンド予測結果。
        """
        bb_direction = context.dm.get_bb_direction()
        df = context.dm.get_df_fromto(context.dm.get_current_index() - (TIME_SERIES_PERIOD - 1),
                                               context.dm.get_current_index())

        if bb_direction == BB_DIRECTION_UPPER:
            target_df = self.manager_upper.create_time_series_data(df)
            prediction = self.manager_upper.predict_model(target_df)
        elif bb_direction == BB_DIRECTION_LOWER:
            target_df = self.manager_lower.create_time_series_data(df)
            prediction = self.manager_lower.predict_model(target_df)

        context.dm.set_prediction(prediction)
        context.log_transaction(f"Prediction: {prediction}")
        return prediction

    def find_four_hour_interval(self,start_datetime_str):
        """
        上記のプログラムは、指定された日時の4時間間隔の開始時刻を計算する関数です。
        Args:
            start_datetime_str (str): 開始時刻の文字列。

        Returns:
            str: 4時間間隔の開始時刻の文字列。
        """
        # 文字列からdatetimeオブジェクトに変換
        #print(f'start_datetime_str:{start_datetime_str}')
        #start_datetime = datetime.strptime(start_datetime_str, '%Y-%m-%d %H:%M:%S')
        start_datetime = start_datetime_str

        # 4時間間隔の開始時刻を計算
        hours_since_start_of_day = start_datetime.hour + start_datetime.minute / 60 + start_datetime.second / 3600
        four_hour_block_number = int(hours_since_start_of_day // 4)
        four_hour_block_start = start_datetime.replace(hour=four_hour_block_number * 4, minute=0, second=0, microsecond=0)

        # 4時間間隔の開始時刻を文字列で返す
        return four_hour_block_start.strftime('%Y-%m-%d %H:%M:%S')

    def predict_trend_rolling(self, context)->int:
        """
        トレンド予測を行います。

        Args:
            context (TradingContext): トレーディングコンテキストオブジェクト。

        Returns:
            int: トレンド予測結果。
        """
        df = context.dm.get_df_fromto(context.dm.get_current_index() - (TIME_SERIES_PERIOD - 1),
                                               context.dm.get_current_index())
        target_df = self.manager_rolling.create_time_series_data(df)
        prediction = self.manager_rolling.predict_model(target_df)
        #bb_direction = context.dm.get_bb_direction()

        """
        if bb_direction == BB_DIRECTION_UPPER:
            if pred == 1:
                prediction = PRED_TYPE_SHORT
            else:
                prediction = PRED_TYPE_LONG
        elif bb_direction == BB_DIRECTION_LOWER:
            if pred == 1:
                prediction = PRED_TYPE_LONG
            else:
                prediction = PRED_TYPE_SHORT
         """
        context.log_transaction(f"Prediction_rolling for Entry: {prediction}")
        return prediction




