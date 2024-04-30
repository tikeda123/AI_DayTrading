import sys
import os
from datetime import datetime, timedelta
import time

# 既存のパス設定コード
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Program A imports
from trading_analysis_kit.online_strategy import OnlineStrategy
from trading_analysis_kit.simulation_strategy_context import SimulationStrategyContext

# Program B imports
from online.data_loader_online import DataLoaderOnline

# Common imports
from common.trading_logger_db import TradingLoggerDB

def main_process():
    # ロガーの初期化
    logger = TradingLoggerDB()

    # Program Aの初期化
    strategy_context = OnlineStrategy()
    context = SimulationStrategyContext(strategy_context)

    # Program Bの初期化
    data_loader_online = DataLoaderOnline()

    next_run_time = datetime.now()
    while True:
        current_time = datetime.now()
        if current_time >= next_run_time:
            # Program Aの処理
            one_frame = context.get_latest_data()
            if one_frame is not None:
                index = context.get_current_index()
                context.event_handle(index)
                logger.log_verbose_message(one_frame)

            # Program Bの処理
            res = data_loader_online.update_historical_data_to_now()
            logger.log_verbose_message(res)
            res_tech = data_loader_online.convert_historical_recent_data_to_tech()
            logger.log_verbose_message(res_tech)

            # 次の実行時間を1時間後に設定
            next_run_time = current_time + timedelta(seconds=20)

        # 10秒ごとにループする
        time.sleep(1)

if __name__ == "__main__":
    main_process()
