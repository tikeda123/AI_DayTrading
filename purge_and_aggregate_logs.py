import numpy as np
import psycopg2
from psycopg2 import extras
from psycopg2.extensions import register_adapter
register_adapter(np.int64, psycopg2._psycopg.AsIs)
import pandas as pd
import os, sys


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from common.utils import get_config
from common.trading_logger_db import TradingLoggerDB
from common.data_loader_tran import DataLoaderTransactionDB


def PurgeAndAggregateLogs():
    """
    ログデータを集計し、集計データをデータベースに保存します。
    """
    # データベース接続
    conf = get_config("ONLINE")
    symbol = conf["SYMBOL"]

    table_name_trading_log = 'trading_log'
    table_name_account_log = f'{symbol}_account'
    table_name_fxtransaction_log = f'{symbol}_fxtransaction'

    data_loader = DataLoaderTransactionDB()
    logger = TradingLoggerDB()

    data_loader.create_table(table_name_trading_log,'trade_log',is_aggregated=True)
    df = data_loader.read_db(table_name_trading_log,num_rows=-1)
    data_loader.write_db_aggregated_table(df,table_name_trading_log)
    data_loader.drop_table_if_exists(table_name_trading_log)
    logger.log_debug_message(f"trading_log aggregated and purged. {len(df)} rows.")
    logger.log_debug_message(df)


    data_loader.create_table(table_name_account_log,'fxaccount',is_aggregated=True)
    df = data_loader.read_db(table_name_account_log,num_rows=-1)
    data_loader.write_db_aggregated_table(df,table_name_account_log)
    data_loader.drop_table_if_exists(table_name_account_log)
    logger.log_debug_message(f"account_log aggregated and purged. {len(df)} rows.")
    logger.log_debug_message(df)



    data_loader.create_table(table_name_fxtransaction_log,'fxtransaction',is_aggregated=True)
    df = data_loader.read_db(table_name_fxtransaction_log,num_rows=-1)
    data_loader.write_db_aggregated_table(df,table_name_fxtransaction_log)
    data_loader.drop_table_if_exists(table_name_fxtransaction_log)
    logger.log_debug_message(f"fxtransaction_log aggregated and purged. {len(df)} rows.")
    logger.log_debug_message(df)




if __name__ == "__main__":
    PurgeAndAggregateLogs()
