import pandas as pd
import operator
import os,sys

# b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得

# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)

from common.trading_logger_db import TradingLoggerDB



def main():

    trading_logger_db = TradingLoggerDB()

    for i in range(10):
        msg = f'test message: {i}'
        trading_logger_db.log_transaction('2021-01-01 03:00:00',msg)


if __name__ == '__main__':
    main()
