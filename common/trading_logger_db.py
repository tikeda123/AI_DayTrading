import logging
import pandas as pd
import sys
import os

# b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得

# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)


from common.trading_logger import TradingLogger
from common.constants import *
from mongodb.data_loader_mongo import MongoDataLoader

class TradingLoggerDB(TradingLogger):
    """
    TradingLoggerクラスを継承し、データベースへの書き込み機能を追加したクラス。

    Attributes:
        __db_loader (DataLoaderTransactionDB): データベースへの書き込み用ロガー。

    Args:
        conf (dict): ロガー設定情報を含む辞書。'VERBOSE', 'LOGPATH', 'LOGFNAME', 'LOGLVL', 'DB_CONFIG'のキーを期待します。
    """
    def __init__(self):
        super().__init__()
        from common.utils import get_config
        conf = get_config('LOG')

        self._initialized = True
        self._table_name = conf["DB_TABLE_NAME"]
        self._db_flag = conf["DB_FLAG"]
        self._verbose = conf['VERBOSE']
        self._db_loader = MongoDataLoader()


    def log_transaction(self, date: str, message: str):
        """
        トランザクションをログに記録し、CSVファイルとデータベースにも追加します。

        Args:
            date (str): トランザクションの日付。
            message (str): トランザクションのメッセージ。
        """
        if self._verbose == False:
            return

        serial = self._db_loader.get_next_serial(TRADING_LOG)
        new_record = {'serial': serial, 'date': date, 'message': message}
        new_df = pd.DataFrame([new_record])  # 辞書をDataFrameに変換
        self._tradelog_df = pd.concat([self._tradelog_df, new_df], ignore_index=True)  # DataFrameを連結
        self._tradelog_df.to_csv(self._logfilename_csv, index=False)
        self.log_message(f'{date}|{message}')

        # データベースへの書き込み
        if self._db_flag:
            self._db_loader.insert_data(new_df, coll_type=TRADING_LOG)
        return serial

    def log_transaction_update(self, serial: int ,date: str, message: str):
        """
        トランザクションをログに記録し、CSVファイルとデータベースにも追加します。

        Args:
            date (str): トランザクションの日付。
            message (str): トランザクションのメッセージ。
        """
        new_record = {'serial': serial, 'date': date, 'message': message}
        new_df = pd.DataFrame([new_record])  # 辞書をDataFrameに変換
        self._tradelog_df = pd.concat([self._tradelog_df, new_df], ignore_index=True)  # DataFrameを連結
        self._tradelog_df.to_csv(self._logfilename_csv, index=False)
        self.log_message(f'{date}|{message}')

        # データベースへの書き込み
        if self._db_flag:
            self._db_loader.update_data_by_serial(serial,new_df, coll_type=TRADING_LOG)


def main():
    logger = TradingLoggerDB()
    serial = logger.log_transaction('2020-01-01', 'test message 11')
    logger.log_transaction('2020-01-01', 'test message 22')
    logger.log_transaction('2020-01-01', 'test message 32')
    logger.log_transaction('2020-01-01', 'test message 42')

    print(f'serial: {serial}')
    logger.log_transaction_update(serial, '2020-01-01', 'test message 52')


if __name__ == '__main__':
    main()