import sys
import os
import pandas as pd

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

# b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得

# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)


from common.trading_logger import TradingLogger
from common.config_manager import ConfigManager
from mongodb.data_loader import DataLoader
from common.constants import *

class MongoDataLoader(DataLoader):
    """
    MongoDBからデータを読み込むためのクラスです。

    Args:
        host (str): MongoDBのホスト名。
        port (int): MongoDBのポート番号。
        username (str): MongoDBの認証ユーザー名。
        password (str): MongoDBの認証パスワード。
        database (str): 使用するデータベース名。
        collection (str): 使用するコレクション名。
    """

    def __init__(self):
        super().__init__()
        self.logger = TradingLogger()
        self.config = ConfigManager()
        self.symbol = self.config.get('SYMBOL')
        self.interval = self.config.get('INTERVAL')
        self.host = self.config.get('MONGODB', 'HOST')
        self.port = int(self.config.get('MONGODB', 'PORT'))
        self.username =  self.config.get('MONGODB', 'USERNAME')
        self.password = self.config.get('MONGODB', 'PASSWORD')
        self.database = self.config.get('MONGODB', 'DATABASE')
        self.set_collection_name(MARKET_DATA)
        self.set_seq_collection_name(TRADING_LOG)

    def set_collection_name(self, collection_name):
        """
        コレクション名を生成します。
        """
        collection_dict = {
            MARKET_DATA: f"{self.symbol}_{self.interval}_market_data",
            MARKET_DATA_TECH: f"{self.symbol}_{self.interval}_market_data_tech",
            MARKET_DATA_ML_UPPER: f"{self.symbol}_{self.interval}_market_data_mlts_upper",
            MARKET_DATA_ML_LOWER: f"{self.symbol}_{self.interval}_market_data_mlts_lower",
            TRANSACTION_DATA: "transaction_data",
            ACCOUNT_DATA: "account_data",
            TRADING_LOG: "trading_log"
        }

        self.collection = collection_dict[collection_name]

    def set_seq_collection_name(self, collection_name):
        seq_collection_dict = {
        TRADING_LOG: "trading_log_seq",
        TRANSACTION_DATA: "transaction_data_seq",
        ACCOUNT_DATA: "account_data_seq"
        }

        self.seq_collection = seq_collection_dict[collection_name]

    def set_unique_index(self, field_name):
        unique_index_dict = {
            MARKET_DATA: "start_at",
            MARKET_DATA_TECH: "start_at",
            MARKET_DATA_ML_UPPER: "start_at",
            MARKET_DATA_ML_LOWER: "start_at",
            TRANSACTION_DATA: "serial",
            ACCOUNT_DATA: "serial",
            TRADING_LOG: "serial"
        }
        self.unique_index = unique_index_dict[field_name]

    def get_next_serial(self,coll_type=None):
        if coll_type is not None:
            self.set_seq_collection_name(coll_type)  # set_seq_collection_nameではなくset_collection_nameを使用
        self.connect(coll_seq=True)  # coll_seqパラメータをTrueに設定
        try:
            query = {'_id': 'serial'}
            update = {'$inc': {'seq': 1}}
            result = self.col.find_one_and_update(query, update, upsert=True, return_document=True)  # return_documentパラメータを追加
            return result['seq']
        except OperationFailure as e:
            self.logger.log_system_message(f"シーケンスの取得に失敗しました: {str(e)}")
            raise
        finally:
            self.close()

    def convert_marketdata(self,df):
        # start_atとdateをdatetime型に変換
        from datetime import datetime
        df['start_at'] = pd.to_datetime(df['start_at'], unit='s')
        df['date'] = pd.to_datetime(df['date'])

        # open, high, low, close, volume, turnoverをfloat型に変換
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover']
        df[numeric_columns] = df[numeric_columns].astype(float)
        return df


    def connect(self, coll_seq=None):
        """
        MongoDBに接続します。
        """
        try:
            self.client = MongoClient(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password
            )
            self.db = self.client[self.database]

            if coll_seq is not None:
                self.col = self.db[self.seq_collection]
            else:
                self.col = self.db[self.collection]
        except ConnectionFailure as e:
            self.logger.log_system_message(f"MongoDBへの接続に失敗しました: {str(e)}")
            raise

    def load_data_from_datetime_period(self, start_date, end_date, coll_type=None):
        """
        MongoDBから指定された日時範囲のデータを読み込み、DataFrameに変換します。
        Args:
            start_date (datetime): データの開始日時。
            end_date (datetime): データの終了日時。
            coll_type (str, optional): コレクションのタイプ。デフォルトはNone。
        """
        if coll_type is not None:
            self.set_collection_name(coll_type)
        self.connect()
        try:
            query = {
                'start_at': {
                    '$gte': pd.to_datetime(start_date),
                    '$lt': pd.to_datetime(end_date)
                }
            }
            #self.logger.log_system_message(f"Query: {query}")  # クエリの内容をログに出力
            data = list(self.col.find(query))
            #self.logger.log_system_message(f"Data count: {len(data)}")  # データの件数をログに出力
            if len(data) > 0:
                self._df = pd.DataFrame(data)
                self._df['start_at'] = pd.to_datetime(self._df['start_at'])
                #self._df.set_index('start_at', inplace=True)
                self.set_df_raw(self._df)
                self.remove_unuse_colums()
                print(self._df)
                return self._df
            else:
                print("指定された日時範囲のデータが見つかりませんでした。")
                self.logger.log_system_message("指定された日時範囲のデータが見つかりませんでした。")
        except OperationFailure as e:
            self.logger.log_system_message(f"データの読み込みに失敗しました: {str(e)}")
            raise
        finally:
            self.close()


    def load_data(self,coll_type=None):
        """
        MongoDBからデータを読み込み、DataFrameに変換します。
        """
        if coll_type is not None:
            self.set_collection_name(coll_type)

        self.connect()

        try:
            data = list(self.col.find())
            self._df = pd.DataFrame(data)
            self.set_df_raw(self._df)
            self.remove_unuse_colums()
        except OperationFailure as e:
            self.logger.log_system_message(f"データの読み込みに失敗しました: {str(e)}")
            raise
        finally:
            self.close()

    def close(self):
        """
        MongoDBへの接続を閉じます。
        """
        try:
            self.client.close()
        except Exception as e:
            self.logger.log_system_message(f"MongoDBへの接続の切断に失敗しました: {str(e)}")
            raise

    def create_unique_index(self,coll_type=None):
        """ 指定されたフィールドにユニークインデックスを作成します。 """
        if coll_type is not None:
            self.set_unique_index(coll_type)

        field_name = self.unique_index
        self.connect()
        try:
            self.col.create_index([(field_name, 1)], unique=True)
            #self.logger.log_system_message(f"ユニークインデックスを作成しました: {field_name}")
        except OperationFailure as e:
            if "already exists" in str(e):
                #self.logger.log_system_message(f"ユニークインデックスは既に存在します: {field_name}")
                pass
            else:
                self.logger.log_system_message(f"インデックスの作成に失敗しました: {str(e)}")
                raise
        finally:
            self.close()

    def insert_data(self, data, coll_type=None):
        if coll_type is not None:
            self.set_collection_name(coll_type)
            self.create_unique_index(coll_type)

        self.connect()

        try:
            docs = data.to_dict(orient='records')
            for doc in docs:
                try:
                    self.col.insert_one(doc)
                except OperationFailure as e:
                    pass
                    #self.logger.log_system_message(f"データの挿入に失敗しました（重複するキー）: {str(e)}")
                    # 重複エラーを無視するか、ここで処理を行う
                    #print(f"データの挿入に失敗しました（重複するキー）: {str(e)}")
        except Exception as e:
            self.logger.log_system_message(f"データの挿入処理中にエラーが発生しました: {str(e)}")
            raise
        finally:
            self.close()

    def update_data(self, query, update, coll_type=None):
        """
        MongoDBのデータを更新します。

        Args:
            query (dict): 更新対象のドキュメントを指定するクエリ。
            update (dict): 更新内容。
        """
        if coll_type is not None:
            self.set_collection_name(coll_type)
            self.create_unique_index(coll_type)

        self.connect()
        try:
            self.col.update_many(query, {'$set': update})
        except OperationFailure as e:
            self.logger.log_system_message(f"データの更新に失敗しました: {str(e)}")
            raise
        finally:
            self.close()

    def update_data_by_serial(self, serial_id, new_df, coll_type=None):
        if coll_type is not None:
            self.set_collection_name(coll_type)
            self.create_unique_index(coll_type)

        self.connect()
        try:
            query = {'serial': serial_id}
            update_data = new_df.to_dict(orient='records')[0]
            result = self.col.update_one(query, {'$set': update_data})
            if result.modified_count == 0:
                self.logger.log_system_message(f"'serial'が{serial_id}のドキュメントが見つかりませんでした。")
        except OperationFailure as e:
            self.logger.log_system_message(f"データの更新に失敗しました: {str(e)}")
            raise
        finally:
            self.close()

    def delete_data(self, query, coll_type=None):
        """
        MongoDBのデータを削除します。

        Args:
            query (dict): 削除対象のドキュメントを指定するクエリ。
        """
        if coll_type is not None:
            self.set_collection_name(coll_type)

        self.connect()
        try:
            self.col.delete_many(query)
        except OperationFailure as e:
            self.logger.log_system_message(f"データの削除に失敗しました: {str(e)}")
            raise
        finally:
            self.close()

def main():
    """
    メイン処理です。
    """


    data_loader = MongoDataLoader()
    #data_loader.set_collection_name(MARKET_DATA)
    #data_loader.connect()
    #data_loader.create_unique_index('start_at')  # start_atにユニークインデックスを作成
    #data_loader.set_df_raw(df)
    #print(data_loader.get_df_raw())
    #data_loader.insert_data(df)
    data_loader.load_data_from_datetime_period('2023-01-01', '2024-02-01', MARKET_DATA)
    df = data_loader.get_df_raw()
    print(df)

    for i in range(10):
        seq = data_loader.get_next_serial(TRADING_LOG)
        print(f'trading_log_seq: {seq}')

    for i in range(10):
        seq = data_loader.get_next_serial(ACCOUNT_DATA)
        print(f'account_data: {seq}')

    for i in range(10):
        seq = data_loader.get_next_serial(TRANSACTION_DATA)
        print(f'trans_data: {seq}')

    data_loader.close()

if __name__ == '__main__':
    main()



