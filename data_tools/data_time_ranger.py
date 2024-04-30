
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common.utils import format_dates, exit_with_message
from common.data_loader_db import DataLoaderDB
from bybit_api.bybit_data_fetcher import BybitDataFetcher
from trading_analysis_kit.technical_analyzer import TechnicalAnalyzer


def parse_command_line_arguments(argv) -> tuple:
    """
    コマンドライン引数から開始日、終了日、およびデータベースフラグを解析して返す。

    Args:
        argv (list): コマンドライン引数のリスト。

    Returns:
        tuple: (開始日, 終了日, データベースフラグ)
    """
    db_flag = '-db' in argv
    dates = [arg for arg in argv[1:] if arg != '-db']

    if len(dates) != 2:
        exit_with_message(
            "使用方法: python script.py <start_date> <end_date> [-db]")

    start_date, end_date = format_dates(*dates)
    return start_date, end_date, db_flag


def main():

    start_date, end_date, db_flag = parse_command_line_arguments(sys.argv)
    print(start_date, end_date, db_flag)


    #bybit_online_api = BybitOnlineAPI()
    bybit_online_api = BybitDataFetcher()

    result = bybit_online_api.fetch_historical_data_all(start_date, end_date)
    print(result)

    if db_flag:
        process_data_with_db_flag(result)


def process_data_with_db_flag(result):
    """
    '-db'フラグが指定された場合に、取得したデータをデータベースに保存し、技術分析を実行する関数。

    この関数は、データベースへのデータの保存と、保存されたデータに対する技術分析を行い、
    分析結果を出力します。この過程は、データの永続化と後続の分析処理のために重要です。

    Args:
        container: 依存性注入コンテナ。アプリケーションの設定やサービスへのアクセスを提供します。
        result (DataFrame): データフレーム形式の取得データ。このデータはデータベースに保存され、分析に使用されます。
    """

    data_loader_db = DataLoaderDB()
    data_loader_db.import_to_db(dataframe=result)

    analyzer = TechnicalAnalyzer()
    analyzer.load_data_from_db()
    analysis_result = analyzer.analyze()
    analyzer.import_to_db()

    print(analysis_result)
    print("データはデータベースに保存されました")


if __name__ == "__main__":
    main()
