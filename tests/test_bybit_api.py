
import sys
from time import sleep
import pandas as pd

sys.path.append('/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0')

from common.config_manager import ConfigManager
from common.trading_logger import TradingLogger
from bybit_api.bybit_api import BybitOnlineAPI

def main():
    # 設定ファイルのパスを指定
    config_path = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/aitrading_settings_ver2.json'

    # ConfigManager インスタンスを作成
    ConfigManager(config_path)

    # TradingLogger インスタンスを作成
    TradingLogger()

    # BybitOnlineAPI インスタンスを作成
    bybit_online_api = BybitOnlineAPI()

    #flag,result = bybit_online_api.sub_get_price_kline(2)
    #print(result)

    #result = bybit_online_api.fetch_historical_price_data("2023-12-20 07:00:00+0900","2023-12-25 07:00:00+0900")
    #result = bybit_online_api.fetch_historical_price_data("2022-12-24 00:00:00+0900","2024-01-1 11:00:00+0900")
    #print(result)

    #result = bybit_online_api.fetch_historical_oi_data("2022-12-24 00:00:00+0900","2024-01-1 11:00:00+0900")

    #result = bybit_online_api.fetch_historical_price_data("2022-12-24 00:00:00+0900","2024-01-1 11:00:00+0900")
    #result = bybit_online_api.aggregate_data("2022-12-24 00:00:00+0900","2024-01-1 11:00:00+0900")
    #flag,result = bybit_online_api.fetch_historical_price_data("2022-12-24 00:00:00+0900","2024-01-1 11:00:00+0900")
    #print(result)
    #result.to_csv('result.csv')

    #flag,result = bybit_online_api.fetch_historical_funding_rate_data("2022-12-24 00:00:00+0900","2024-01-1 11:00:00+0900")
    #print(result)
    #result.to_csv('result.csv')

    #result = bybit_online_api.fetch_historical_data_all("2022-12-24 00:00:00+0900","2024-01-1 11:00:00+0900")
    #flag,result = bybit_online_api.fetch_historical_data("2023-01-01 00:00:00+0900","2024-01-1 11:00:00+0900",'premium_index')
    result = bybit_online_api.fetch_historical_data_all("2023-01-01 00:00:00+0900","2024-01-1 11:00:00+0900")
    print(result)
    result.to_csv('p_result.csv')

    #flag,result = bybit_online_api.fetch_historical_data("2022-12-24 00:00:00+0900","2024-01-1 11:00:00+0900",'funding_rate')
    #print(result)
    #result.to_csv('result.csv')


if __name__ == "__main__":
    main()
