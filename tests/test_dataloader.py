import pandas as pd
import operator
import os,sys

sys.path.append('/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0')

from common.trading_logger import TradingLogger
from common.config_manager import ConfigManager
from common.data_loader import DataLoader

def main():
    config_path ='/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/aitrading_settings_ver2.json'
    ConfigManager(config_path)
    TradingLogger()
    data_loader = DataLoader()
    data_loader.load_data_from_csv('result.csv')

    print(data_loader.get_raw())
    df = data_loader.filter_and('bb_direction',operator.eq,'upper','bb_profit',operator.ne,0.0)
    df.to_csv('result_upper_01.csv')
    print(df)
    
    df = data_loader.filter_and('bb_direction',operator.eq,'lower','bb_profit',operator.ne,0.0)
    df.to_csv('result_lower_01.csv')
    print(df)


if __name__ == '__main__':
    main()


