import pandas as pd
import sys

sys.path.append('/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0')

from common.config_manager import ConfigManager
from common.trading_logger import TradingLogger
from trading_analysis_kit.trailing_stop import TrailingStopAnalyzer

def main():
    config_path ='/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/aitrading_settings_ver2.json'

    config_manager = ConfigManager(config_path)
    trading_logger = TradingLogger(config_manager)
    trailing_stop = TrailingStopAnalyzer(config_manager, trading_logger)

    file_path = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/data/BTCUSDT_20231224000_2024011110_240.csv'

    btc_data = pd.read_csv(file_path)
    btc_data['exit_price'] = None
    btc_data['islong'] = None
    trailing_stop.apply_trailing_stop_strategy(btc_data)
    print(btc_data.head())

if __name__ == "__main__":
    main()
