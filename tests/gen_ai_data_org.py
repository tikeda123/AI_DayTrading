import pandas as pd
import sys

sys.path.append('/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0')

from common.config_manager import ConfigManager
from common.trading_logger import TradingLogger
from trading_analysis_kit.trading_strategy_org import TradingStrategy
from trading_analysis_kit.technical_analyzer import TechnicalAnalyzer
from trading_analysis_kit.data_loader import DataLoader

def main():
    config_path ='/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/aitrading_settings_ver2.json'

    config_manager = ConfigManager(config_path)
    trading_logger = TradingLogger(config_manager)
    data_loader = DataLoader(config_manager, trading_logger)
    data_loader.load_test_data()
    btc_data = data_loader.get_raw()

    technical_analyzer = TechnicalAnalyzer(btc_data,config_manager, trading_logger)
    analysis_result = technical_analyzer.analize()
    print(analysis_result)
    trading_strategy = TradingStrategy(analysis_result,config_manager, trading_logger)
 
    resutl=trading_strategy.analyze_bollinger_band_breakouts()
    resutl.to_csv('oi_test01_60.csv')
    upper_result = trading_strategy.analyze_macd_trends_after_bollinger_breakouts("upper")
    upper_result.to_csv('oi_test01_upper_60.csv')
    print(upper_result)

    lower_result = trading_strategy.analyze_macd_trends_after_bollinger_breakouts("lower")
    lower_result.to_csv('oi_test01_lower_60.csv')
    print(lower_result)

 
    

    """ 
    btc_data['exit_price'] = None
    btc_data['islong'] = None
    trailing_stop.apply_trailing_stop_strategy(btc_data)
    print(btc_data.head())
    """
if __name__ == "__main__":
    main()
