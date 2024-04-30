import sys
#from pathlib import Path
import pandas as pd

# Adjust path imports

sys.path.append('/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0')

from common.config_manager import ConfigManager
from common.trading_logger import TradingLogger
from trading_analysis_kit.technical_analyzer import TechnicalAnalyzer
from trading_analysis_kit.data_loader import DataLoader


def main():
    config_path ='/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/aitrading_settings_ver2.json'


    config_manager = ConfigManager(config_path)
    trading_logger = TradingLogger(config_manager)
    dataloader = DataLoader(config_manager, trading_logger)
    trading_analysis = TechnicalAnalyzer(dataloader.get_raw(),config_manager, trading_logger)
    df = trading_analysis.get_data()
    df.to_csv('result01.csv')
    print(df)

""" 
    result = trading_analysis.analyze_macd_trends_after_bollinger_breakouts("upper")#result = trading_analysis.analyze_macd_trends_after_bollinger_breakouts("upper",isPlus=Fals
    print(result)
    result.to_csv('test01_upper_240.csv')

    result = trading_analysis.analyze_macd_trends_after_bollinger_breakouts("lower")
    print(result)
    result.to_csv('test01_lower_240.csv')
"""
if __name__ == "__main__":
    main()
