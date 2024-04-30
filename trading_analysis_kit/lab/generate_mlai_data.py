import sys
#from pathlib import Path
import pandas as pd

# Adjust path imports

sys.path.append('/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0')

from common.config_manager import ConfigManager
from common.trading_logger import TradingLogger
from trading_analysis_kit.trading_analysis import TradingAnalysis


class GenerateMLData:
    def __init__(self, config_manager, trading_logger):
        self.config_manager = config_manager
        self.trading_logger = trading_logger
        self.trading_analysis = TradingAnalysis(config_manager, trading_logger)

    def generate_ml_data(self):
        self.trading_analysis.calculate_bollinger_bands()
        self.trading_analysis.calculate_macd()
        self.trading_analysis.calculate_rsi()
        self.trading_analysis.analyze_bollinger_band_breakouts()
        self.trading_analysis.macd_analysis()

        result = self.trading_analysis.analyze_macd_trends_after_bollinger_breakouts("upper")#result = trading_analysis.analyze_macd_trends_after_bollinger_breakouts("upper",isPlus=Fals
        print(result)
        result.to_csv('test01_upper_60.csv')

        result = self.trading_analysis.analyze_macd_trends_after_bollinger_breakouts("lower")
        print(result)
        result.to_csv('test01_lower_60.csv')