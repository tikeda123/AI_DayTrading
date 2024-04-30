import unittest
import json
import sys
import os

sys.path.append('/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0')

from common.config_manager import ConfigManager
from common.trading_logger import TradingLogger

def main():
    # 設定ファイルのパスを指定

    config_path = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/aitrading_settings_v2.json'

    # ConfigManager インスタンスを作成
    config_manager = ConfigManager(config_path)

    # TradingLogger インスタンスを作成
    trading_logger = TradingLogger(config_manager)

    # トレードメッセージの記録
    trading_logger.log_message("Trade executed successfully.")

    # 詳細なトレードメッセージの記録（VERBOSE フラグが True の場合のみ記録される）
    trading_logger.log_verbose_message("Trade details: Buy 10 BTC at $30000.")

    # トランザクションの記録
    trading_logger.log_transaction("2023-01-01", "Sold 5 BTC at $32000.")

    # システムメッセージの記録
    trading_logger.log_system_message("System started successfully.")

if __name__ == "__main__":
    main()