import unittest
import json
import sys
import os
# b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得

# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)

from common.config_manager import ConfigManager


def main():
    # 設定ファイルのパスを指定
    config_path = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/aitrading_settings.json'

    # ConfigManager インスタンスを作成
    config_manager = ConfigManager(config_path)

    # データベース設定を取得
    db_config = config_manager.get('BYBIT_API')
    print("BYBIT_API:", db_config)

    # サーバーポートを取得
    server_port = config_manager.get('ACCOUNT', 'INITAMOUNT')
    print("INITAMOUNT:", server_port)

if __name__ == "__main__":
    main()