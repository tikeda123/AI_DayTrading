

import sys,os
from pybit.unified_trading import HTTP

 # b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得

# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)


from bybit_api.bybit_trader import BybitTrader

def main():
    trader = BybitTrader()
    current_price = trader.data_fetcher.fetch_latest_info()

    target_price = current_price * 1.01
    stop_loss_price = current_price * 0.95

    order_id = trader.trade_entry(ENTRY_TYPE_SHORT,0.331,target_price,stop_loss_price)

    print(order_id)
