
import sys,os
import time


# b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得

# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)

from common.constants import *
from bybit_api.bybit_trader import BybitTrader

def main():
    trader = BybitTrader()

    current_price = trader.data_fetcher.fetch_latest_info()

    qty = 0.331
    trade_tpye = ENTRY_TYPE_LONG
    target_price = current_price*0.9999
    trigger_price = target_price*0.9999
    stop_loss_price = target_price *0.98
    print(f'current_price: {current_price}, target_price: {target_price}, stop_loss_price: {stop_loss_price}')

    order_id = trader.trade_entry_trigger(qty,
                                          trade_tpye,
                                          target_price,
                                          trigger_price,
                                          stop_loss_price)
    print(order_id)

    for i in range(100):
        order_status = trader.get_order_status(order_id)
        print(f'order_status: {order_status}')
        if order_status == 'Filled':
            break
        time.sleep(2)

    for i in range(10):
        position_status = trader.position_manager.get_open_position_status()
        print(f'position_status: {position_status}')
        time.sleep(2)

    success,id = trader.trade_exit(qty,trade_tpye)

    if not success:
        print('trade exit failed')
        return

    print(f'trade exit success, order_id: {id}')

    time.sleep(2)
    pnl,exit_price = trader.get_closed_pnl()
    print(f'pnl: {pnl}, exit_price: {exit_price}')

    for i in range(5):
        position_status = trader.position_manager.get_open_position_status()
        print(f'position_status: {position_status}')
        time.sleep(2)
if __name__ == '__main__':
    main()
