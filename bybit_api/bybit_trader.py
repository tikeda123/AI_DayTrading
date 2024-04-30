
import sys,os


# b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得

# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)

from common.constants import *

from bybit_api.bybit_base_api import BybitBaseAPI
from bybit_api.bybit_data_fetcher import BybitDataFetcher
from bybit_api.bybit_order_manager import BybitOrderManager
from bybit_api.bybit_position_manager import BybitPositionManager
from bybit_api.bybit_pnl_manager import BybitPnlManager


class BybitTrader(BybitBaseAPI):
    def __init__(self):
        super().__init__()

        self.data_fetcher = BybitDataFetcher()
        self.order_manager = BybitOrderManager()
        self.position_manager = BybitPositionManager()
        self.pnl_manager = BybitPnlManager()
        self.order_id = None

        self.init_manaers()

    def init_manaers(self):
        self.position_manager.set_my_leverage( self._leverage )


    def get_current_price(self):
        return self.data_fetcher.fetch_latest_info()

    def trade_entry_trigger(self,
                            qty,
                            trade_type,
                            target_price=None,
                            trigger_price=None,
                            stop_loss_price=None):
        """
        トレードのエントリートリガーを設定します。トリガー価格に基づき、現在価格を超える場合に適切に調整します。

        Args:
            qty (float): 取引数量
            trade_type (str): 取引タイプ ('Long' または 'Short')
            target_price (float): 目標価格
            stop_loss_price (float): ストップロス価格
            trigger_price (float): トリガー価格

        Returns:
            str: 注文ID
        """
        current_price = self.data_fetcher.fetch_latest_info()

        if trade_type not in [ENTRY_TYPE_LONG, ENTRY_TYPE_SHORT]:
            raise ValueError("Invalid trade type specified")

        trigger_price = self.adjust_trigger_price(trade_type, trigger_price, current_price)
        qty = self.qty_round(qty)
        print(f'stop_loss_price: {stop_loss_price}')

        self.order_id = self.order_manager.trade_entry_trigger(qty,
                                                               trade_type,
                                                               target_price,
                                                               trigger_price,
                                                               stop_loss_price)
        return self.order_id

    def adjust_trigger_price(self, trade_type, trigger_price, current_price):
        """
        トリガー価格を現在価格に基づいて適切に調整します。
        """
        print(f'trade_type: {trade_type}, trigger_price: {trigger_price}, current_price: {current_price}')
        if trade_type == ENTRY_TYPE_LONG and trigger_price >= current_price:
            return current_price * 0.9999
        elif trade_type == ENTRY_TYPE_SHORT and trigger_price <= current_price:
            return current_price * 1.0001

        return trigger_price

    def get_order_status(self,order_id):
        return self.order_manager.get_order_status(order_id)

    def get_closed_pnl(self):
        pnl,exit_price  = self.pnl_manager.get_pnl()
        return pnl,exit_price

    def trade_exit(self,qty,trade_type):
        qty = self.qty_round(qty)
        return self.order_manager.trade_exit(qty,trade_type)

    def cancel_order(self,order_id):
        return self.order_manager.cancel_order(order_id)

    def get_open_position_status(self):
        return self.position_manager.get_open_position_status()
