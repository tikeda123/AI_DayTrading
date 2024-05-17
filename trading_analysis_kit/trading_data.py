import pandas as pd
import os,sys


from common.constants import *

class TradingStateData:
    def __init__(self) -> None:
        self._entry_index = 0
        self._exit_index = 0
        self._fx_serial = 0
        self._order_id = 0
        self._bb_direction = None
        self._entry_price = 0.0
        self._prediction = 0
        self._entry_type = None

    def reset_index(self):
        """
        現在のインデックス、エントリーインデックス、エグジットインデックス、およびFXシリアル番号をリセットします。
        """
        self._entry_index = 0
        self._entry_type = None
        self._exit_index = 0
        self._fx_serial = 0
        self._order_id = 0
        self._bb_direction = None
        self._entry_price = 0.0
        self._prediction = 0

    def get_entry_type(self) -> str:
        """
        エントリータイプを取得します。

        Returns:
            str: エントリータイプ。
        """
        return self._entry_type

    def set_entry_type(self, entry_type: str):
        """
        エントリータイプを設定します。

        Args:
            entry_type (str): エントリータイプ。
        """
        self._entry_type = entry_type

    def get_entry_price(self) -> float:
        """
        エントリー価格を取得します。

        Returns:
            float: エントリー価格。
        """
        return self._entry_price

    def set_entry_price(self, price: float):
        """
        エントリー価格を設定します。

        Args:
            price (float): エントリー価格。
        """
        self._entry_price = price

    def get_prediction(self) -> int:
        return self._prediction

    def set_prediction(self, prediction: int):
        self._prediction = prediction

    def get_current_index(self) -> int:
        """
        現在のインデックスを取得します。

        Returns:
            int: 現在のインデックス番号。
        """
        return self._current_index

    def set_current_index(self, index: int):
        """
        現在のデータインデックスを設定します。

        Args:
            index (int): 設定するインデックスの値。
        """
        self._current_index = index

    def get_bb_direction(self) -> str:
        """
        Bollinger Bandの方向を取得します。

        Returns:
            str: Bollinger Bandの方向。
        """
        return self._bb_direction

    def set_bb_direction(self, direction: str):
        """
        Bollinger Bandの方向を設定します。

        Args:
            direction (str): Bollinger Bandの方向。
        """
        self._bb_direction = direction

    def get_order_id(self) -> int:
        """
        注文IDを取得します。

        Returns:
            int: 注文ID。
        """
        return self._order_id

    def set_order_id(self, id):
        """
        注文IDを設定します。

        Args:
            order_id (int): 注文ID。
        """
        self._order_id = id

    def get_entry_counter(self) -> int:
        """
        エントリーのカウンターを取得します。

        Returns:
            int: 現在のエントリーのカウンター値。
        """
        return self._entry_counter

    def set_entry_counter(self, counter: int):
        """
        エントリーのカウンターを設定します。

        Args:
            counter (int): 設定するカウンターの値。
        """
        self._entry_counter = counter

    def get_entry_index(self) -> int:
        """
        エントリーしたトレードのインデックスを取得します。

        Returns:
            int: トレードのエントリーインデックス。
        """
        return self._entry_index

    def set_entry_index(self, index: int):
        """
        エントリーしたトレードのインデックスを設定します。

        Args:
            index (int): 設定するインデックスの値。
        """
        self._entry_index = index

    def get_exit_index(self) -> int:
        """
        エグジットしたトレードのインデックスを取得します。

        Returns:
            int: トレードのエグジットインデックス。
        """
        return self._exit_index

    def set_exit_index(self, index: int):
        """
        エグジットしたトレードのインデックスを設定します。

        Args:
            index (int): 設定するインデックスの値。
        """
        self._exit_index = index

    def get_fx_serial(self) -> int:
        """
        FX取引のシリアル番号を取得します。

        Returns:
            int: FX取引のシリアル番号。
        """
        return self._fx_serial

    def set_fx_serial(self, serial: int):
        """
        FX取引のシリアル番号を設定します。

        Args:
            serial (int): 設定するシリアル番号。
        """
        self._fx_serial = serial









