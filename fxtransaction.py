import pandas as pd
import os,sys



from common.trading_logger_db import TradingLoggerDB
from common.config_manager import ConfigManager
from fxaccount import FXAccount
from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import TRANSACTION_DATA

class FXTransactionDataFrame:
    """
    FX取引データを管理するクラスです。取引ログの初期化、新規取引データの追加、
    ログファイルの保存、特定のシリアル番号の存在確認、次のシリアル番号の取得、
    取引データの取得と設定を行います。

    Attributes:
        __logger (TradingLogger): ロギングを行うインスタンス。
        __config_manager (ConfigManager): 設定を管理するインスタンス。
        __fxcol_names (list): FX取引データのカラム名リスト。
        __fxtrn (pd.DataFrame): FX取引データを格納するDataFrame。
    """

    def __init__(self):
        """
        FXTransactionDataFrameのインスタンスを初期化します。

        Args:
            config_manager (ConfigManager): 設定を管理するConfigManagerのインスタンス。
            trading_logger (TradingLogger): トレーディング情報をログに記録するTradingLoggerのインスタンス。
        """
        self.__logger = TradingLoggerDB()
        self.__config_manager = ConfigManager()
        self.__data_loader = MongoDataLoader()
        self.startup_flag = True
        self.initialize_db_log()

    def get_startup_flag(self):
        if self.startup_flag:
            self.startup_flag = False
            return 1
        else:
            return 0

    def initialize_db_log(self):
        self.__fxcol_names = [
            'serial',
            'init_equity',
            'equity',
            'leverage',
            'contract',
            'qty',
            'entry_price',
            'losscut_price',
            'exit_price',
            'limit_price',
            'pl',
            'pred',
            'tradetype',
            'stage',
            'losscut',
            'entrytime',
            'exittime',
            'direction',
            'startup_flag'
        ]
        # 空のデータフレームを生成
        self.__fxtrn = pd.DataFrame(columns=self.__fxcol_names)
        self.__contract = self.__config_manager.get('ACCOUNT', 'CONTRACT')
        self.table_name = f"{self.__contract}"+"_fxtransaction"

    def _create_filename(self):
        """
        ログファイルの完全なファイル名を生成します。

        Returns:
            str: 生成されたファイル名。
        """
        log_dir = self.__config_manager.get('LOG', 'LOGPATH')
        file_idf = self.__config_manager.get('LOG', 'FILE_IDF_FX')
        self.__contract = self.__config_manager.get('ACCOUNT', 'CONTRACT')
        return os.path.join(log_dir, f"{self.__contract}{file_idf}")

    def _check_and_remove_file(self, filename):
        """
        指定されたファイルが存在する場合、それを削除します。

        Args:
            filename (str): 削除するファイルのパス。

        Returns:
            bool: ファイル操作に成功したかどうか。
        """
        try:
            if os.path.isfile(filename):
                os.remove(filename)
        except OSError as e:
            self.__logger.log_system_message(f'File Operation Error: {filename}, {e}')
            return False
        return True

    def _initialize_fxtranzaction_log(self):
        """
        FXトランザクションログファイルを初期化します。

        Returns:
            bool: ログファイルの初期化に成功したかどうか。
        """
        filename = self._create_filename()
        return self._check_and_remove_file(filename)

    def _new_fxtrn_dataframe(self, serial, init_equity, equity, leverage, contract,
                         qty, entry_price, losscut_price,exit_price, limit_price, pl,
                         pred, tradetype, stage, losscut, entrytime, exittime, direction):
        """
        新規FX取引データをDataFrameに追加します。

        Args:
            各引数は取引データのカラムに対応します。
        """
        # 新規レコードをディクショナリとして定義
        new_record = {
            'serial': serial,
            'init_equity': init_equity,
            'equity': equity,
            'leverage': leverage,
            'contract': contract,
            'qty': qty,
            'entry_price': entry_price,
            'losscut_price': losscut_price,
            'exit_price': exit_price,
            'limit_price': limit_price,
            'pl': pl,
            'pred': pred,
            'tradetype': tradetype,
            'stage': stage,
            'losscut': losscut,
            'entrytime': entrytime,
            'exittime': exittime,
            'direction': direction,
            'startup_flag': self.get_startup_flag()
        }
        # DataFrameのデータ型を指定
        dtypes = {
            'serial': 'int64',
            'init_equity': 'float64',
            'equity': 'float64',
            'leverage': 'float64',
            'contract': 'object',
            'qty': 'float64',
            'entry_price': 'float64',
            'losscut_price': 'float64',
            'exit_price': 'float64',
            'limit_price': 'float64',
            'pl': 'float64',
            'pred': 'int64',
            'tradetype': 'object',
            'stage': 'object',
            'losscut': 'int64',
            'entrytime': 'datetime64[ns]',
            'exittime': 'datetime64[ns]',
            'direction': 'object',
            'startup_flag': 'int64'
        }

        # 新規レコードをDataFrameに変換
        new_record_df = pd.DataFrame([new_record])

        # 各列のデータ型を指定
        for column, dtype in dtypes.items():
            new_record_df[column] = new_record_df[column].astype(dtype)

        # 現在のDataFrameに新規レコードのDataFrameを結合

        if not self.__fxtrn.empty:
            self.__fxtrn = pd.concat([self.__fxtrn, new_record_df], ignore_index=True)
        else:
            self.__fxtrn = new_record_df

        new_record_df['exittime'] = pd.to_datetime(new_record_df['exittime'], errors='coerce')
        new_record_df['exittime'] = new_record_df['exittime'].apply(lambda x: x.to_pydatetime() if pd.notnull(x) else None)

        self.__data_loader.insert_data(new_record_df,coll_type=TRANSACTION_DATA)

    def _update_fxtrn_dataframe(self, serial):
        """
        指定されたシリアル番号の取引データを更新します。

        Args:
            serial (int): シリアル番号。
        """
        df = self.__fxtrn[self.__fxtrn['serial'] == serial]
        self.__data_loader.update_data_by_serial(serial ,df,coll_type=TRANSACTION_DATA)

    def save_fxtrn_log(self)->bool:
        """
        FX取引ログをファイルに保存します。

        Returns:
            bool: ログの保存に成功したかどうか。
        """
        filename = self._create_filename()
        if not self._check_and_remove_file(filename):
            return False
        try:
            self.__fxtrn.to_csv(filename)
        except OSError as e:
            self.__logger.log_system_message(f'File Write Error: {filename}, {e}')
            return False
        return True


    def does_serial_exist(self, serial: int)->bool:
        """
        指定されたserialがレコードに存在するかどうかを返します。

        Args:
            serial (int): 確認するシリアル番号。

        Returns:
            bool: 指定されたシリアル番号が存在するかどうか。
        """
        return serial in self.__fxtrn['serial'].values

    def get_next_serial(self)->int:
        """
        次の利用可能なシリアル番号を返します。

        Returns:
            int: 次のシリアル番号。
        """
        return self.__data_loader.get_next_serial(coll_type=TRANSACTION_DATA)



    def get_fxtrn_dataframe(self):
        """
        FX取引のDataFrameを返します。

        Returns:
            pd.DataFrame: FX取引データを格納するDataFrame。
        """
        return self.__fxtrn

    def set_fd(self, serial: int, col: str, valume):
        """
        指定したシリアル番号の取引データの特定のカラムに値を設定します。

        Args:
            serial (int): シリアル番号。
            col (str): 値を設定するカラム名。
            value: 設定する値。
        """
        #print(f'serial:{serial},col:{col},valume:{valume}')
        self.__fxtrn.loc[self.__fxtrn['serial'] == serial, col] = valume

    def get_fd(self, serial: int, col: str):
        """
        指定したシリアル番号の取引データから特定のカラムの値を取得します。

        Args:
            serial (int): シリアル番号。
            col (str): 値を取得するカラム名。

        Returns:
            取得した値。
        """
        value = self.__fxtrn.loc[self.__fxtrn['serial'] == serial, col]
        if not value.empty:
            return value.iloc[0]
        return None


class FXTransaction:
    """
    FX取引を管理するクラスです。取引のエントリーとエグジット処理、損益計算、
    ロスカットの確認、勝率計算などFX取引に関わる一連の操作を行います。

    Attributes:
        __config_manager (ConfigManager): 設定を管理するインスタンス。
        __logger (TradingLogger): ロギングを行うインスタンス。
        __fxac (FXAccount): FX口座管理クラスのインスタンス。
        __fxtrn (FXTransactionDataFrame): FX取引データフレーム管理クラスのインスタンス。
        __contract (str): 取引契約の種類。
        __init_equity (float): 初期資本。
        __leverage (int): レバレッジ。
        __ptc (float): 取引コストのパーセンテージ。
        __losscut (float): ロスカット率。
    """
    def __init__(self):
        """
        FXTransactionのインスタンスを初期化します。

        Args:
            config_manager (ConfigManager): 設定を管理するConfigManagerのインスタンス。
            trading_logger (TradingLogger): トレーディング情報をログに記録するTradingLoggerのインスタンス。
        """
        self.__config_manager = ConfigManager()
        self.__logger = TradingLoggerDB()
        self.__fxac = FXAccount()
        self.__fxtrn = FXTransactionDataFrame()

        # 簡略化した設定変数の読み込み
        self.__contract = self.__config_manager.get('ACCOUNT', 'CONTRACT')
        self.__init_equity = self.__config_manager.get('ACCOUNT', 'INIT_EQUITY')
        self.__leverage = self.__config_manager.get('ACCOUNT', 'LEVERAGE')
        self.__ptc = self.__config_manager.get('ACCOUNT', 'PTC')
        self.__losscut = self.__config_manager.get('ACCOUNT', 'LOSSCUT')
        self.__symbol = self.__config_manager.get('SYMBOL')
        self.set_round()


    def set_round(self):
        if self.__symbol == 'BTCUSDT':
            self.__ROUND_DIGIT = 3
        elif self.__symbol == 'ETHUSDT':
            self.__ROUND_DIGIT = 2
        else:
            self.__ROUND_DIGIT = 2

    def get_qty(self,serial):
        return self.__fxtrn.get_fd(serial,'qty')


    def get_losscut_price(self,serial):
        return self.__fxtrn.get_fd(serial,'losscut_price')

    def trade_entry(self, tradetype: str, pred: int, entry_price: float, entry_time : str, direction: str)->int:
        """
        取引エントリーを行います。新しい取引番号(serial)を戻り値として返します。

        Args:
            tradetype (str): 取引タイプ（'LONG' または 'SHORT'）。
            pred (int): 予測値（1: 上昇予測, 0: 下降予測）。
            entry_price (float): エントリー価格。
            entry_time (str): エントリー時刻。
            direction (str): 取引方向。

        Returns:
            int: 新しい取引番号。
        """
        init_equity = self.__fxac.withdraw(entry_time, self.__init_equity)
        equity = init_equity

        # cash flowが枯渇した事象
        if equity == 0:
            self.__logger.log_message(f'out of cashflow:{equity:.2f}')
            exit(0)

        qty = (equity * self.__leverage )/ entry_price
        qty = round(qty, self.__ROUND_DIGIT)


        if tradetype == 'LONG':
            # 強制決済される価格の下限を計算
            limit_price = entry_price - (equity/qty)
        else:  # SHORT ENTRYの場合
            limit_price = entry_price + (equity/qty)

        # トレード番号を取得
        serial = self.__fxtrn.get_next_serial()
        losscut_price = self.calculate_losscut_price(entry_price,tradetype)
        exit_time = None
        # データフレームに新規レコードを追加
        self.__fxtrn._new_fxtrn_dataframe(serial, init_equity, equity, self.__leverage, self.__contract,
                                     qty, entry_price, losscut_price,0, limit_price, 0,
                                     pred, tradetype, 0, self.__losscut, entry_time, exit_time, direction)
        # ログに出力
        self.__logger.log_transaction(entry_time,f'Entry: {direction}, {tradetype}, pred:{pred}, entry_price{entry_price}')
        self.__fxtrn.save_fxtrn_log()
        return serial


    def get_pandl(self, serial: int, exit_price: float)->float:
        """
        指定された取引番号の損益を計算します。

        Args:
            serial (int): トレード番号。
            exit_price (float): エグジット価格。

        Returns:
            float: 計算された損益。
        """
        # 指定されたserialがレコードに存在しなかったら0を返す
        if not self.__fxtrn.does_serial_exist(serial):
            return 0

        tradetype = self.__fxtrn.get_fd(serial,'tradetype')
        qty = self.__fxtrn.get_fd(serial,'qty')
        entry_price = self.__fxtrn.get_fd(serial,'entry_price')
        equity = self.__fxtrn.get_fd(serial,'equity')

        buymnt = (qty * entry_price)  # 買った時の金額
        selmnt = (qty * exit_price)
        buy_fee = equity*self.__ptc*self.__leverage
        sel_fee = equity*self.__ptc*self.__leverage

        if tradetype == "LONG":  # LONGの場合の利益計算
            return selmnt - buymnt  - (buy_fee + sel_fee)# 収益P＆L
        else:  # SHORTの場合の利益計算
            return buymnt - selmnt  - (buy_fee + sel_fee)# 収益P＆L

    def trade_exit(self, serial: int, exit_price: float, time: str, pandl=None,losscut=None) -> bool:
        """
        トレードエグジットを行います。

        Args:
            serial (int): トレード番号。
            exit_price (float): エグジット価格。
            time (str): エグジット時刻。
            losscut (float, optional): ロスカット価格。

        Returns:
            bool: 正常終了した場合はTrue、そうでなければFalse。
        """
        # 指定されたserialがレコードに存在しなかったら0を返す
        if not self.__fxtrn.does_serial_exist(serial):
            self.__logger.log_message(f'trade_exit:no exit record: {serial}')
            raise ValueError(f'trade_exit:no exit record: {serial}')

        equity = self.__fxtrn.get_fd(serial,'equity')
        tradetype = self.__fxtrn.get_fd(serial,'tradetype')

        if pandl is None:
            pandl = self.get_pandl(serial, exit_price)

        equity += pandl

        # 一回の取引ごとに証拠金はクリアーする。現金口座に戻す。
        self.__fxac.deposit(time, equity)

        # 'LONG_ENTRY -> 'LONG_EXIT'の文字列変換
        tradetype = tradetype.split('_')[0] + '_' + 'EXIT'
        # データフレームに新規レコードを追加
        self.__fxtrn.set_fd(serial, 'equity', equity)
        self.__fxtrn.set_fd(serial, 'exit_price', exit_price)
        self.__fxtrn.set_fd(serial, 'pl', pandl)
        self.__fxtrn.set_fd(serial, 'tradetype', tradetype)
        self.__fxtrn.set_fd(serial, 'exittime', time)

        if losscut is not None:
            self.__fxtrn.set_fd(serial, 'losscut', losscut)
        # ログに出力
        entry_price = self.__fxtrn.get_fd(serial, 'entry_price')
        self.__fxtrn._update_fxtrn_dataframe(serial)
        self.__logger.log_transaction(time,f'Exit: {serial}, {tradetype}, {losscut}, Entry_price:{entry_price}, Exit_price:{exit_price},P%L: {pandl:.2f}')

        # ログファイルに保存
        self.__fxtrn.save_fxtrn_log()
        self.__fxac.print_performance(time)
        return pandl


    def trade_cancel(self, serial: int, date: str):
        """
        指定された取引をキャンセルします。

        Args:
            serial (int): キャンセルする取引のシリアル番号。
        """
        # 指定されたserialがレコードに存在しなかったら0を返す
        if not self.__fxtrn.does_serial_exist(serial):
            self.__logger.log_message(f'trade_cancel:no exit record: {serial}')
            raise ValueError(f'trade_cancel:no exit record: {serial}')

        equity = self.__fxtrn.get_fd(serial,'equity')
        tradetype = self.__fxtrn.get_fd(serial,'tradetype')
        pandl = 0
        equity += pandl

        # 一回の取引ごとに証拠金はクリアーする。現金口座に戻す。
        self.__fxac.deposit(date, equity)

        # 'LONG_ENTRY -> 'LONG_EXIT'の文字列変換
        tradetype = tradetype.split('_')[0] + '_' + 'CANCEL'
        # データフレームに新規レコードを追加
        self.__fxtrn.set_fd(serial, 'equity', equity)
        self.__fxtrn.set_fd(serial, 'exit_price',0)
        self.__fxtrn.set_fd(serial, 'pl', pandl)
        self.__fxtrn.set_fd(serial, 'tradetype', tradetype)
        self.__fxtrn.set_fd(serial, 'exittime', date)

        # ログに出力
        self.__fxtrn._update_fxtrn_dataframe(serial)
        self.__logger.log_transaction(date,f'Cancel: {serial}, {tradetype}')

        # ログファイルに保存
        self.__fxtrn.save_fxtrn_log()
        self.__fxac.print_performance(date)
        return

    def check_losscut(self,serial:int,current_price: float)->(bool,float):
        """
        指定された取引のロスカット条件を現在価格に基づいて確認します。

        ロスカットが発動すべき条件下であるかどうか（現在価格がロスカット価格を
        超えているかどうか）と、設定されたロスカット価格を返します。

        Args:
            serial (int): 確認する取引のシリアル番号。
            current_price (float): 現在の市場価格。

        Returns:
            (bool, float): ロスカットが発動すべきかのブール値とロスカット価格。
                           ロスカットが発動すべき場合はTrue、そうでない場合はFalse。
        """
        if not self.__fxtrn.does_serial_exist(serial):
            self.__logger.log_message(f'check_losscut:no exit record: {serial}')
            raise ValueError(f'check_losscut:no exit record: {serial}')

        losscut_price = self.__fxtrn.get_fd(serial,'losscut_price')
        tradetype = self.__fxtrn.get_fd(serial,'tradetype')

        if tradetype == 'LONG' and losscut_price <= current_price:
            return False ,losscut_price
        elif tradetype == 'SHORT' and losscut_price >= current_price:
            return False,losscut_price
        else:
            return True,losscut_price



    def is_losscut_triggered(self, serial: int, current_price: float) -> (bool, float):
        """
        指定された取引において、現在の価格でロスカットがトリガーされるかどうかを確認します。

        Args:
            serial (int): トレード番号。
            current_price (float): 現在の価格。

        Returns:
            (bool, float): ロスカットがトリガーされたかどうかのブール値と、ロスカット価格。
        """
        LONG = 'LONG'
        SHORT = 'SHORT'
        # 指定されたserialがレコードに存在しない場合
        if not self.__fxtrn.does_serial_exist(serial):
            self.__logger.log_message(f'is_losscut_triggered:no exit record: {serial}')
            #致命的なエラー処理
            raise ValueError(f'is_losscut_triggered:no exit record: {serial}')

            #return False, 0

        losscut_price = self.__fxtrn.get_fd(serial, 'losscut_price')
        tradetype = self.__fxtrn.get_fd(serial, 'tradetype')

        is_long_triggered = tradetype == LONG and losscut_price <= current_price
        #if tradetype == LONG:
        #    print(f'long: losscut_price:{losscut_price},current_price:{current_price}')
        is_short_triggered = tradetype == SHORT and losscut_price >= current_price

        is_triggered = not (is_long_triggered or is_short_triggered)
        return is_triggered, losscut_price


    def calculate_losscut_price(self, entry_price: float, tradetype: str) -> float:
        """
        レバレッジを考慮したロスカット価格を計算します。

        Args:
            entry_price (float): エントリー価格。
            tradetype (str): 取引タイプ（'LONG' または 'SHORT'）。

        Returns:
            float: 計算されたロスカット価格。
        """
        # 手数料を考慮したレバレッジを用いた取引量の計算
        leveraged_amount = self.__init_equity * self.__leverage

        # 許容損失額の計算
        allowable_loss = self.__init_equity * self.__losscut

        # 1ビットコインあたりの許容損失額の計算
        loss_per_btc = allowable_loss / (leveraged_amount / entry_price)

        # 手数料を含めたロスカット価格の計算
        if tradetype == 'LONG':
            return entry_price - loss_per_btc
        else:  # SHORTの場合
            return entry_price + loss_per_btc


    def plot_balance_over_time(self):
        """
        時間経過に伴う口座残高の変化をグラフで表示します。
        """
        # 残高の変化をプロット
        self.__fxac.plot_balance_over_time()

    def calculate_win_rate(self, direction, tradetype_exit):
        """
        指定された方向と取引タイプに基づく勝率を計算し表示します。

        Args:
            direction (str): 取引方向（'lower' または 'upper'）。
            tradetype_exit (str): エグジットの取引タイプ（'LONG_EXIT' または 'SHORT_EXIT'）。
        """
        # 指定された方向と取引タイプに基づいてフィルタリング
        filtered_trades = self.__fxtrn.get_fxtrn_dataframe()[
            (self.__fxtrn.get_fxtrn_dataframe()['direction'] == direction) &
            (self.__fxtrn.get_fxtrn_dataframe()['tradetype'] == tradetype_exit)
        ]

        # 勝ちの取引数
        wins = filtered_trades[filtered_trades['pl'] > 0].shape[0]

        # 負けの取引数
        losses = filtered_trades[filtered_trades['pl'] < 0].shape[0]

        # 総取引数
        total_trades = filtered_trades.shape[0]

        # 勝率を計算（総取引数が0の場合は勝率も0とする）
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

        # 勝った時のplの平均値
        average_win_pl = filtered_trades[filtered_trades['pl'] > 0]['pl'].mean() if wins > 0 else 0

        # 負けた時のplの平均値
        average_loss_pl = filtered_trades[filtered_trades['pl'] < 0]['pl'].mean() if losses > 0 else 0

        self.__logger.log_verbose_message(f"Direction: {direction}, TradeType: {tradetype_exit}, Win Rate: {win_rate:.2f}%, Average Win PL: {average_win_pl:.2f}, Average Loss PL: {average_loss_pl:.2f}")


    def display_all_win_rates(self):
        """
        すべての方向と取引タイプに対する勝率を計算し表示します。
        """
        # 方向と取引タイプの組み合わせ
        directions = ['lower', 'upper']
        tradetype_exits = ['LONG_EXIT', 'SHORT_EXIT']

        # すべての組み合わせについてループ
        for direction in directions:
            for tradetype_exit in tradetype_exits:
                # 勝率を計算し表示
                self.calculate_win_rate(direction, tradetype_exit)

# 以下のようにFXTransactionクラスを使用することができます
# config_manager = ConfigManager(...) # 適切に初期化
# trading_logger = TradingLogger(...) # 適切に初期化
# fx_transaction = FXTransaction(config_manager, trading_logger)

