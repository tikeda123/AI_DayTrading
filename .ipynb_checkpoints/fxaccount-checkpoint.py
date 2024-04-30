import pandas as pd
import os
import matplotlib.pyplot as plt

from common.config_manager import ConfigManager
from common.trading_logger_db import TradingLoggerDB
from data_loader_tran import DataLoaderTransactionDB


class FXAccount:
    """
    FX口座管理を行うクラスです。口座の入金・出金、取引記録の管理、
    口座残高の時系列変化のグラフ表示などをサポートします。

    Attributes:
        config_manager (ConfigManager): 設定を管理するインスタンス。
        logger (TradingLogger): ロギングを行うインスタンス。
        contract (str): 取引契約の種類。
        init_amount (float): 初期残高。
        amount (float): 現在の残高。
        leverage (str): 使用するレバレッジ。
        trn_df (pd.DataFrame): 取引記録を格納するDataFrame。
        log_dir (str): ログファイルのディレクトリ。
        file_idf (str): ログファイルの識別子。
    """

    def __init__(self):
        """
        FXAccountのインスタンスを初期化します。

        Args:
            config_manager (ConfigManager): 設定を管理するConfigManagerのインスタンス。
            trading_logger (TradingLogger): トレーディング情報をログに記録するTradingLoggerのインスタンス。
        """
        self.config_manager = ConfigManager()
        self.logger = TradingLoggerDB()
        self.data_loader = DataLoaderTransactionDB()
        self.contract = self.config_manager.get('ACCOUNT', 'CONTRACT')
        self.init_amount = float(self.config_manager.get('ACCOUNT', 'INIT_AMOUNT'))
        self.amount = float(self.config_manager.get('ACCOUNT', 'AMOUNT'))
        self.leverage = self.config_manager.get('ACCOUNT', 'LEVERAGE')

        self.log_dir = self.config_manager.get('LOG', 'LOGPATH')
        self.file_idf = self.config_manager.get('LOG', 'FILE_IDF_AC')
        self.startup_flag = True
        self.initialize_db_log()

    def get_startup_flag(self):
        if self.startup_flag:
            self.startup_flag = False
            return 1
        else:
            return 0

    def initialize_db_log(self):
        self.trn_df = None
        self.table_name = f"{self.contract}"+"_account"
        self.data_loader.create_table(self.table_name,'fxaccount')

    def generate_filename(self) -> str:
        """ログファイルの完全なパスを生成して返します。

        Returns:
            str: ログファイルの完全なパス。
        """
        '''ログファイルの完全なパスを生成します。'''
        return os.path.join(self.log_dir, f"{self.contract}{self.file_idf}")

    def update_transaction_log(self, date, cash_in, cash_out) -> None:
        """取引ログに新しい記録を追加します。

        Args:
            date (str): 取引日。
            cash_in (float): 入金額。
            cash_out (float): 出金額。
        """
        serial = self.data_loader.get_next_serial(self.table_name)

        new_record = {
            'serial': serial,
            'date': date,
            'cash_in': cash_in,
            'cash_out': cash_out,
            'amount': self.amount,
            'startup_flag': self.get_startup_flag()
        }

        if self.trn_df is not None:
            self.trn_df = pd.concat([self.trn_df, pd.DataFrame([new_record])], ignore_index=True)
        else:
            self.trn_df = pd.DataFrame([new_record])

        self.data_loader.write_db(pd.DataFrame([new_record]),self.table_name)

    def save_log(self) -> bool:
        """取引ログをファイルに保存します。

        Returns:
            bool: 保存に成功した場合はTrue、失敗した場合はFalse。
        """
        try:
            self.trn_df.to_csv(self.generate_filename())
        except OSError as e:
            self.logger.log_system_message(f'File Open Error: {self.generate_filename()}, {e}')
            return False
        return True

    def initialize_log(self) -> bool:
        """ログファイルを初期化します。ファイルが存在する場合は削除します。

        Returns:
            bool: 初期化に成功した場合はTrue、失敗した場合はFalse。
        """
        try:
            filename = self.generate_filename()
            if os.path.isfile(filename):
                os.remove(filename)
        except OSError as e:
            self.logger.log_system_message(f'File Open Error: {filename}, {e}')
            return False
        return True

    def withdraw(self, date, cash) -> float:
        """口座から指定された額を引き出します。

        Args:
            date (str): 引き出し日。
            cash (float): 引き出し額。

        Returns:
            float: 実際に引き出された額。
        """
        cash_out = min(cash, self.amount)
        self.amount -= cash_out
        self.update_transaction_log(date, 0, cash_out)
        self.save_log()
        return cash_out

    def deposit(self, date, cash) -> float:
        """口座に指定された額を預けます。

        Args:
            date (str): 預け入れ日。
            cash (float): 預け入れ額。

        Returns:
            float: 実際に預け入れられた額。
        """
        self.amount += cash
        self.update_transaction_log(date, cash, 0)
        self.save_log()
        return cash

    def print_balance(self):
        '''現在の口座残高を表示します。'''
        self.logger.log_message(f'Current balance: {self.amount:.2f}')

    def print_performance(self,time: str):
        '''口座のパフォーマンスを表示します。'''
        date = self.trn_df['date'].iloc[-1] if not self.trn_df.empty else 'N/A'
        self.logger.log_transaction(time,'=' * 55)
        self.logger.log_transaction(time,f'{date}: {self.contract} Leverage: {self.leverage}')
        self.logger.log_transaction(time,f'Initial balance [$]: {self.init_amount:.2f}')
        self.logger.log_transaction(time,f'Final balance   [$]: {self.amount:.2f}')
        perf = ((self.amount - self.init_amount) / self.init_amount * 100)
        self.logger.log_transaction(time,f'Net Performance [%]: {perf:.2f}')
        self.logger.log_transaction(time,'=' * 55)

    def plot_balance_over_time(self):
        '''時系列で口座残高の変化をグラフ表示します。'''
        if self.trn_df.empty:
            self.logger.log_message("取引記録がありません。")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.trn_df['date'], self.trn_df['amount']+self.trn_df['cash_out'], marker='o', linestyle='-', color='blue')
        plt.title('Account Balance Over Time')
        plt.xlabel('Date')
        plt.ylabel('Balance [$]')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()