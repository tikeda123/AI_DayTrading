
import os, sys

 # b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得

# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)

from common.init_common_module import init_common_module
from common.utils import configure_container
from fxaccount import FXAccount

def main():
    container = configure_container(name=__name__)
    # Context setup
    #strategy_context = MLDataCreationStrategy()
    #context = TradingContext(strategy_context, config_manager, trading_logger, dataloader, trading_analysis)
    init_common_module()

    # FXAccount インスタンスを作成
    print("FXAccount インスタンスを作成")
    fx_account = FXAccount()

    # テスト用のデータを作成
    fx_account.deposit('2023-1-11  22:00:00', 500)

    # $200を引き出す
    fx_account.withdraw('2023-1-11  23:00:00', 200)
    fx_account.withdraw('2023-1-12  00:00:00', 100)
    fx_account.withdraw('2023-1-12  01:00:00', 200)

    #fx_account.save_fxaccount_log()
    # 現在の残高とパフォーマンスを表示する
    fx_account.print_balance()
    fx_account.print_performance()



if __name__ == "__main__":
    main()
