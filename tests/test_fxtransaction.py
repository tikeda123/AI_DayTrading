import os, sys

 # b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得

# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)

from fxtransaction import FXTransaction

def main():

    fx_transaction = FXTransaction()

    """
    # FXAccount インスタンスを作成
    entry_price = 20000
    serial = fx_transaction.trade_entry('LONG',  1, entry_price, '2023-01-01 19:00:00')
    fx_transaction.trade_exit(serial, 'STAGE1', entry_price + 100, '2023-01-01 20:00:00')


    entry_price = 20000
    serial = fx_transaction.trade_entry('LONG',  1, entry_price, '2023-01-02 19:00:00')
    #flag,losscut_price = fx_transaction.check_losscut(serial,19000)
    #print(f'flag:{flag},losscut_price:{losscut_price}')

    #fx_transaction.trade_exit(serial, 'STAGE1', losscut_price, '2023-01-02 20:00:00')
    fx_transaction.trade_exit(serial, 'STAGE1',entry_price-1000, '2023-01-02 20:00:00')

    """
    entry_price = 20000
    serial = fx_transaction.trade_entry('SHORT',  1, entry_price, '2023-01-03 19:00:00',None)
    #flag,losscut_price = fx_transaction.check_losscut(serial,21000)
    #fx_transaction.trade_exit(serial, 'STAGE1',losscut_price, '2023-01-03 20:00:00')
    fx_transaction.trade_exit(serial, entry_price, '2023-01-03 20:00:00')

if __name__ == "__main__":
    main()