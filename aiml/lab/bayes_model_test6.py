import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os,sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay

# b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Aディレクトリーのパスを取得
# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)
from common.data_loader_db import DataLoaderDB
from common.constants import COLUMN_CLOSE, COLUMN_RSI, COLUMN_MACDHIST, COLUMN_BOL_DIFF

def partial_dependence_plot(df):
    # 特徴量とターゲットを定義
    features = [COLUMN_RSI, COLUMN_MACDHIST, COLUMN_BOL_DIFF]
    target = 'price_increased'

    # 価格が5期間の間に上昇したかどうかを判断
    df[target] = np.where(df[COLUMN_CLOSE].rolling(window=5).max().shift(-5) > df[COLUMN_CLOSE], 1, 0)

    # 特徴量とターゲットをDataFrameに変換
    X = df[features]
    y = df[target]

    # ランダムフォレストモデルの初期化と学習
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 部分依存性プロットの作成
    display = PartialDependenceDisplay.from_estimator(
        model, X, features=features, target=target, n_jobs=-1, grid_resolution=50
    )
    display.figure_.suptitle("Partial Dependence Plots")
    display.figure_.subplots_adjust(top=0.9)  # タイトルとサブプロットの間隔を調整
    plt.show()

# b.pyのディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Aディレクトリーのパスを取得
# Aディレクトリーのパスをsys.pathに追加
sys.path.append(parent_dir)
from common.data_loader_db import DataLoaderDB

def main():
    data_loader = DataLoaderDB()
    df = data_loader.load_data_from_db("btcusdt_60_market_data_tech")
    partial_dependence_plot(df)

if __name__ == "__main__":
    main()