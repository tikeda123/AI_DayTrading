import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# CSVファイルの読み込み
file_path = 'test01_upper_240.csv'  # ファイルパス
btc_data = pd.read_csv(file_path)

# 目的変数の作成：'bb_diff' > 0 の場合はTrue、そうでない場合はFalse
btc_data['target'] = btc_data['bb_diff'] > 0

# 特徴量の選択
features = [
    'close',  'volume', 'turnover',
    'macd', 'macdsignal', 'macdhist',
    'upper', 'lower', 'middle','rsi'
]

# 特徴量 (X) とターゲット (y) の分離
X = btc_data[features]
y = btc_data['target']

# 特徴量の標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ランダムフォレストモデルの設定
model = RandomForestClassifier(random_state=42)

# K分割クロスバリデーションの設定
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# クロスバリデーションでのモデル評価
cross_val_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='accuracy')

# 結果の表示
print("クロスバリデーションスコア: ", cross_val_scores)
print("平均スコア: ", np.mean(cross_val_scores))
