import pandas as pd
import operator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler,PolynomialFeatures

# CSVファイルの読み込み

""" 
# 目的変数の作成
btc_data['target'] = btc_data['bb_diff'].apply(lambda x: 1 if x >= 200 else (-1 if x <= -200 else 0))

# 各クラスの割合を計算して表示
positive_ratio = np.mean(btc_data['target'] == 1)
neutral_ratio = np.mean(btc_data['target'] == 0)
negative_ratio = np.mean(btc_data['target'] == -1)
print(f"'bb_diff' > 0 の割合: {positive_ratio:.2f}")
print(f"'bb_diff' = 0 の割合: {neutral_ratio:.2f}")
print(f"'bb_diff' < 0 の割合: {negative_ratio:.2f}")
"""

"""
# 特徴量の選択
features = [
    'close','macdhist',
    'middle_diff','band_diff','di_diff','volume_ma_diff'
]

# 特徴量の選択
features = [
    'close', 'volume', 'turnover',
    'macd', 'macdsignal', 'macdhist',
    'upper', 'lower', 'middle',
    'p_di', 'm_di', 'adx', 'adxr'
]

features = [
    'close','volume','di_diff','volume_ma',
    'volume_ma_diff','band_diff','macdhist','middle_diff'
]

features = [
    'close','volume_ma_diff','macdhist','di_diff','up_ratio'
]

"""


# CSVファイルの読み込み
file_path = 'result_upper_01.csv'  # ファイルパスを指定
btc_data = pd.read_csv(file_path)

# 目的変数の作成
btc_data = btc_data[ operator.ge(btc_data['volume'].abs(),7000) ]
ds=btc_data['volume'].describe()
print(ds)

btc_data['target'] = btc_data['bb_profit'] >= 0
ds = btc_data['bb_profit'].describe()
print(ds)
#Trueの件数, Falseの件数
print(btc_data['target'].value_counts())


print(btc_data['target'].value_counts(normalize=True))

features = [
  'close','volume','volume_ma_diff','di_diff','middle_diff','macdhist','oi'
]



# 特徴量 (X) とターゲット (y) の分離
X = btc_data[features]
y = btc_data['target']

# データセットをトレーニングセットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 特徴量の標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)


# ランダムフォレストモデルの作成
model = RandomForestClassifier(random_state=42)

# パラメータグリッドの設定
# パラメータグリッドの設定
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]  # 'auto'を削除し、'sqrt'と'log2'を使用
}

# グリッドサーチの設定
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)

# グリッドサーチの実行

#grid_search.fit(X_train_scaled, y_train)
grid_search.fit(X_train_poly, y_train)

# 最適なパラメータの表示
print("Best parameters found: ", grid_search.best_params_)

# 最適なパラメータを使用してモデルをトレーニング
best_model = grid_search.best_estimator_
#best_model.fit(X_train_scaled, y_train)
best_model.fit(X_train_poly, y_train)
# 予測と評価
#y_pred = best_model.predict(X_test_scaled)
y_pred = best_model.predict(X_test_poly)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

