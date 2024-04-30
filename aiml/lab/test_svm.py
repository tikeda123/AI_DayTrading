import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# CSVファイルの読み込み
#file_path = 'oi_test01_upper_240.csv'  # ファイルパスを指定
file_path = '/Users/ikedatoshihiko/workspace/btctrading_wk/offline/btctrading_offline_ver2_0/data/BTCUSDT_20210101000_20230901000_60_price_lower_ml.csv'

btc_data = pd.read_csv(file_path)

# 目的変数の作成
btc_data['target'] = btc_data['bb_profit'] > 0

# 特徴量の選択

""" 
features = [
    'close',  'volume', 'turnover',
    'macd', 'macdsignal', 'macdhist',
    'upper', 'lower', 'middle'
  'close','middle_diff', 'volume_ma_diff', 'di_diff', 'macdhist',
]
features = [
    'close','middle_diff', 'volume_ma_diff','macdhist','exit_price'
]
features = [
    'close','volume_ma_diff','macdhist','di_diff'
]
features = [
    'close','volume_ma_diff','macdhist','di_diff','up_ratio'
]
"""
features = [
    'close','volume_ma_diff','oi','funding_rate'
]
# 特徴量 (X) とターゲット (y) の分離
X = btc_data[features]
y = btc_data['target']

positive_ratio = np.mean(btc_data['target'])
negative_ratio = 1 - positive_ratio
print(f"'bb_diff' > 0 の割合: {positive_ratio:.2f}")
print(f"'bb_diff' < 0 の割合: {negative_ratio:.2f}")

# データセットをトレーニングセットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 特徴量の標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#X_train_scaled = X_train
#X_test_scaled = X_test

# SVMのパラメータグリッド
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.1, 1, 10],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
}

# グリッドサーチの初期化
grid_search = GridSearchCV(
    estimator=SVC(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    verbose=2,
    n_jobs=-1
)

# グリッドサーチの実行
grid_search.fit(X_train_scaled, y_train)

# 最適なパラメータの表示
print("Best parameters found: ", grid_search.best_params_)
print("Best accuracy found: ", grid_search.best_score_)

# 最適なモデルを取得
best_model = grid_search.best_estimator_

# テストデータでの評価
y_pred = best_model.predict(X_test_scaled)
"""
 # SVMモデルの設定（最適なパラメータを使用）  {'C': 1, 'gamma': 1, 'kernel': 'poly'}
svm_model = SVC(C=1, gamma=1, kernel='poly', random_state=42)
#svm_model = SVC('C': 1, 'gamma': 1, 'kernel': 'poly')
# モデルのトレーニング
svm_model.fit(X_train_scaled, y_train)

y_pred = svm_model.predict(X_test_scaled)
"""
accuracy = accuracy_score(y_test, y_pred)
print("\nTest Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 特徴量の重要度の表示 (SVMでは使用できないため、削除またはコメントアウト)
# feature_importances = best_model.feature_importances_
# sorted_indices = np.argsort(feature_importances)[::-1]

# plt.figure(figsize=(10, 6))
# plt.title("Feature Importances")
# plt.bar(range(X_train.shape[1]), feature_importances[sorted_indices], align='center')
# plt.xticks(range(X_train.shape[1]), [features[i] for i in sorted_indices], rotation=90)
# plt.tight_layout()
# plt.show()
