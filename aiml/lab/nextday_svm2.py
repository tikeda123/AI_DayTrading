import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

# CSVファイルの読み込み
file_path = 'nextday01_60.csv'  # ファイルパスを指定
btc_data = pd.read_csv(file_path)


"""" 
# 特徴量の選択
features = [
  'close','volume_ma_diff','macdhist','di_diff'
]
features = [
    'close', 'up_ratio','rsi_up_ratio','oi','funding_rate'
]

features = [
    'close','macdhist','band_diff','di_diff','volume_ma_diff'
]
"""
# PCAのために目的変数 'bb_diff' を除外
# データセットをトレーニングセットとテストセットに分割
target = btc_data['islong'].astype(int)


# データの標準化
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)





# 訓練セットとテストセットへの分割
X_train, X_test, y_train, y_test = train_test_split(pca_features, target, test_size=0.3, random_state=42)

# 特徴量の標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


"""

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
best_model = grid_search.best_estimator_

"""
best_model = SVC(kernel='rbf', degree=3, gamma='scale', C=2.0, probability=False,shrinking=True, tol=1e-3)
#

best_model.fit(X_train, y_train)

# テストセットでの予測
y_pred = best_model.predict(X_test)

# モデルの性能評価
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# 混同行列と分類レポートの表示
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

