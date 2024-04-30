import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# CSVファイルの読み込み
file_path = 'test01_lower_60.csv'  # ファイルパスを指定
btc_data = pd.read_csv(file_path)

# 目的変数の作成
btc_data['target'] = btc_data['bb_diff'] > 0

# 特徴量の選択
features = [
      'close','middle_diff', 'volume_ma_diff','macdhist','di_diff','up_ratio'
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

# モデルの初期化（指定されたパラメータを使用）
model = LGBMClassifier(
    learning_rate=0.1,
    max_depth=10,
    min_child_samples=20,
    n_estimators=100,
    num_leaves=31,
    random_state=42
)

# モデルのトレーニング
model.fit(X_train_scaled, y_train)

# テストデータでの評価
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 特徴量の重要度の表示
feature_importances = model.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), feature_importances[sorted_indices], align='center')
plt.xticks(range(X_train.shape[1]), [features[i] for i in sorted_indices], rotation=90)
plt.tight_layout()
plt.show()

