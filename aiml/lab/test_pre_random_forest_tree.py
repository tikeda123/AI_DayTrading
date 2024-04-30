import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree

# CSVファイルの読み込み
file_path = 'test01_lower_240.csv'  # ファイルパス
btc_data = pd.read_csv(file_path)

# 目的変数の作成：'bb_diff' > 0 の場合はTrue、そうでない場合はFalse
btc_data['target'] = btc_data['bb_diff'] > 0

positive_ratio = np.mean(btc_data['target'])
negative_ratio = 1 - positive_ratio
print(f"'bb_diff' > 0 の割合: {positive_ratio:.2f}")
print(f"'bb_diff' < 0 の割合: {negative_ratio:.2f}")

# 特徴量の選択
features = [
    'close', 'volume', 'turnover', 'macdhist'
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

# ランダムフォレストモデルの作成とトレーニング
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# 予測と評価
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
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

# 決定木の可視化
tree = model.estimators_[0]
plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=features, class_names=['False', 'True'], filled=True, rounded=True)
plt.show()

