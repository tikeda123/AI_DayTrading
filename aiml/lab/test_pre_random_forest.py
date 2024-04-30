import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# CSVファイルの読み込み
file_path = 'test01_upper_240.csv'  # ファイルパス
btc_data = pd.read_csv(file_path)

# 目的変数の作成：'bb_diff' > 0 の場合はTrue、そうでない場合はFalse
btc_data['target'] = btc_data['bb_diff'] > 300

positive_ratio = np.mean(btc_data['target'])
negative_ratio = 1 - positive_ratio
print(f"'bb_diff' > 0 の割合: {positive_ratio:.2f}")
print(f"'bb_diff' < 0 の割合: {negative_ratio:.2f}")

# 特徴量の選択
"""
features = [
    'close',  'volume', 'turnover',
    'macd', 'macdsignal', 'macdhist',
    'upper', 'lower', 'middle',
    'macdhist_positive', 'macd_rising', 'macdsignal_rising', 
    'macd_positive', 'macdsignal_positive'
]
"""
features = [
    'close',  'volume', 'turnover',
    'macd', 'macdsignal', 'macdhist',
    'upper', 'lower', 'middle'
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

# 予測確率の取得
y_pred_proba = model.predict_proba(X_test_scaled)

# 確信度が高い予測のみを選択するための閾値を設定
threshold = 0.7

# 閾値を超える予測のみをフィルタリング
high_confidence_predictions = []
high_confidence_actuals = []
for i in range(len(y_pred_proba)):
    if y_pred_proba[i][1] > threshold:
        high_confidence_predictions.append(1)
        high_confidence_actuals.append(y_test.iloc[i])
    elif y_pred_proba[i][0] > threshold:
        high_confidence_predictions.append(0)
        high_confidence_actuals.append(y_test.iloc[i])

# 高確信度の予測の正解率を計算
if high_confidence_predictions:
    high_confidence_accuracy = accuracy_score(high_confidence_actuals, high_confidence_predictions)
    print(f"\n高確信度予測の正解率: {high_confidence_accuracy:.2f}")
    print(f"高確信度予測数: {len(high_confidence_predictions)}")
    print(f"全予測数: {len(y_test)}")

    # 高確信度の予測に対する分類レポートの出力
    high_confidence_report = classification_report(high_confidence_actuals, high_confidence_predictions)
    print("\n高確信度予測の分類レポート:")
    print(high_confidence_report)
else:
    print("\n指定した閾値を超える予測はありませんでした。")


"""" 
# CSVファイルからデータを再度読み込む
test_data = pd.read_csv('test01_60.csv')

# 目的変数の設定
test_data['target'] = test_data['bb_diff'] > 0
correct_count = 0
incorrect_count = 0

# 各行に対する予測と実際の値の比較
for i in range(len(test_data)):
    row = test_data.iloc[i:i+1]  # DataFrame形式で行を選択
    X_row = scaler.transform(row[features])  # 特徴量の標準化
    y_row_pred = model.predict(X_row)
    y_row_pred_proba = model.predict_proba(X_row)  # 予測確率の取得
    proba = y_row_pred_proba[0, int(y_row_pred[0])]  # 予測クラスの確率を取得
    high_confidence = proba > 0.7  # 確信度が70%以上かどうか
    high_confidence_str = "高確信度" if high_confidence else "低確信度"

    if y_row_pred[0] == row['target'].values[0]:
        correct_count += 1
        correct = "正解"
    else:
        incorrect_count += 1
        correct = "不正解"
    print(f"行 {i+1}: 予測 = {y_row_pred[0]}, 確信度 = {proba:.2f} - {high_confidence_str}, 実際 = {row['target'].values[0]} - {correct}")

# 正解と不正解のカウントの出力
print(f"\n正解数: {correct_count}")
print(f"不正解数: {incorrect_count}")
"""

