import pandas as pd
import os
import sys
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得
sys.path.append(parent_dir)

from mongodb.data_loader_mongo import MongoDataLoader
from common.constants import *
from sklearn.utils import resample

def get_df_balanced(df, target_column):
    # bb_profit > 0とbb_profit < 0のサンプルに分割
    df_positive = df[df[target_column] > 0]
    df_negative = df[df[target_column] < 0]

    # bb_profit > 0のサンプル数を取得
    n_positive = len(df_positive)

    # bb_profit < 0のサンプルをbb_profit > 0のサンプル数と同じになるようにオーバーサンプリング
    df_negative_oversampled = resample(df_negative, replace=True, n_samples=n_positive, random_state=42)

    # オーバーサンプリングされたデータフレームを結合
    df_balanced = pd.concat([df_positive, df_negative_oversampled])

    return df_balanced

def main():
    # 特徴量のリストを定義

    #feature_columns = ['mfi','rsi','aroon','bbvi','atr','rsi']
    feature_columns =["wclprice","bbvi","rsi","ema","sma","ema_sma","aroon","adline","macd","macdhist","macdsignal","upper2","volume","atr","roc","mfi"]
    target_column = 'bb_profit'
    mongo = MongoDataLoader()
    df = mongo.load_data_from_datetime_period("2020-01-01", "2024-01-03", MARKET_DATA_ML_LOWER)
    df = df[df[target_column] != 0]
    print(df)


    df_balanced = get_df_balanced(df, target_column)
    # Extract relevant columns
    data_filtered = df_balanced[feature_columns + [target_column]]
    # Create a binary variable for bb_profit > 0
    data_filtered['bb_profit_binary'] = (data_filtered['bb_profit'] > 0).astype(int)

    # Drop rows with missing values
    data_filtered = data_filtered.dropna()

    # Define features and target
    X = data_filtered[feature_columns]
    y = data_filtered['bb_profit_binary']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the SVM model
    svm_model = SVC()
    svm_model.fit(X_train_scaled, y_train)






    # Make predictions
    y_pred = svm_model.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    print(f'Report: {report}')
    print(f'predicted values: {y_pred}')

if __name__ == '__main__':
    main()



