import pandas as pd
from scipy.stats import chi2_contingency
import scipy.stats as stats
import os, sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Aディレクトリーのパスを取得
sys.path.append(parent_dir)

from mongodb.data_loader_mongo  import MongoDataLoader
from common.constants import *


def main():
    # ダミーデータの作成
    feature_columns = 'aroon'
    target_column = 'bb_profit'
    mongo = MongoDataLoader()
    df = mongo.load_data_from_datetime_period("2020-01-01", "2024-06-03",MARKET_DATA_ML_LOWER)
    df = df[df[target_column]!=0]

    # Extract relevant columns
    data_filtered = df[[feature_columns, 'bb_profit']]
    # Create a binary variable for bb_profit > 0
    data_filtered['bb_profit_binary'] = (data_filtered['bb_profit'] > 0).astype(int)

    # Drop rows with missing values
    data_filtered = data_filtered.dropna()

    # Define features and target
    X = data_filtered[[feature_columns]]
    y = data_filtered['bb_profit_binary']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the logistic regression model
    log_reg = LogisticRegression()
    log_reg.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = log_reg.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    print(f'Report: {report}')
    print(f'predicted values: {y_pred}')

if __name__ == '__main__':
    main()