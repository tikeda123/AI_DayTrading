import os
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, List
from tqdm import tqdm

# Add parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common.constants import MARKET_DATA_ML_UPPER
from mongodb.data_loader_mongo import MongoDataLoader
from aiml.transformer_prediction_ts_model import TransformerPredictionTSModel

# Constants
START_DATE = "2020-01-01"
END_DATE = "2024-01-01"
NEW_DATA_START = "2024-01-01 00:00:00"
NEW_DATA_END = "2024-06-01 00:00:00"
COLLECTIONS = [
    "BTCUSDT_5_market_data_mlts_upper",
    "BTCUSDT_15_market_data_mlts_upper",
    "BTCUSDT_30_market_data_mlts_upper",
    "BTCUSDT_60_market_data_mlts_upper",
    "BTCUSDT_120_market_data_mlts_upper",
    "BTCUSDT_240_market_data_mlts_upper",
    "BTCUSDT_720_market_data_mlts_upper"
]

def setup_gpu():
    """Setup GPU if available."""
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("TensorFlow version:", tf.__version__)
    print("Is GPU available: ", tf.test.is_gpu_available())

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU is available and configured")
        except RuntimeError as e:
            print(f"Error configuring GPU: {e}")
    else:
        print("GPU is not available, using CPU")

def load_data(db: MongoDataLoader, start_date: str, end_date: str) -> pd.DataFrame:
    """Load and combine data from multiple collections."""
    df_root = pd.DataFrame()
    for collection in COLLECTIONS:
        db.set_collection(collection)
        df_next = db.load_data_from_datetime_period(start_date, end_date)
        df_root = pd.concat([df_root, df_next], ignore_index=True)
    return df_root

def train_and_evaluate(model: TransformerPredictionTSModel, db: MongoDataLoader, device: str) -> Tuple[List[float], float, str, np.ndarray]:
    """Train the model using cross-validation and evaluate it on new data."""
    df_root = load_data(db, START_DATE, END_DATE)
    x_train, x_test, y_train, y_test = model.load_and_prepare_data_time_series_mix(df_root)
    x_all = np.concatenate((x_train, x_test), axis=0)
    y_all = np.concatenate((y_train, y_test), axis=0)

    with tf.device(device):
        cv_scores = model.train_with_cross_validation(x_all, y_all)

    # Evaluate on new data
    x_train, x_test, y_train, y_test = model.load_and_prepare_data_time_series(
        NEW_DATA_START,
        NEW_DATA_END,
        MARKET_DATA_ML_UPPER,
        test_size=0.9,
        random_state=None,
        oversample=False
    )
    accuracy, report, conf_matrix = model.evaluate(x_test, y_test)
    print(f"Accuracy: {accuracy}")
    print(report)
    print(conf_matrix)

    return cv_scores, accuracy, report, conf_matrix

def main(use_gpu: bool = True):
    setup_gpu()

    model = TransformerPredictionTSModel("mix_upper_mlts")
    db = MongoDataLoader()

    device = '/GPU:0' if use_gpu and tf.test.is_gpu_available() else '/CPU:0'

    target_accuracy = 0.70
    max_iterations = 100
    patience = 5
    best_accuracy = 0
    no_improvement = 0
    best_report = ""
    best_conf_matrix = None

    for i in tqdm(range(max_iterations), desc="Training Progress"):
        print(f"\nIteration {i+1}")
        cv_scores, new_accuracy, new_report, new_conf_matrix = train_and_evaluate(model, db, device)

        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {np.mean(cv_scores)}")
        print(f"New data accuracy: {new_accuracy}")

        if new_accuracy > best_accuracy:
            best_accuracy = new_accuracy
            best_report = new_report
            best_conf_matrix = new_conf_matrix
            no_improvement = 0
            print("New best model!")
            # Save the model after cross-validation
            model.save_model()
        else:
            no_improvement += 1

        if new_accuracy >= target_accuracy:
            print(f"Target accuracy {target_accuracy} achieved. Stopping training.")
            break

        if no_improvement >= patience:
            print(f"No improvement for {patience} iterations. Stopping training.")
            break

    print("\nTraining completed.")
    print(f"Best accuracy on new data: {best_accuracy}")
    print("Final classification report on new data:")
    print(best_report)
    print("Final confusion matrix:")
    print(best_conf_matrix)

    print("test model load")
    model.load_model()
        # Evaluate on new data
    x_train, x_test, y_train, y_test = model.load_and_prepare_data_time_series(
        NEW_DATA_START,
        NEW_DATA_END,
        MARKET_DATA_ML_UPPER,
        test_size=0.9,
        random_state=None,
        oversample=False
    )

    accuracy, report, conf_matrix = model.evaluate(x_test, y_test)
    print(f"Accuracy: {accuracy}")
    print(report)
    print(conf_matrix)


if __name__ == "__main__":
    main(use_gpu=True)