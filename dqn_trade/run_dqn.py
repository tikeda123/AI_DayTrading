import pandas as pd
import sys, os
from datetime import datetime
import gym
from gym import spaces
import numpy as np

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from dqn_trade.tradeing_dqn import TradingDQNEnv
from common.constants import *
from mongodb.data_loader_mongo import MongoDataLoader


import matplotlib.pyplot as plt

# インタラクティブモードを有効にする
plt.ion()

def plot_live(x, y, y2=None, title='', ylabel='Value', legend=None):
    plt.figure(figsize=(10, 5))
    plt.clf()  # 現在の図をクリア
    plt.plot(x, y, label=legend[0] if legend else 'Reward')
    if y2 is not None:
        plt.plot(x, y2, label=legend[1] if legend else 'Action')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Step')
    if legend:
        plt.legend()
    plt.pause(0.1)  # グラフを更新するための一時停止時間を少し長くする
    plt.show()

class VisualizerCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.rewards = []
        self.actions = []
        self.steps = []

    def on_step_end(self, step, logs=None):
        logs = logs or {}
        self.rewards.append(logs.get('reward', 0))
        self.actions.append(logs.get('action', 0))
        self.steps.append(step)
        plot_live(self.steps, self.rewards, self.actions, title='Training Progress', ylabel='Value', legend=['Reward', 'Action'])

    def on_epoch_end(self, epoch, logs=None):
        # エポックの終了時にリストをクリア
        self.rewards.clear()
        self.actions.clear()
        self.steps.clear()


def main():
    db = MongoDataLoader()
    data = db.load_data_from_datetime_period("2023-01-01", "2024-06-01",coll_type=MARKET_DATA_TECH)
    data = data.drop(columns=['date'])
    #data['start_at'] = data['start_at'].astype(int) / 10**9
    data = data[['start_at', 'close','volume','rsi','sma','macdhist']]
# インデックスを日時型に設定
    data['datetime'] = pd.to_datetime(data['start_at'], unit='s')
    data.set_index('datetime', inplace=True)
   # data.drop(columns=['start_at'], inplace=True)

        # データの正規化
    #data['close'] = (data['close'] - data['close'].min()) / (data['close'].max() - data['close'].min())
    #data['volume'] = (data['volume'] - data['volume'].min()) / (data['volume'].max() - data['volume'].min())
    #data['rsi'] = data['rsi'] / 100
    #data['sma'] = (data['sma'] - data['sma'].min()) / (data['sma'].max() - data['sma'].min())


    train_data = data[:'2024-01-01 00:00:00']
    test_data = data['2024-02-01 00:00:00':'2024-03-01 00:00:00']

    print(f'Train data shape: {train_data.shape}')
    print(f'Test data shape: {test_data.shape}')

    train_env = TradingDQNEnv(train_data, initial_balance=1000)
    test_env = TradingDQNEnv(test_data, initial_balance=1000)

    model = Sequential()
    model.add(Flatten(input_shape=(1, 5)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(3, activation='linear'))

    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()

    dqn = DQNAgent(model=model, nb_actions=3, memory=memory, nb_steps_warmup=5000, target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# エージェントの設定
    #dqn = DQNAgent(model=model, nb_actions=3, memory=memory, nb_steps_warmup=5000, target_model_update=1e-2, policy=policy)
    #dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    dqn.fit(train_env, nb_steps=30000, visualize=False, verbose=2, callbacks=[EpisodeLogger()])
    #  訓練の実行
    #dqn.fit(train_env, nb_steps=30000, visualize=False, verbose=2, callbacks=[VisualizerCallback()])

    scores = dqn.test(test_env, nb_episodes=5, visualize=False)
    print(f'Average score on test data: {np.mean(scores.history["episode_reward"])}')

class EpisodeLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        self.observations = {}

    def on_epoch_end(self, epoch, logs):
        self.observations[epoch] = logs
        episode_reward = logs['episode_reward']
        loss = logs.get('loss', None)  # 'loss' キーが存在しない場合は None を返す
        if loss is not None:
            print(f'Epoch {epoch}: Avg. Reward = {episode_reward}, Loss = {loss}')
        else:
            print(f'Epoch {epoch}: Avg. Reward = {episode_reward}')


if __name__ == "__main__":
    main()