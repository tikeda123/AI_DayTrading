import pandas as pd
import sys, os


from datetime import datetime
import gym
from gym import spaces
import numpy as np


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from trading_analysis_kit.simulation_strategy_context import SimulationStrategyContext


def trade_entry_action():
    pass

def trade_exit_action():
    pass

class TradingDQNEnv(gym.Env):
    def __init__(self, data, initial_balance=1000):
        super(TradingDQNEnv, self).__init__()
        self.data = data
        self.n_steps = len(data)
        self.current_step = 0
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0
        self.positions_value = 0
        self.total_profit = 0
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.positions_value = 0
        self.total_profit = 0
        return self._next_observation()

    def _next_observation(self):
        obs = self.data.iloc[self.current_step][['close','volume','rsi','sma','macdhist']].values
        return obs

    def step(self, action):
        self.current_step += 1
        current_price = self.data.iloc[self.current_step]['close']

        if action == 0:  # 売る
            if self.position > 0:
                profit = (current_price - self.buy_price) * self.position
                self.balance += current_price * self.position
                self.positions_value = 0
                self.position = 0
                self.total_profit += profit
                reward = profit
                #print(f'sell: current_price:{current_price}-buy_price:{self.buy_price}*position:{self.position}= profit: {profit}')
            else:
                reward = 0
        elif action == 1:  # 何もしない
            #print('hold')
            reward = 0
        elif action == 2:  # 買う
            if self.balance > 0:
                self.buy_price = current_price
                self.position = self.balance / current_price
                self.positions_value = self.balance
                self.balance = 0
                reward = 0
                #print(f'buy: {current_price}, position: {self.position}')
            else:
                reward = 0

        done = self.current_step >= self.n_steps - 1

        if done:
            total_return = (self.balance + self.positions_value - self.initial_balance) / self.initial_balance
            additional_reward = total_return * 100
            reward += additional_reward

        info = {}  # この行を追加
        return self._next_observation(), reward, done, info  # この行を修正

    def render(self, mode='human'):
        if mode == 'human':
            profit = self.balance + self.positions_value - self.initial_balance
            print(f'Step: {self.current_step}, Balance: {self.balance}, Positions Value: {self.positions_value}, Profit: {profit}')