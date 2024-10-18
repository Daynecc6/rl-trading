import gym
from gym import spaces
import numpy as np
import pandas as pd
from scipy import stats
import os
import warnings

# Disable TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO, WARNING, and ERROR logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom ops

# Suppress Gym warnings
warnings.filterwarnings("ignore", category=UserWarning, module='gym')


class TradingEnv(gym.Env):
    def __init__(self, data, lookback_window=20):
        super(TradingEnv, self).__init__()
        
        self.data = data.copy()
        self.lookback_window = lookback_window
        self.current_step = lookback_window
        
        # Action space: [position_size, stop_loss, take_profit]
        self.action_space = spaces.Box(
            low=np.array([-1, 0, 0]),
            high=np.array([1, 0.1, 0.2]),
            shape=(3,),
            dtype=np.float32
        )
        
        # Select a subset of features for the observation space
        self.feature_columns = ['Close', 'Volume', 'Returns', 'MA14', 'RSI14', 'MACD', 'ATR']
        self.num_features = len(self.feature_columns)
        
        # Flattened observation space
        total_obs_size = self.lookback_window * self.num_features + 3  # market_data + account_info
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_size,),
            dtype=np.float32
        )
        
        # Normalize the data
        self.data[self.feature_columns] = (self.data[self.feature_columns] - self.data[self.feature_columns].mean()) / self.data[self.feature_columns].std()
        
        self.market_data_mean = self.data[self.feature_columns].mean().values
        self.market_data_std = self.data[self.feature_columns].std().values
        
        self.account_info_mean = np.array([100000.0, 0.0, 0.0])
        self.account_info_std = np.array([100000.0, 1000.0, 10000.0])  # Adjust based on expected ranges
        
        self.reset()
        
    def _get_observation(self):
        # Get the last `lookback_window` rows for selected features
        last_rows = self.data[self.feature_columns].iloc[self.current_step - self.lookback_window:self.current_step]
        
        # Flatten market data
        market_data = last_rows.values.flatten().astype(np.float32)
        
        # Account information
        account_info = np.array([
            self.cash,
            self.shares_held,
            self.total_profit
        ], dtype=np.float32)
        
        # Concatenate market data and account info
        observation = np.concatenate((market_data, account_info))
        
        return observation

        
    def step(self, action):
        # Unpack action
        position_size, stop_loss, take_profit = action
        
        # Store previous portfolio value
        prev_portfolio_value = self.cash + (self.shares_held * self.data.iloc[self.current_step]['Close'])
        
        # Current price and position value
        current_price = self.data.iloc[self.current_step]['Close']
        position_value = self.shares_held * current_price
        
        # Dynamic trading costs based on volatility
        vol_factor = self.data.iloc[self.current_step]['Volatility'] / self.data['Volatility'].mean()
        base_trading_cost = 0.001  # 0.1% base cost
        trading_cost_pct = base_trading_cost * (1 + vol_factor)
        
        # Position management
        if self.shares_held != 0:
            # Check stop loss and take profit
            entry_price = self.last_trade_price
            price_change_pct = (current_price - entry_price) / entry_price
            
            if (price_change_pct <= -stop_loss or 
                price_change_pct >= take_profit):
                # Close position
                sale_value = self.shares_held * current_price
                trading_cost = sale_value * trading_cost_pct
                self.cash += sale_value - trading_cost
                self.total_trading_cost += trading_cost
                self.shares_held = 0
        
        # Execute new position
        if self.shares_held == 0:  # Only enter new position if not already in one
            portfolio_value = self.cash + position_value
            
            if position_size > 0:  # Buy
                max_shares = self.cash // current_price
                shares_to_buy = int(max_shares * position_size)
                cost = shares_to_buy * current_price
                trading_cost = cost * trading_cost_pct
                
                if cost + trading_cost <= self.cash and shares_to_buy > 0:
                    self.cash -= (cost + trading_cost)
                    self.shares_held += shares_to_buy
                    self.total_trades += 1
                    self.total_trading_cost += trading_cost
                    self.last_trade_price = current_price
            
            elif position_size < 0:  # Short (if implemented)
                # Add short selling logic here if desired
                pass
        
        # Update step and check if done
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Calculate current portfolio value and returns
        new_position_value = self.shares_held * current_price
        new_portfolio_value = self.cash + new_position_value
        
        # Update metrics
        self.total_profit = new_portfolio_value - self.initial_cash
        self.current_profit_pct = (new_portfolio_value / self.initial_cash - 1) * 100
        
        # Calculate reward
        profit_reward = (new_portfolio_value - prev_portfolio_value) / self.initial_cash
        risk_penalty = -abs(position_size) * (self.data.iloc[self.current_step]['Volatility'] / 0.02)
        cost_penalty = -(trading_cost_pct if abs(position_size) > 0 else 0)
        
        reward = (profit_reward * 100) + risk_penalty + cost_penalty
        reward = np.clip(reward, -10, 10)
        
        # Update portfolio values for Sharpe ratio and max drawdown calculations
        self.portfolio_values.append(new_portfolio_value)
        
        info = {
            'portfolio_value': new_portfolio_value,
            'position_value': new_position_value,
            'cash': self.cash,
            'shares_held': self.shares_held,
            'total_profit': self.total_profit,
            'return_percentage': self.current_profit_pct,
            'total_trades': self.total_trades,
            'total_trading_cost': self.total_trading_cost,
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'max_drawdown': self._calculate_max_drawdown(),
        }
        
        # Return the new observation
        return self._get_observation(), reward, done, info
        
    def _calculate_sharpe_ratio(self):
        if len(self.portfolio_values) < 2:
            return 0
            
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
            
        return np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        
    def _calculate_max_drawdown(self):
        if len(self.portfolio_values) < 2:
            return 0
            
        peak = self.portfolio_values[0]
        max_dd = 0
        
        for value in self.portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
            
        return max_dd
        
    def reset(self):
        self.current_step = self.lookback_window
        self.cash = 100000.0
        self.initial_cash = self.cash
        self.shares_held = 0
        self.last_trade_price = 0
        self.total_profit = 0.0
        self.current_profit_pct = 0.0
        self.total_trades = 0
        self.total_trading_cost = 0.0
        self.portfolio_values = [self.cash]
        
        return self._get_observation()