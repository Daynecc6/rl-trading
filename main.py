import matplotlib.pyplot as plt
import torch
import os
import logging
from trading_env import TradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import numpy as np
from datetime import datetime
import ta
import os
import warnings

# Disable TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO, WARNING, and ERROR logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom ops

# Suppress Gym warnings
warnings.filterwarnings("ignore", category=UserWarning, module='gym')



class TradingCallback(BaseCallback):
    def __init__(self, eval_env, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.best_reward = -np.inf
        self.evaluation_interval = 10000
        
        
    def _on_step(self):
        if self.n_calls % self.evaluation_interval == 0:
            mean_reward = self._evaluate_agent()
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.model.save(f"best_model_{mean_reward:.0f}")
        return True

    def _evaluate_agent(self):
        mean_reward, _ = evaluate_policy(self.model, self.eval_env, n_eval_episodes=5)
        self.logger.record('eval/mean_reward', mean_reward)
        return mean_reward

def prepare_data(symbol='AAPL', start='2015-01-01', end='2023-12-31'):
    try:
        data = pd.read_csv(f'{symbol}_data.csv', index_col=0, parse_dates=True)
        print(f"Loaded {symbol} data from local CSV.")
    except FileNotFoundError:
        data = yf.download(symbol, start=start, end=end)
        data.to_csv(f'{symbol}_data.csv')
        print(f"Downloaded {symbol} data and saved to local CSV.")
    
    # Ensure all required columns exist
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = data[required_columns]
    
    # Add technical indicators
    data = add_technical_indicators(data)
    
    # Calculate returns and volatility
    data['Returns'] = data['Close'].pct_change()
    data['Volatility'] = data['Returns'].rolling(window=20).std() * np.sqrt(252)  # Annualized volatility
    
    # Fit HMM
    hmm_features = data[['Returns']].dropna().values
    n_components = 4
    hmm_model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=1000, random_state=42)
    hmm_model.fit(hmm_features)
    
    # Get state probabilities
    state_probs = hmm_model.predict_proba(hmm_features)
    state_probs_df = pd.DataFrame(state_probs, columns=[f'State_{i}' for i in range(n_components)], index=data.index[1:])
    data = pd.concat([data, state_probs_df], axis=1)
    
    # Normalize data
    scaler = StandardScaler()
    cols_to_scale = data.columns.drop(['Volume', 'Returns'] + [f'State_{i}' for i in range(n_components)])
    data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])
    
    # Handle inf and NaN values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.ffill().bfill()  # Forward fill then backward fill
    
    return data

def add_technical_indicators(data):
    # Basic indicators
    data['MA14'] = ta.trend.sma_indicator(data['Close'], window=14)
    data['RSI14'] = ta.momentum.rsi(data['Close'], window=14)
    bb_indicator = ta.volatility.BollingerBands(data['Close'], window=14)
    data['BB_upper'] = bb_indicator.bollinger_hband()
    data['BB_lower'] = bb_indicator.bollinger_lband()
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_signal'] = macd.macd_signal()
    data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'])
    data['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
    
    # Additional indicators from trading_env.py
    data['BB_width'] = (data['BB_upper'] - data['BB_lower']) / data['MA14']
    data['ROC'] = data['Close'].pct_change(10)
    data['MACD_hist'] = data['MACD'] - data['MACD_signal']
    
    # Money Flow Index
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    money_flow = typical_price * data['Volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
    data['MFI'] = 100 - (100 / (1 + positive_flow / negative_flow))
    
    # ADX (Average Directional Index)
    data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'])
    
    # Ichimoku Cloud
    ichimoku = ta.trend.IchimokuIndicator(data['High'], data['Low'])
    data['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
    
    return data

def make_env(data, lookback_window, rank):
    def _init():
        env = TradingEnv(data, lookback_window)
        return env  # No need to wrap with Monitor unless you're logging
    set_random_seed(42 + rank)
    return _init


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Prepare data
    data = prepare_data(symbol='AAPL', start='2015-01-01', end='2023-12-31')
    print(f"Data shape: {data.shape}")
    print(f"Number of features: {len(data.columns)}")
    
    # Split data into train and test sets
    train_data = data.iloc[:int(len(data)*0.8)]
    test_data = data.iloc[int(len(data)*0.8):]
    
    # Create parallel environments
    n_envs = 8
    lookback_window = 20
    env_fns = [make_env(train_data, lookback_window=lookback_window, rank=i) for i in range(n_envs)]
    train_env = SubprocVecEnv(env_fns)
    # train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # Create test environment
    test_env = SubprocVecEnv([make_env(test_data, lookback_window=lookback_window, rank=0)])
    test_env = VecNormalize(test_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # Initialize model with custom policy
    import torch

    policy_kwargs = dict(
    net_arch=[128, 128],  # List of hidden layer sizes
    activation_fn=torch.nn.ReLU,
    )
    
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log="./ppo_trading_tensorboard/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64 * n_envs,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        policy_kwargs=policy_kwargs
    )

    
    # Set up callbacks
    eval_callback = EvalCallback(test_env, best_model_save_path='./logs/',
                                 log_path='./logs/', eval_freq=10000,
                                 deterministic=True, render=False)
    
    # Train the model
    try:
        model.learn(
            total_timesteps=1000000,
            callback=[eval_callback],
            progress_bar=True
        )
        print("Training complete.")
        
        # Save the final model
        model.save("final_trading_model")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    
    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Test the model
    obs = test_env.reset()
    done = [False]
    total_reward = 0
    actions = []
    rewards = []
    
    while not done[0]:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        total_reward += reward[0]
        actions.append(action[0])
        rewards.append(reward[0])
    
    print(f"Total reward: {total_reward}")
    
    # Convert actions and rewards to numpy arrays for easier plotting
    actions = np.array(actions)
    rewards = np.array(rewards)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(actions[:, 0], label='Position Size')
    plt.title('Actions - Position Size')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(actions[:, 1], label='Stop Loss')
    plt.plot(actions[:, 2], label='Take Profit')
    plt.title('Actions - Stop Loss and Take Profit')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(rewards, label='Rewards')
    plt.title('Rewards')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('trading_results.png')
    plt.show()

    # Print some statistics
    print(f"Number of trades: {len(actions)}")
    print(f"Average position size: {np.mean(actions[:, 0]):.4f}")
    print(f"Average stop loss: {np.mean(actions[:, 1]):.4f}")
    print(f"Average take profit: {np.mean(actions[:, 2]):.4f}")
    print(f"Total cumulative reward: {np.sum(rewards):.4f}")