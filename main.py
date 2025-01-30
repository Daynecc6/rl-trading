import numpy as np
import matplotlib.pyplot as plt
import optuna

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env import RealisticPortfolioEnv
from utils import fetch_data, split_data


def main():
    # 1) Fetch data
    assets = ['AAPL', 'TSLA', 'SPY', 'QQQ']
    data, returns_df = fetch_data(assets, start="2015-01-01", end="2023-12-31")

    # 2) Split train/test
    train_df, test_df = split_data(returns_df, train_end="2020-12-31")
    train_returns = train_df.to_numpy(dtype=np.float32)
    test_returns  = test_df.to_numpy(dtype=np.float32)

    print(f"Train range: {train_df.index[0]} to {train_df.index[-1]} (days={len(train_df)})")
    print(f"Test  range: {test_df.index[0]} to {test_df.index[-1]} (days={len(test_df)})")

    def objective(trial: optuna.Trial):
        """
        Optuna objective: build a PPO with trial-suggested hyperparams,
        train briefly, then measure Sharpe on test set.
        """
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        ent_coef      = trial.suggest_float("ent_coef", 0.0, 0.05)
        n_steps       = trial.suggest_categorical("n_steps", [1024, 2048])
        batch_size    = trial.suggest_categorical("batch_size", [128, 256, 512])

        # Create environment
        def make_env():
            env = RealisticPortfolioEnv(
                returns=train_returns,
                window_size=20,
                max_position=0.3,
                target_vol=0.15,
                base_cost=0.0005,
                impact_cost=0.0002
            )
            return env

        # Single environment in a DummyVecEnv
        env = DummyVecEnv([make_env])

        # PPO with these hyperparams
        model = PPO(
            "MlpPolicy",
            env,
            device="cuda",
            verbose=0,
            n_steps=n_steps,
            batch_size=batch_size,
            ent_coef=ent_coef,
            learning_rate=learning_rate,
            policy_kwargs=dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])]),
        )

        # Train for fewer timesteps to keep tuning quick
        model.learn(total_timesteps=200_000)

        # Evaluate on test set
        final_val, daily_returns, _ = test_single_pass(model, test_returns)
        sharpe = compute_sharpe(daily_returns)

        env.close()
        return sharpe

    # 3) Run hyperparam search
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3)
    print("\n=== Hyperparam Search Complete ===")
    print("Best value (Sharpe):", study.best_value)
    print("Best params:", study.best_params)

    # 4) Now final training with best hyperparams
    best_params = study.best_params
    final_env = DummyVecEnv([lambda: RealisticPortfolioEnv(
        returns=train_returns,
        window_size=20,
        max_position=0.3,
        target_vol=0.15,
        base_cost=0.0005,
        impact_cost=0.0002
    )])
    model = PPO(
        "MlpPolicy",
        final_env,
        device="cuda",
        verbose=1,
        n_steps=best_params["n_steps"],
        batch_size=best_params["batch_size"],
        ent_coef=best_params["ent_coef"],
        learning_rate=best_params["learning_rate"],
        policy_kwargs=dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])]),
        tensorboard_log="./tensorboard_logs/",
    )
    model.learn(total_timesteps=500_000)
    final_env.close()

    # 5) Final test
    final_value, daily_returns, values_series = test_single_pass(model, test_returns)
    plt.plot(values_series, label="RL (PPO) OOS Value")
    plt.title("Out-of-Sample Portfolio Value")
    plt.xlabel("Test Days")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    sharpe = compute_sharpe(daily_returns)
    print(f"Final Value: {final_value:,.2f}")
    print(f"Avg Daily Return: {100*np.mean(daily_returns):.3f}%")
    print(f"Sharpe Ratio: {sharpe:.3f}")


def compute_sharpe(daily_returns):
    avg_r = np.mean(daily_returns)
    std_r = np.std(daily_returns) + 1e-8
    return avg_r / std_r


def test_single_pass(model, test_returns, window_size=20, max_position=0.3,
                     target_vol=0.15, base_cost=0.0005, impact_cost=0.0002):
    """
    Single run from day=window_size to end of test_returns.
    Similar to your existing approach.
    """
    initial_capital = 1_000_000
    n_assets = test_returns.shape[1]
    portfolio_value = initial_capital
    portfolio_weights = np.zeros(n_assets, dtype=np.float32)
    daily_rets_list = []
    values_series = [portfolio_value]

    current_step = window_size

    def get_obs(step):
        slice_returns = test_returns[step - window_size: step]
        realized_vol = np.array([
            np.std(slice_returns[max(0, i-5):i], axis=0)
            for i in range(1, window_size+1)
        ])
        obs = np.concatenate([slice_returns.flatten(), realized_vol.flatten()])
        return obs.astype(np.float32)

    obs = get_obs(current_step)

    while current_step < len(test_returns) - 1:
        action, _ = model.predict(obs, deterministic=True)

        # sum(weights)<=1
        action = np.clip(action, 0, 1)
        sum_a = np.sum(action)
        if sum_a > 1.0:
            action *= (1.0 / sum_a)
        action = np.minimum(action, max_position)

        daily_ret_vec = test_returns[current_step]
        hist_returns = test_returns[current_step - window_size : current_step]
        realized_vol_assets = np.std(hist_returns, axis=0) + 1e-8
        annualized_vol_assets = realized_vol_assets * np.sqrt(252)
        scale_factors = target_vol / annualized_vol_assets

        portfolio_return_raw = float(np.dot(daily_ret_vec * scale_factors, action))

        trade_size = np.sum(np.abs(portfolio_weights - action))
        tcost = base_cost * trade_size + impact_cost * (trade_size**2)

        day_ret = portfolio_return_raw - tcost

        portfolio_value *= (1.0 + day_ret)
        daily_rets_list.append(day_ret)
        values_series.append(portfolio_value)

        portfolio_weights = action

        current_step += 1
        if current_step >= len(test_returns):
            break
        obs = get_obs(current_step)

    return portfolio_value, daily_rets_list, values_series


if __name__ == "__main__"w:
    main()
