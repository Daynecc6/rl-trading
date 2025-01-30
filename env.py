import gym
import numpy as np
from gym import spaces


class RealisticPortfolioEnv(gym.Env):
    """
    A more realistic environment with:
      - Rolling window observations (past N days of returns & realized vol)
      - Position limits (max 30% in any single asset, sum(weights) <= 1 => no leverage)
      - Volatility targeting around 15% annual
      - More realistic transaction costs (fixed + market impact)
      - Agent does NOT see today's return before acting (uses only historical data)

    This environment is simplified but more advanced than typical basic examples.
    """

    def __init__(
        self,
        returns: np.ndarray,
        window_size=20,
        max_position=0.3,       # no more than 30% in a single asset
        target_vol=0.15,        # ~15% annualized volatility target
        base_cost=0.0005,       # 0.05% base cost
        impact_cost=0.0002,     # scaled quadratically with trade size
        trading_days_per_year=252
    ):
        """
        Args:
            returns: shape [T, n_assets]; daily returns in decimal form (e.g. 0.01 = 1%)
            window_size: number of past days to provide as observation
            max_position: maximum fraction allowed in any single asset
            target_vol: target annual volatility for position scaling
            base_cost: fixed portion of transaction cost per unit (e.g. 0.0005 = 0.05%)
            impact_cost: market impact cost scale (per trade^2)
            trading_days_per_year: ~252 for stocks
        """
        super().__init__()
        self.returns_all = returns
        self.T = len(returns)
        self.n_assets = returns.shape[1]

        self.window_size = window_size
        self.max_position = max_position
        self.target_vol = target_vol
        self.base_cost = base_cost
        self.impact_cost = impact_cost
        self.trading_days_per_year = trading_days_per_year

        # We'll track rolling realized volatility over the window
        # for each asset as an input feature, if desired.
        # (Alternatively, you could do single-asset vol or correlation, etc.)

        # Observation includes:
        #   - Past N days returns, shape [window_size, n_assets]
        #   - Past N days realized volatility (stdev of daily returns), shape [window_size, n_assets]
        obs_shape = (2 * self.window_size, self.n_assets)

        # We'll flatten the 2D observation into 1D for simplicity in SB3.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_shape[0]*obs_shape[1],),
            dtype=np.float32
        )

        # Action space = portfolio weights for each asset, in [0, 1]
        # But we must ensure sum(weights) <= 1. We enforce that in step().
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )

        # Internal state
        self.current_step = None
        self.done = None
        self.portfolio_weights = None
        self.steps_elapsed = None

    def reset(self):
        """
        Start from a random day so we can have window_size days of history behind it,
        and also enough days left to step forward. 
        We pick a random start in [window_size, T-1].
        """
        # We require at least 'window_size' days before the current day for the observation,
        # so start can be in [window_size, T-1).
        self.current_step = np.random.randint(self.window_size, self.T - 1)
        self.done = False
        self.portfolio_weights = np.zeros(self.n_assets, dtype=np.float32)
        self.steps_elapsed = 0

        return self._get_obs()

    def step(self, action):
        # 1) Enforce sum(weights) <= 1
        action = np.clip(action, 0, 1)
        sum_a = np.sum(action)
        if sum_a > 1.0:
            action = action * (1.0 / sum_a)

        # 2) Enforce per-asset max_position
        action = np.minimum(action, self.max_position)

        # 3) Compute daily return with "vol targeting"
        #    We scale the raw daily returns by ratio of target_vol / realized_vol_rolling
        #    A simpler approach is to just do dot(action, returns) but let's illustrate:
        daily_ret = self.returns_all[self.current_step]
        # Realized vol for each asset over the past window
        hist_returns = self.returns_all[self.current_step - self.window_size : self.current_step]
        realized_vol_assets = np.std(hist_returns, axis=0) + 1e-8
        annualized_vol_assets = realized_vol_assets * np.sqrt(self.trading_days_per_year)

        # scale_factor: how much to scale positions to achieve target vol
        # For example, if asset's vol is 30%, but target_vol=15%, we scale position by 0.5
        scale_factors = self.target_vol / annualized_vol_assets
        # Weighted and scaled daily return
        scaled_ret_vec = daily_ret * scale_factors
        portfolio_return_raw = float(np.dot(scaled_ret_vec, action))

        # 4) Transaction cost = base_cost * trade + impact_cost * trade^2
        #    trade = sum of absolute difference from old weights
        trade_size = np.sum(np.abs(self.portfolio_weights - action))
        transaction_cost = self.base_cost * trade_size + self.impact_cost * (trade_size**2)

        # 5) Final daily return = scaled portfolio return - transaction_cost
        #    (This is just the daily "return" from an abstract viewpoint.)
        portfolio_return = portfolio_return_raw - transaction_cost

        # 6) Convert to a reward. 
        #    We can do log(1 + portfolio_return) or just use the raw daily return.
        #    Let's do a mild log transform:
        effective_ret = max(portfolio_return, -0.999999)
        reward = np.log(1.0 + effective_ret)

        # 7) Update portfolio weights
        self.portfolio_weights = action

        # 8) Step forward
        self.current_step += 1
        self.steps_elapsed += 1
        if self.current_step >= (self.T - 1):
            self.done = True

        obs = self._get_obs() if not self.done else np.zeros(self.observation_space.shape, dtype=np.float32)

        return obs, reward, self.done, {}

    def _get_obs(self):
        """
        Return a concatenation of:
          - Past 'window_size' days of returns for each asset
          - Past 'window_size' days of realized volatility for each asset
        The agent does NOT see today's return. We only give up to day-1.
        """
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step
        # If index <0, pad with zeros
        if start_idx < 0:
            # e.g. for the first call
            pad_len = abs(start_idx)
            returns_slice = np.vstack([
                np.zeros((pad_len, self.n_assets), dtype=np.float32),
                self.returns_all[0:end_idx]
            ])
        else:
            returns_slice = self.returns_all[start_idx:end_idx]

        # shape [window_size, n_assets], possibly partially zero-padded
        realized_vol = np.array([
            np.std(returns_slice[max(0, i-5):i], axis=0) 
            for i in range(1, self.window_size+1)
        ])
        # This is a simplistic rolling volatility with a 5-day sub-window 
        # or you could do a direct formula. We'll just illustrate the concept.

        # Flatten: returns_slice => shape [window_size*n_assets]
        #          realized_vol  => shape [window_size*n_assets]
        obs = np.concatenate([returns_slice.flatten(), realized_vol.flatten()])
        return obs.astype(np.float32)
