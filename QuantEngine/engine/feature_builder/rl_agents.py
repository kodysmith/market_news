"""
Deep Reinforcement Learning Agents for AI Quant Trading System

Implements:
- Gym-like trading environment with risk-adjusted rewards
- DQN agent with experience replay and target networks
- PPO agent with actor-critic architecture and GAE
- Unified training/evaluation orchestrator
- Fallback tabular Q-learning when deep learning libraries unavailable
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import deque
import logging
import random
import copy

# Deep learning libraries (optional imports - try torch first, then tensorflow)
TORCH_AVAILABLE = False
TF_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    optim = None
    F = None
    Categorical = None

if not TORCH_AVAILABLE:
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers as tf_layers
        TF_AVAILABLE = True
    except ImportError:
        tf = None
        keras = None
        tf_layers = None

DL_AVAILABLE = TORCH_AVAILABLE or TF_AVAILABLE

logger = logging.getLogger(__name__)

if not DL_AVAILABLE:
    logger.warning(
        "Neither PyTorch nor TensorFlow available. "
        "Install torch or tensorflow for deep RL agents. "
        "Falling back to tabular Q-learning."
    )


# ---------------------------------------------------------------------------
# Trading Environment
# ---------------------------------------------------------------------------

class TradingEnvironment:
    """
    Gym-like trading environment for reinforcement learning agents

    Simulates a single-asset trading environment with realistic transaction
    costs, position limits, and risk-adjusted reward shaping.
    """

    def __init__(
        self,
        price_data: pd.DataFrame,
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.001,
        max_position: float = 1.0,
    ):
        """
        Initialize trading environment.

        Args:
            price_data: DataFrame with at least a 'close' column
            initial_balance: Starting cash balance
            transaction_cost: Proportional cost per trade
            max_position: Maximum position size as fraction of portfolio
        """
        self.price_data = price_data.copy().reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position

        # Pre-compute price features for observations
        self._precompute_features()

        # State variables (set on reset)
        self.current_step: int = 0
        self.balance: float = initial_balance
        self.position: float = 0.0
        self.entry_price: float = 0.0
        self.total_trades: int = 0
        self.returns_history: List[float] = []
        self.portfolio_values: List[float] = []
        self.done: bool = False

    def _precompute_features(self) -> None:
        """Pre-compute normalized price features used in observations."""
        close = self.price_data['close'].values.astype(np.float64)
        n = len(close)

        # Returns at multiple horizons
        ret_1 = np.zeros(n)
        ret_5 = np.zeros(n)
        ret_10 = np.zeros(n)
        for lag, arr in [(1, ret_1), (5, ret_5), (10, ret_10)]:
            arr[lag:] = (close[lag:] - close[:-lag]) / (close[:-lag] + 1e-10)

        # Simple moving average ratios
        def _sma_ratio(window: int) -> np.ndarray:
            sma = pd.Series(close).rolling(window, min_periods=1).mean().values
            return (close - sma) / (sma + 1e-10)

        sma_5 = _sma_ratio(5)
        sma_20 = _sma_ratio(20)

        # Volatility (rolling std of 1-period returns)
        vol_20 = pd.Series(ret_1).rolling(20, min_periods=1).std().fillna(0).values

        # RSI-like momentum indicator
        gains = np.maximum(ret_1, 0)
        losses = np.maximum(-ret_1, 0)
        avg_gain = pd.Series(gains).rolling(14, min_periods=1).mean().values
        avg_loss = pd.Series(losses).rolling(14, min_periods=1).mean().values
        rsi = avg_gain / (avg_gain + avg_loss + 1e-10)

        self._features = np.column_stack([
            ret_1, ret_5, ret_10,
            sma_5, sma_20,
            vol_20,
            rsi,
        ])

        # Feature dimension: price features + position + unrealised_pnl + balance_ratio
        self.observation_dim = self._features.shape[1] + 3

    def reset(self) -> np.ndarray:
        """
        Reset the environment to its initial state.

        Returns:
            Initial observation vector
        """
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.total_trades = 0
        self.returns_history = []
        self.portfolio_values = [self.initial_balance]
        self.done = False
        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: 0 = hold, 1 = buy, 2 = sell

        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.done:
            return self._get_observation(), 0.0, True, {}

        current_price = self.price_data['close'].iloc[self.current_step]
        prev_portfolio = self._portfolio_value(current_price)

        # Execute action
        trade_cost = 0.0
        if action == 1 and self.position <= 0:
            # Buy / close short
            trade_value = self.balance * self.max_position
            trade_cost = trade_value * self.transaction_cost
            shares = (trade_value - trade_cost) / (current_price + 1e-10)
            self.position = shares
            self.balance -= trade_value
            self.entry_price = current_price
            self.total_trades += 1
        elif action == 2 and self.position > 0:
            # Sell / close long
            trade_value = self.position * current_price
            trade_cost = trade_value * self.transaction_cost
            self.balance += trade_value - trade_cost
            self.position = 0.0
            self.entry_price = 0.0
            self.total_trades += 1

        # Advance time
        self.current_step += 1
        if self.current_step >= len(self.price_data) - 1:
            self.done = True

        new_price = self.price_data['close'].iloc[self.current_step]
        new_portfolio = self._portfolio_value(new_price)
        self.portfolio_values.append(new_portfolio)

        # Compute step return
        step_return = (new_portfolio - prev_portfolio) / (prev_portfolio + 1e-10)
        self.returns_history.append(step_return)

        # Risk-adjusted reward (Sharpe-like)
        reward = self._compute_reward(step_return)

        info = {
            'portfolio_value': new_portfolio,
            'position': self.position,
            'balance': self.balance,
            'step_return': step_return,
            'trade_cost': trade_cost,
            'total_trades': self.total_trades,
        }

        return self._get_observation(), reward, self.done, info

    def _get_observation(self) -> np.ndarray:
        """
        Construct observation vector from current state.

        Returns:
            Numpy array of normalised features, position info, and account info
        """
        price_feats = self._features[self.current_step]
        current_price = self.price_data['close'].iloc[self.current_step]

        # Position indicator normalised to [-1, 1]
        pos_value = self.position * current_price
        portfolio = self._portfolio_value(current_price)
        position_ratio = pos_value / (portfolio + 1e-10)

        # Unrealised PnL normalised
        if self.position > 0 and self.entry_price > 0:
            unrealised_pnl = (current_price - self.entry_price) / (self.entry_price + 1e-10)
        else:
            unrealised_pnl = 0.0

        # Balance ratio (fraction of initial capital currently in cash)
        balance_ratio = self.balance / (self.initial_balance + 1e-10)

        obs = np.concatenate([
            price_feats,
            np.array([position_ratio, unrealised_pnl, balance_ratio]),
        ]).astype(np.float32)

        return obs

    def _portfolio_value(self, price: float) -> float:
        """Calculate total portfolio value at a given price."""
        return self.balance + self.position * price

    def _compute_reward(self, step_return: float) -> float:
        """
        Compute risk-adjusted reward using a rolling Sharpe-like measure.

        Args:
            step_return: The portfolio return for the current step

        Returns:
            Scalar reward
        """
        # Use recent window for reward shaping
        window = min(len(self.returns_history), 20)
        if window < 2:
            return step_return * 100.0

        recent = np.array(self.returns_history[-window:])
        mean_ret = np.mean(recent)
        std_ret = np.std(recent) + 1e-10

        # Sharpe-like ratio scaled for RL
        sharpe = mean_ret / std_ret

        # Penalise drawdown
        peak = max(self.portfolio_values)
        current = self.portfolio_values[-1]
        drawdown = (peak - current) / (peak + 1e-10)

        reward = sharpe - 0.5 * drawdown
        return float(reward)


# ---------------------------------------------------------------------------
# Experience Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-size experience replay buffer for off-policy RL agents."""

    def __init__(self, capacity: int = 100000):
        self.buffer: deque = deque(maxlen=capacity)

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Tuple]:
        """
        Sample a random mini-batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            List of (state, action, reward, next_state, done) tuples
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Rollout Buffer (for PPO)
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Trajectory storage buffer for on-policy PPO training."""

    def __init__(self) -> None:
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def store(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        """Store a single timestep."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self) -> None:
        """Clear all stored data."""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def get(self) -> Dict[str, np.ndarray]:
        """Return stored trajectories as numpy arrays."""
        return {
            'states': np.array(self.states, dtype=np.float32),
            'actions': np.array(self.actions, dtype=np.int64),
            'log_probs': np.array(self.log_probs, dtype=np.float32),
            'rewards': np.array(self.rewards, dtype=np.float32),
            'values': np.array(self.values, dtype=np.float32),
            'dones': np.array(self.dones, dtype=np.float32),
        }

    def __len__(self) -> int:
        return len(self.states)


# ---------------------------------------------------------------------------
# PyTorch-based DQN Agent
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class _DQNNetwork(nn.Module):
        """3-layer MLP for DQN Q-value estimation."""

        def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
            super().__init__()
            self.fc1 = nn.Linear(state_dim, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.fc3 = nn.Linear(hidden, action_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    class DQNAgent:
        """
        Deep Q-Network agent with experience replay and target network

        Uses PyTorch backend for network construction and training.
        """

        def __init__(self, state_dim: int, action_dim: int = 3, config: Optional[Dict[str, Any]] = None):
            """
            Initialize DQN agent.

            Args:
                state_dim: Dimension of observation vector
                action_dim: Number of discrete actions (default 3: hold/buy/sell)
                config: Hyperparameter dictionary with keys lr, gamma, epsilon,
                        epsilon_min, epsilon_decay, buffer_size, batch_size,
                        target_update_freq, grad_clip
            """
            self.config = config or {}
            self.state_dim = state_dim
            self.action_dim = action_dim

            self.lr = self.config.get('lr', 1e-3)
            self.gamma = self.config.get('gamma', 0.99)
            self.epsilon = self.config.get('epsilon', 1.0)
            self.epsilon_min = self.config.get('epsilon_min', 0.01)
            self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
            self.batch_size = self.config.get('batch_size', 64)
            self.target_update_freq = self.config.get('target_update_freq', 10)
            self.grad_clip = self.config.get('grad_clip', 1.0)

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            self.policy_net = self._build_network().to(self.device)
            self.target_net = self._build_network().to(self.device)
            self.update_target_network()

            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('lr_step_size', 200),
                gamma=self.config.get('lr_gamma', 0.95),
            )

            buffer_size = self.config.get('buffer_size', 100000)
            self.replay_buffer = ReplayBuffer(capacity=buffer_size)

            self.train_steps = 0

        def _build_network(self) -> nn.Module:
            """
            Build the Q-network.

            Returns:
                A 3-layer MLP nn.Module
            """
            hidden = self.config.get('hidden_size', 128)
            return _DQNNetwork(self.state_dim, self.action_dim, hidden)

        def select_action(self, state: np.ndarray) -> int:
            """
            Select action using epsilon-greedy policy.

            Args:
                state: Current observation vector

            Returns:
                Chosen action index
            """
            if random.random() < self.epsilon:
                return random.randint(0, self.action_dim - 1)

            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t)
                return int(q_values.argmax(dim=1).item())

        def train_step(self, batch: Optional[List[Tuple]] = None) -> float:
            """
            Perform one training step on a mini-batch.

            Args:
                batch: Optional pre-sampled batch; if None, samples from buffer

            Returns:
                Scalar TD loss value
            """
            if batch is None:
                if len(self.replay_buffer) < self.batch_size:
                    return 0.0
                batch = self.replay_buffer.sample(self.batch_size)

            states, actions, rewards, next_states, dones = zip(*batch)

            states_t = torch.FloatTensor(np.array(states)).to(self.device)
            actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            rewards_t = torch.FloatTensor(rewards).to(self.device)
            next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
            dones_t = torch.FloatTensor(dones).to(self.device)

            # Current Q values
            q_values = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

            # Target Q values
            with torch.no_grad():
                next_q = self.target_net(next_states_t).max(dim=1)[0]
                target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

            loss = F.mse_loss(q_values, target_q)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
            self.optimizer.step()

            self.train_steps += 1
            if self.train_steps % self.target_update_freq == 0:
                self.update_target_network()

            return float(loss.item())

        def update_target_network(self) -> None:
            """Copy weights from policy network to target network."""
            self.target_net.load_state_dict(self.policy_net.state_dict())

        def decay_epsilon(self) -> None:
            """Decay exploration rate."""
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        def step_scheduler(self) -> None:
            """Step the learning rate scheduler."""
            self.scheduler.step()

    # -------------------------------------------------------------------
    # PyTorch-based PPO Agent
    # -------------------------------------------------------------------

    class _ActorCriticNetwork(nn.Module):
        """Shared-backbone actor-critic network for PPO."""

        def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(state_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            self.policy_head = nn.Linear(hidden, action_dim)
            self.value_head = nn.Linear(hidden, 1)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            shared = self.shared(x)
            logits = self.policy_head(shared)
            value = self.value_head(shared)
            return logits, value

    class PPOAgent:
        """
        Proximal Policy Optimization agent with actor-critic architecture

        Uses clipped surrogate objective and Generalized Advantage Estimation.
        """

        def __init__(self, state_dim: int, action_dim: int = 3, config: Optional[Dict[str, Any]] = None):
            """
            Initialize PPO agent.

            Args:
                state_dim: Dimension of observation vector
                action_dim: Number of discrete actions
                config: Hyperparameter dictionary with keys lr, gamma, gae_lambda,
                        clip_epsilon, n_epochs, entropy_coef, value_coef, grad_clip
            """
            self.config = config or {}
            self.state_dim = state_dim
            self.action_dim = action_dim

            self.lr = self.config.get('lr', 3e-4)
            self.gamma = self.config.get('gamma', 0.99)
            self.gae_lambda = self.config.get('gae_lambda', 0.95)
            self.clip_epsilon = self.config.get('clip_epsilon', 0.2)
            self.n_epochs = self.config.get('n_epochs', 4)
            self.entropy_coef = self.config.get('entropy_coef', 0.01)
            self.value_coef = self.config.get('value_coef', 0.5)
            self.grad_clip = self.config.get('grad_clip', 0.5)

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            hidden = self.config.get('hidden_size', 128)
            self.network = _ActorCriticNetwork(state_dim, action_dim, hidden).to(self.device)
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('lr_step_size', 200),
                gamma=self.config.get('lr_gamma', 0.95),
            )

            self.rollout_buffer = RolloutBuffer()

        def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
            """
            Select action from current policy.

            Args:
                state: Current observation vector

            Returns:
                Tuple of (action, log_probability, state_value)
            """
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                logits, value = self.network(state_t)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            return int(action.item()), float(log_prob.item()), float(value.item())

        def train(self, trajectories: Optional[List] = None) -> Dict[str, float]:
            """
            Run PPO update using stored rollout data.

            Args:
                trajectories: Optional external trajectory list (unused when
                              rollout_buffer is populated)

            Returns:
                Dictionary of training metrics
            """
            data = self.rollout_buffer.get()
            if len(data['states']) == 0:
                return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}

            # Compute GAE advantages
            advantages, returns = self._compute_gae(
                data['rewards'], data['values'], data['dones'],
            )

            states_t = torch.FloatTensor(data['states']).to(self.device)
            actions_t = torch.LongTensor(data['actions']).to(self.device)
            old_log_probs_t = torch.FloatTensor(data['log_probs']).to(self.device)
            advantages_t = torch.FloatTensor(advantages).to(self.device)
            returns_t = torch.FloatTensor(returns).to(self.device)

            # Normalise advantages
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy = 0.0

            for _ in range(self.n_epochs):
                logits, values = self.network(states_t)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions_t)
                entropy = dist.entropy().mean()

                # Policy loss (clipped surrogate)
                ratio = torch.exp(new_log_probs - old_log_probs_t)
                surr1 = ratio * advantages_t
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_t
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values.squeeze(-1), returns_t)

                # Combined loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
                self.optimizer.step()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.item())

            self.rollout_buffer.clear()

            n = self.n_epochs
            return {
                'policy_loss': total_policy_loss / n,
                'value_loss': total_value_loss / n,
                'entropy': total_entropy / n,
            }

        def _compute_gae(
            self,
            rewards: np.ndarray,
            values: np.ndarray,
            dones: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray]:
            """
            Compute Generalized Advantage Estimation.

            Args:
                rewards: Array of step rewards
                values: Array of value estimates
                dones: Array of episode-done flags

            Returns:
                Tuple of (advantages, returns)
            """
            n = len(rewards)
            advantages = np.zeros(n, dtype=np.float32)
            last_gae = 0.0

            for t in reversed(range(n)):
                if t == n - 1:
                    next_value = 0.0
                else:
                    next_value = values[t + 1]

                delta = rewards[t] + self.gamma * next_value * (1.0 - dones[t]) - values[t]
                last_gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * last_gae
                advantages[t] = last_gae

            returns = advantages + values
            return advantages, returns

        def decay_epsilon(self) -> None:
            """No-op for API compatibility with DQN agent."""
            pass

        def step_scheduler(self) -> None:
            """Step the learning rate scheduler."""
            self.scheduler.step()


# ---------------------------------------------------------------------------
# TensorFlow-based agents (used only when torch is unavailable)
# ---------------------------------------------------------------------------

elif TF_AVAILABLE:

    class DQNAgent:
        """
        Deep Q-Network agent using TensorFlow/Keras backend

        Provides the same interface as the PyTorch variant.
        """

        def __init__(self, state_dim: int, action_dim: int = 3, config: Optional[Dict[str, Any]] = None):
            self.config = config or {}
            self.state_dim = state_dim
            self.action_dim = action_dim

            self.lr = self.config.get('lr', 1e-3)
            self.gamma = self.config.get('gamma', 0.99)
            self.epsilon = self.config.get('epsilon', 1.0)
            self.epsilon_min = self.config.get('epsilon_min', 0.01)
            self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
            self.batch_size = self.config.get('batch_size', 64)
            self.target_update_freq = self.config.get('target_update_freq', 10)
            self.grad_clip = self.config.get('grad_clip', 1.0)

            self.policy_net = self._build_network()
            self.target_net = self._build_network()
            self.update_target_network()

            self.optimizer = keras.optimizers.Adam(learning_rate=self.lr)

            buffer_size = self.config.get('buffer_size', 100000)
            self.replay_buffer = ReplayBuffer(capacity=buffer_size)

            self.train_steps = 0
            self._lr_decay_step = self.config.get('lr_step_size', 200)
            self._lr_gamma = self.config.get('lr_gamma', 0.95)

        def _build_network(self) -> keras.Model:
            """
            Build a 3-layer MLP Keras model.

            Returns:
                Compiled Keras model
            """
            hidden = self.config.get('hidden_size', 128)
            inputs = keras.Input(shape=(self.state_dim,))
            x = tf_layers.Dense(hidden, activation='relu')(inputs)
            x = tf_layers.Dense(hidden, activation='relu')(x)
            outputs = tf_layers.Dense(self.action_dim)(x)
            return keras.Model(inputs=inputs, outputs=outputs)

        def select_action(self, state: np.ndarray) -> int:
            if random.random() < self.epsilon:
                return random.randint(0, self.action_dim - 1)
            q_values = self.policy_net(np.expand_dims(state, 0), training=False)
            return int(np.argmax(q_values.numpy()[0]))

        def train_step(self, batch: Optional[List[Tuple]] = None) -> float:
            if batch is None:
                if len(self.replay_buffer) < self.batch_size:
                    return 0.0
                batch = self.replay_buffer.sample(self.batch_size)

            states, actions, rewards, next_states, dones = zip(*batch)

            states_np = np.array(states, dtype=np.float32)
            actions_np = np.array(actions, dtype=np.int32)
            rewards_np = np.array(rewards, dtype=np.float32)
            next_states_np = np.array(next_states, dtype=np.float32)
            dones_np = np.array(dones, dtype=np.float32)

            next_q = self.target_net(next_states_np, training=False).numpy()
            target_q = rewards_np + self.gamma * np.max(next_q, axis=1) * (1.0 - dones_np)

            with tf.GradientTape() as tape:
                q_values = self.policy_net(states_np, training=True)
                action_masks = tf.one_hot(actions_np, self.action_dim)
                q_selected = tf.reduce_sum(q_values * action_masks, axis=1)
                loss = tf.reduce_mean(tf.square(q_selected - target_q))

            grads = tape.gradient(loss, self.policy_net.trainable_variables)
            grads = [tf.clip_by_norm(g, self.grad_clip) for g in grads]
            self.optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))

            self.train_steps += 1
            if self.train_steps % self.target_update_freq == 0:
                self.update_target_network()

            # Learning rate scheduling
            if self.train_steps % self._lr_decay_step == 0:
                new_lr = float(self.optimizer.learning_rate) * self._lr_gamma
                self.optimizer.learning_rate.assign(new_lr)

            return float(loss.numpy())

        def update_target_network(self) -> None:
            self.target_net.set_weights(self.policy_net.get_weights())

        def decay_epsilon(self) -> None:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        def step_scheduler(self) -> None:
            """Step the learning rate scheduler (handled internally for TF)."""
            pass

    class PPOAgent:
        """
        PPO agent using TensorFlow/Keras backend

        Provides the same interface as the PyTorch variant.
        """

        def __init__(self, state_dim: int, action_dim: int = 3, config: Optional[Dict[str, Any]] = None):
            self.config = config or {}
            self.state_dim = state_dim
            self.action_dim = action_dim

            self.lr = self.config.get('lr', 3e-4)
            self.gamma = self.config.get('gamma', 0.99)
            self.gae_lambda = self.config.get('gae_lambda', 0.95)
            self.clip_epsilon = self.config.get('clip_epsilon', 0.2)
            self.n_epochs = self.config.get('n_epochs', 4)
            self.entropy_coef = self.config.get('entropy_coef', 0.01)
            self.value_coef = self.config.get('value_coef', 0.5)
            self.grad_clip = self.config.get('grad_clip', 0.5)

            self.network = self._build_network()
            self.optimizer = keras.optimizers.Adam(learning_rate=self.lr)
            self.rollout_buffer = RolloutBuffer()

        def _build_network(self) -> keras.Model:
            hidden = self.config.get('hidden_size', 128)
            inputs = keras.Input(shape=(self.state_dim,))
            x = tf_layers.Dense(hidden, activation='relu')(inputs)
            x = tf_layers.Dense(hidden, activation='relu')(x)
            policy_logits = tf_layers.Dense(self.action_dim, name='policy')(x)
            value = tf_layers.Dense(1, name='value')(x)
            return keras.Model(inputs=inputs, outputs=[policy_logits, value])

        def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
            state_np = np.expand_dims(state, 0).astype(np.float32)
            logits, value = self.network(state_np, training=False)
            logits_np = logits.numpy()[0]
            probs = np.exp(logits_np - np.max(logits_np))
            probs = probs / probs.sum()
            action = np.random.choice(self.action_dim, p=probs)
            log_prob = float(np.log(probs[action] + 1e-10))
            return int(action), log_prob, float(value.numpy()[0, 0])

        def train(self, trajectories: Optional[List] = None) -> Dict[str, float]:
            data = self.rollout_buffer.get()
            if len(data['states']) == 0:
                return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}

            advantages, returns = self._compute_gae(
                data['rewards'], data['values'], data['dones'],
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            states_np = data['states']
            actions_np = data['actions']
            old_log_probs_np = data['log_probs']

            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy = 0.0

            for _ in range(self.n_epochs):
                with tf.GradientTape() as tape:
                    logits, values = self.network(states_np, training=True)
                    values = tf.squeeze(values, axis=-1)

                    # Policy
                    log_probs_all = tf.nn.log_softmax(logits)
                    action_masks = tf.one_hot(actions_np, self.action_dim)
                    new_log_probs = tf.reduce_sum(log_probs_all * action_masks, axis=1)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(log_probs_all) * log_probs_all, axis=1))

                    ratio = tf.exp(new_log_probs - old_log_probs_np)
                    adv_t = tf.constant(advantages, dtype=tf.float32)
                    surr1 = ratio * adv_t
                    surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv_t
                    policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

                    ret_t = tf.constant(returns, dtype=tf.float32)
                    value_loss = tf.reduce_mean(tf.square(values - ret_t))

                    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                grads = tape.gradient(loss, self.network.trainable_variables)
                grads = [tf.clip_by_norm(g, self.grad_clip) for g in grads]
                self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))

                total_policy_loss += float(policy_loss.numpy())
                total_value_loss += float(value_loss.numpy())
                total_entropy += float(entropy.numpy())

            self.rollout_buffer.clear()
            n = self.n_epochs
            return {
                'policy_loss': total_policy_loss / n,
                'value_loss': total_value_loss / n,
                'entropy': total_entropy / n,
            }

        def _compute_gae(
            self,
            rewards: np.ndarray,
            values: np.ndarray,
            dones: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray]:
            n = len(rewards)
            advantages = np.zeros(n, dtype=np.float32)
            last_gae = 0.0
            for t in reversed(range(n)):
                next_value = 0.0 if t == n - 1 else values[t + 1]
                delta = rewards[t] + self.gamma * next_value * (1.0 - dones[t]) - values[t]
                last_gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * last_gae
                advantages[t] = last_gae
            returns = advantages + values
            return advantages, returns

        def decay_epsilon(self) -> None:
            pass

        def step_scheduler(self) -> None:
            pass


# ---------------------------------------------------------------------------
# Fallback: Tabular Q-Learning Agent (no deep learning dependency)
# ---------------------------------------------------------------------------

class SimpleRLAgent:
    """
    Tabular Q-learning agent with state discretisation

    Works without any deep learning library by discretising the continuous
    state space into bins and maintaining a Q-table.
    """

    def __init__(self, state_dim: int, action_dim: int = 3, config: Optional[Dict[str, Any]] = None):
        """
        Initialize simple Q-learning agent.

        Args:
            state_dim: Dimension of observation vector
            action_dim: Number of discrete actions
            config: Hyperparameters with keys lr, gamma, epsilon, epsilon_min,
                    epsilon_decay, n_bins
        """
        self.config = config or {}
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.lr = self.config.get('lr', 0.1)
        self.gamma = self.config.get('gamma', 0.99)
        self.epsilon = self.config.get('epsilon', 1.0)
        self.epsilon_min = self.config.get('epsilon_min', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        self.n_bins = self.config.get('n_bins', 10)

        # Q-table as a dictionary mapping discretised states to action values
        self.q_table: Dict[Tuple, np.ndarray] = {}

        # Bin edges per dimension (initialised lazily from observed data)
        self.bin_edges: Optional[List[np.ndarray]] = None
        self._state_buffer: List[np.ndarray] = []
        self._warmup_steps = self.config.get('warmup_steps', 200)

        # Buffers for interface compatibility with DQN and PPO agents
        self.replay_buffer = ReplayBuffer(capacity=self.config.get('buffer_size', 100000))
        self.rollout_buffer = RolloutBuffer()

    def _discretise(self, state: np.ndarray) -> Tuple:
        """
        Convert continuous state to a discrete bin tuple.

        Args:
            state: Raw observation vector

        Returns:
            Tuple of bin indices
        """
        if self.bin_edges is None:
            # During warmup, use simple rounding
            return tuple(np.clip(np.round(state * 5).astype(int), -10, 10))

        indices = []
        for i in range(min(len(state), len(self.bin_edges))):
            idx = int(np.digitize(state[i], self.bin_edges[i]))
            indices.append(idx)
        return tuple(indices)

    def _get_q(self, state_key: Tuple) -> np.ndarray:
        """Get or initialise Q-values for a state key."""
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)
        return self.q_table[state_key]

    def select_action(self, state: np.ndarray) -> Union[int, Tuple[int, float, float]]:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current observation vector

        Returns:
            Action index (or tuple with dummy log_prob and value for PPO compat)
        """
        self._state_buffer.append(state.copy())
        if len(self._state_buffer) == self._warmup_steps and self.bin_edges is None:
            self._fit_bins()

        state_key = self._discretise(state)
        q_values = self._get_q(state_key)

        if random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            action = int(np.argmax(q_values))

        # Return tuple for PPO-compatible interface
        return action

    def train_step(self, batch: Optional[List[Tuple]] = None) -> float:
        """
        Single Q-learning update step.

        Args:
            batch: List of (state, action, reward, next_state, done) tuples.
                   If None, samples from internal replay buffer.

        Returns:
            Mean TD error
        """
        if batch is None:
            batch_size = self.config.get('batch_size', 32)
            if len(self.replay_buffer) < batch_size:
                return 0.0
            batch = self.replay_buffer.sample(batch_size)

        if len(batch) == 0:
            return 0.0

        total_error = 0.0
        for state, action, reward, next_state, done in batch:
            s_key = self._discretise(state)
            ns_key = self._discretise(next_state)

            q = self._get_q(s_key)
            nq = self._get_q(ns_key)

            target = reward + self.gamma * np.max(nq) * (1.0 - float(done))
            td_error = target - q[action]
            q[action] += self.lr * td_error
            total_error += abs(td_error)

        return total_error / len(batch)

    def train(self, trajectories: Optional[List] = None) -> Dict[str, float]:
        """
        Train from rollout buffer (PPO-compatible interface).

        Uses simple Q-learning updates on stored transitions.

        Args:
            trajectories: Unused, kept for interface compatibility

        Returns:
            Dictionary with training metrics
        """
        data = self.rollout_buffer.get()
        if len(data['states']) == 0:
            return {'td_error': 0.0}

        total_error = 0.0
        n = len(data['states'])
        for i in range(n):
            s = data['states'][i]
            a = int(data['actions'][i])
            r = data['rewards'][i]
            ns = data['states'][min(i + 1, n - 1)]
            done = data['dones'][i]

            s_key = self._discretise(s)
            ns_key = self._discretise(ns)
            q = self._get_q(s_key)
            nq = self._get_q(ns_key)

            target = r + self.gamma * np.max(nq) * (1.0 - done)
            td_error = target - q[a]
            q[a] += self.lr * td_error
            total_error += abs(td_error)

        self.rollout_buffer.clear()
        return {'td_error': total_error / max(n, 1)}

    def update_target_network(self) -> None:
        """No-op for tabular agent (interface compatibility)."""
        pass

    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def step_scheduler(self) -> None:
        """No-op for tabular agent (interface compatibility)."""
        pass

    def _fit_bins(self) -> None:
        """Fit bin edges from observed states."""
        states = np.array(self._state_buffer)
        self.bin_edges = []
        for i in range(states.shape[1]):
            col = states[:, i]
            edges = np.linspace(np.percentile(col, 1), np.percentile(col, 99), self.n_bins + 1)
            self.bin_edges.append(edges)
        logger.info("SimpleRLAgent: fitted discretisation bins from %d observations", len(states))

    def generate_position_size(self, state: np.ndarray) -> float:
        """
        Generate position size based on Q-value confidence.

        Args:
            state: Current observation vector

        Returns:
            Position size in [-1, 1] range
        """
        state_key = self._discretise(state)
        q_values = self._get_q(state_key)

        # Softmax over Q-values for confidence
        exp_q = np.exp(q_values - np.max(q_values))
        probs = exp_q / (exp_q.sum() + 1e-10)

        # Map actions to position: hold=0, buy=+1, sell=-1
        position_map = np.array([0.0, 1.0, -1.0])
        expected_position = np.dot(probs, position_map)

        # Scale by confidence (max probability)
        confidence = np.max(probs)
        return float(expected_position * confidence)


# ---------------------------------------------------------------------------
# Use DQN/PPO from deep learning backends, or fallback
# ---------------------------------------------------------------------------

if not DL_AVAILABLE:
    DQNAgent = SimpleRLAgent  # type: ignore[misc]
    PPOAgent = SimpleRLAgent  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RL Trainer (orchestrates training and evaluation)
# ---------------------------------------------------------------------------

class RLTrainer:
    """
    Orchestrates RL agent training, evaluation, and signal generation

    Supports DQN and PPO agent types with automatic backend selection.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RL trainer.

        Args:
            config: Configuration dictionary with keys:
                - agent_type: 'dqn' or 'ppo' (default 'dqn')
                - env_config: dict passed to TradingEnvironment
                - training_config: dict passed to agent constructor
        """
        self.config = config
        self.agent_type = config.get('agent_type', 'dqn')
        self.env_config = config.get('env_config', {})
        self.training_config = config.get('training_config', {})

        self.agent = None
        self.env = None
        self.training_history: List[Dict[str, Any]] = []

        logger.info(
            "RLTrainer initialised: agent_type=%s, backend=%s",
            self.agent_type,
            'torch' if TORCH_AVAILABLE else ('tensorflow' if TF_AVAILABLE else 'tabular'),
        )

    def _create_agent(self, state_dim: int) -> None:
        """Instantiate the appropriate agent."""
        if self.agent_type == 'ppo':
            self.agent = PPOAgent(state_dim, action_dim=3, config=self.training_config)
        else:
            self.agent = DQNAgent(state_dim, action_dim=3, config=self.training_config)

    def train(self, price_data: pd.DataFrame, n_episodes: int = 1000) -> Dict[str, Any]:
        """
        Train the RL agent on historical price data.

        Args:
            price_data: DataFrame with at least a 'close' column
            n_episodes: Number of training episodes

        Returns:
            Dictionary of training metrics and history
        """
        self.env = TradingEnvironment(price_data, **self.env_config)
        self._create_agent(self.env.observation_dim)

        logger.info("Starting RL training: %d episodes", n_episodes)

        episode_rewards: List[float] = []
        episode_values: List[float] = []

        for episode in range(n_episodes):
            state = self.env.reset()
            total_reward = 0.0
            done = False

            while not done:
                if self.agent_type == 'ppo':
                    result = self.agent.select_action(state)
                    if isinstance(result, tuple):
                        action, log_prob, value = result
                    else:
                        action, log_prob, value = result, 0.0, 0.0

                    next_state, reward, done, info = self.env.step(action)

                    self.agent.rollout_buffer.store(
                        state, action, log_prob, reward, value, done,
                    )
                else:
                    action = self.agent.select_action(state)
                    if isinstance(action, tuple):
                        action = action[0]
                    next_state, reward, done, info = self.env.step(action)

                    self.agent.replay_buffer.store(state, action, reward, next_state, done)
                    self.agent.train_step()

                state = next_state
                total_reward += reward

            # End-of-episode updates
            if self.agent_type == 'ppo':
                self.agent.train()

            self.agent.decay_epsilon()
            self.agent.step_scheduler()

            final_value = self.env.portfolio_values[-1]
            episode_rewards.append(total_reward)
            episode_values.append(final_value)

            ep_info = {
                'episode': episode,
                'total_reward': total_reward,
                'final_value': final_value,
                'total_trades': self.env.total_trades,
            }
            self.training_history.append(ep_info)

            if (episode + 1) % max(1, n_episodes // 10) == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                avg_value = np.mean(episode_values[-50:])
                logger.info(
                    "Episode %d/%d  avg_reward=%.4f  avg_portfolio=%.2f  epsilon=%.3f",
                    episode + 1, n_episodes, avg_reward, avg_value,
                    getattr(self.agent, 'epsilon', 0.0),
                )

        return {
            'episodes': n_episodes,
            'final_avg_reward': float(np.mean(episode_rewards[-50:])),
            'final_avg_portfolio': float(np.mean(episode_values[-50:])),
            'best_portfolio': float(np.max(episode_values)),
            'training_history': self.training_history,
        }

    def evaluate(self, price_data: pd.DataFrame, n_episodes: int = 10) -> Dict[str, Any]:
        """
        Evaluate the trained agent on (potentially unseen) price data.

        Args:
            price_data: DataFrame with at least a 'close' column
            n_episodes: Number of evaluation episodes

        Returns:
            Dictionary with Sharpe ratio, total return, max drawdown, etc.
        """
        if self.agent is None:
            logger.error("Agent not trained. Call train() first.")
            return {'error': 'Agent not trained'}

        eval_env = TradingEnvironment(price_data, **self.env_config)
        all_returns: List[float] = []
        all_drawdowns: List[float] = []
        all_trades: List[int] = []

        # Disable exploration during evaluation
        original_epsilon = getattr(self.agent, 'epsilon', 0.0)
        if hasattr(self.agent, 'epsilon'):
            self.agent.epsilon = 0.0

        for _ in range(n_episodes):
            state = eval_env.reset()
            done = False

            while not done:
                result = self.agent.select_action(state)
                action = result[0] if isinstance(result, tuple) else result
                state, _, done, _ = eval_env.step(action)

            # Compute metrics for this episode
            values = np.array(eval_env.portfolio_values)
            total_return = (values[-1] - values[0]) / (values[0] + 1e-10)
            all_returns.append(total_return)

            # Max drawdown
            peak = np.maximum.accumulate(values)
            drawdown = (peak - values) / (peak + 1e-10)
            all_drawdowns.append(float(np.max(drawdown)))
            all_trades.append(eval_env.total_trades)

        # Restore epsilon
        if hasattr(self.agent, 'epsilon'):
            self.agent.epsilon = original_epsilon

        # Annualised Sharpe (assume daily steps, ~252 trading days)
        returns_arr = np.array(all_returns)
        mean_return = float(np.mean(returns_arr))
        std_return = float(np.std(returns_arr)) + 1e-10
        sharpe = mean_return / std_return * np.sqrt(252)

        return {
            'sharpe_ratio': float(sharpe),
            'mean_return': mean_return,
            'std_return': float(np.std(returns_arr)),
            'max_drawdown': float(np.mean(all_drawdowns)),
            'avg_trades': float(np.mean(all_trades)),
            'n_episodes': n_episodes,
        }

    def generate_signals(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Use the trained agent to produce trading signals for each timestep.

        Args:
            price_data: DataFrame with at least a 'close' column

        Returns:
            DataFrame with columns 'signal' (-1, 0, 1) and 'confidence'
        """
        if self.agent is None:
            logger.error("Agent not trained. Call train() first.")
            return pd.DataFrame()

        signal_env = TradingEnvironment(price_data, **self.env_config)
        state = signal_env.reset()

        signals: List[int] = []
        confidences: List[float] = []

        # Disable exploration
        original_epsilon = getattr(self.agent, 'epsilon', 0.0)
        if hasattr(self.agent, 'epsilon'):
            self.agent.epsilon = 0.0

        done = False
        while not done:
            result = self.agent.select_action(state)
            action = result[0] if isinstance(result, tuple) else result

            # Map action to signal: 0=hold->0, 1=buy->1, 2=sell->-1
            signal_map = {0: 0, 1: 1, 2: -1}
            signals.append(signal_map[action])

            # Confidence from SimpleRLAgent or from network outputs
            if isinstance(self.agent, SimpleRLAgent):
                conf = abs(self.agent.generate_position_size(state))
            else:
                conf = 1.0  # Deep agents: full confidence when not exploring
            confidences.append(conf)

            state, _, done, _ = signal_env.step(action)

        # Pad to match price_data length (first step has no signal)
        n_missing = len(price_data) - len(signals)
        signals = [0] * n_missing + signals
        confidences = [0.0] * n_missing + confidences

        # Restore epsilon
        if hasattr(self.agent, 'epsilon'):
            self.agent.epsilon = original_epsilon

        return pd.DataFrame({
            'signal': signals,
            'confidence': confidences,
        }, index=price_data.index)


# ---------------------------------------------------------------------------
# Factory / convenience
# ---------------------------------------------------------------------------

def create_rl_agent(config: Dict[str, Any]) -> RLTrainer:
    """
    Factory function for creating an RL trainer.

    Args:
        config: Configuration dictionary (see RLTrainer.__init__)

    Returns:
        Configured RLTrainer instance
    """
    return RLTrainer(config)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_rl_agents() -> Dict[str, Any]:
    """Test RL agent training and evaluation pipeline."""

    print("Testing RL Agents")
    print(f"  Backend: {'PyTorch' if TORCH_AVAILABLE else ('TensorFlow' if TF_AVAILABLE else 'Tabular fallback')}")

    # Generate synthetic price data
    np.random.seed(42)
    n_days = 500
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    close = 100.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n_days)))
    price_data = pd.DataFrame({'close': close}, index=dates)

    # Test TradingEnvironment independently
    print("  Testing TradingEnvironment...")
    env = TradingEnvironment(price_data, initial_balance=100000)
    obs = env.reset()
    assert obs.shape[0] == env.observation_dim, "Observation dimension mismatch"

    total_r = 0.0
    for _ in range(50):
        action = random.randint(0, 2)
        obs, reward, done, info = env.step(action)
        total_r += reward
        if done:
            break
    print(f"  Environment OK: obs_dim={env.observation_dim}, steps=50, reward={total_r:.4f}")

    # Test DQN training
    print("  Testing DQN training (20 episodes)...")
    dqn_config = {
        'agent_type': 'dqn',
        'training_config': {
            'lr': 1e-3,
            'epsilon_decay': 0.99,
            'batch_size': 32,
            'buffer_size': 10000,
            'target_update_freq': 5,
        },
    }
    trainer = RLTrainer(dqn_config)
    train_results = trainer.train(price_data, n_episodes=20)
    print(f"  DQN train: avg_reward={train_results['final_avg_reward']:.4f}, "
          f"avg_portfolio={train_results['final_avg_portfolio']:.2f}")

    # Evaluate
    eval_results = trainer.evaluate(price_data, n_episodes=3)
    print(f"  DQN eval: sharpe={eval_results['sharpe_ratio']:.4f}, "
          f"return={eval_results['mean_return']:.4f}")

    # Generate signals
    signals_df = trainer.generate_signals(price_data)
    print(f"  Signals generated: {len(signals_df)} rows, "
          f"distribution={signals_df['signal'].value_counts().to_dict()}")

    # Test PPO training
    print("  Testing PPO training (20 episodes)...")
    ppo_config = {
        'agent_type': 'ppo',
        'training_config': {
            'lr': 3e-4,
            'n_epochs': 2,
            'clip_epsilon': 0.2,
        },
    }
    ppo_trainer = RLTrainer(ppo_config)
    ppo_results = ppo_trainer.train(price_data, n_episodes=20)
    print(f"  PPO train: avg_reward={ppo_results['final_avg_reward']:.4f}, "
          f"avg_portfolio={ppo_results['final_avg_portfolio']:.2f}")

    print("  All RL agent tests passed!")

    return {
        'dqn_results': train_results,
        'ppo_results': ppo_results,
        'eval_results': eval_results,
        'signals_shape': signals_df.shape,
    }


if __name__ == "__main__":
    test_rl_agents()
