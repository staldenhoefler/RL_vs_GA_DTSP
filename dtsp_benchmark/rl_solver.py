import numpy as np
import gymnasium as gym
from typing import Tuple, List, Dict

class QLearningSolver:
    """
    Q-Learning agent for the Dynamic TSP environment.
    Uses a state representation of (current_city, frozenset(visited_cities)).
    """
    def __init__(self, alpha: float = 0.1, gamma: float = 0.99, 
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01, 
                 epsilon_decay: float = 0.995, n_episodes: int = 2000):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.n_episodes = n_episodes
        
        # Q-table: (current_city, frozenset(visited_cities)) -> np.ndarray[n_actions]
        self.q_table: Dict[Tuple[int, frozenset], np.ndarray] = {}

    def _get_q_values(self, state: Tuple[int, frozenset], n_actions: int) -> np.ndarray:
        """Returns the Q-values for a given state, initializing if necessary."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(n_actions)
        return self.q_table[state]

    def _get_state(self, obs: dict) -> Tuple[int, frozenset]:
        """Converts an observation dictionary to a hashable state."""
        visited_indices = frozenset(np.where(obs["visited"] == 1)[0])
        return (int(obs["current_city"]), visited_indices)

    def train(self, env: gym.Env) -> List[float]:
        """
        Trains the Q-learning agent on the provided environment instance.
        Temporarily disables the environment's auto-relocation to allow
        the agent to learn the current spatial configuration.
        """
        n_actions = env.action_space.n
        history = []
        
        # Temporarily disable city relocation during training iterations
        original_auto_relocate = getattr(env.unwrapped, 'auto_relocate', True)
        if hasattr(env.unwrapped, 'auto_relocate'):
            env.unwrapped.auto_relocate = False

        for episode in range(self.n_episodes):
            obs, info = env.reset()
            state = self._get_state(obs)
            done = False
            total_reward = 0.0
            
            while not done:
                # Get valid unvisited actions
                valid_actions = [a for a in range(n_actions) if obs["visited"][a] == 0]
                
                if not valid_actions:
                    break  # Should be handled by env done flag, safety check

                # Epsilon-greedy action selection
                if np.random.rand() < self.epsilon:
                    action = int(np.random.choice(valid_actions))
                else:
                    q_vals = self._get_q_values(state, n_actions)
                    masked_q = np.full(n_actions, -np.inf)
                    masked_q[valid_actions] = q_vals[valid_actions]
                    action = int(np.argmax(masked_q))
                
                next_obs, reward, done, truncated, info = env.step(action)
                next_state = self._get_state(next_obs)
                
                # Q-learning temporal difference update
                q_vals = self._get_q_values(state, n_actions)
                next_q_vals = self._get_q_values(next_state, n_actions)
                
                if done:
                    max_next_q = 0.0
                else:
                    next_valid_actions = [a for a in range(n_actions) if next_obs["visited"][a] == 0]
                    if next_valid_actions:
                        max_next_q = np.max(next_q_vals[next_valid_actions])
                    else:
                        max_next_q = 0.0
                
                # Update rule
                q_vals[action] = q_vals[action] + self.alpha * (reward + self.gamma * max_next_q - q_vals[action])
                
                state = next_state
                obs = next_obs
                total_reward += reward

            history.append(total_reward)
            
            # Epsilon decay
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Restore environment state
        if hasattr(env.unwrapped, 'auto_relocate'):
            env.unwrapped.auto_relocate = original_auto_relocate
            
        return history

    def solve(self, env: gym.Env) -> Tuple[List[int], float]:
        """
        Runs a greedy policy using the learned Q-table, extracting the best route.
        """
        n_actions = env.action_space.n
        obs, info = env.reset()
        state = self._get_state(obs)
        done = False
        
        original_auto_relocate = getattr(env.unwrapped, 'auto_relocate', True)
        if hasattr(env.unwrapped, 'auto_relocate'):
            env.unwrapped.auto_relocate = False

        total_reward = 0.0
        
        while not done:
            valid_actions = [a for a in range(n_actions) if obs["visited"][a] == 0]
            if not valid_actions:
                break
                
            q_vals = self._get_q_values(state, n_actions)
            masked_q = np.full(n_actions, -np.inf)
            if valid_actions:
                masked_q[valid_actions] = q_vals[valid_actions]
            else:
                break # Default backup if no valid actions
            action = int(np.argmax(masked_q))
            
            next_obs, reward, done, truncated, info = env.step(action)
            
            state = self._get_state(next_obs)
            obs = next_obs
            total_reward += reward
            
        total_distance = -total_reward
        route = info.get("tour", [])
        
        if hasattr(env.unwrapped, 'auto_relocate'):
            env.unwrapped.auto_relocate = original_auto_relocate
            
        return route, total_distance
