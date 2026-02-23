import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Any, Dict

from utils import calculate_distance_matrix, plot_tour

class DynamicTSP(gym.Env):
    """
    A Gymnasium environment for the Dynamic Traveling Salesman Problem (DTSP).
    """
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, n_cities: int = 20, grid_size: int = 100, dynamism_rate: float = 0.1, seed: Optional[int] = None):
        super().__init__()
        self.n_cities = n_cities
        self.grid_size = grid_size
        self.dynamism_rate = dynamism_rate
        self.auto_relocate = True  # Can be disabled during training sweeps
        
        if seed is not None:
            np.random.seed(seed)
            self._np_random = np.random.default_rng(seed)
        else:
            self._np_random = np.random.default_rng()

        # Action: pick the index of the next city to visit
        self.action_space = spaces.Discrete(self.n_cities)
        
        # State observation: distance matrix, visited mask, current city
        self.observation_space = spaces.Dict(
            {
                "distances": spaces.Box(low=0, high=np.sqrt(2 * grid_size**2), shape=(n_cities, n_cities), dtype=np.float32),
                "visited": spaces.MultiBinary(n_cities),
                "current_city": spaces.Discrete(n_cities),
            }
        )

        self.coords: Optional[np.ndarray] = None
        self.distance_matrix: Optional[np.ndarray] = None
        self.visited: Optional[np.ndarray] = None
        self.current_city: int = 0
        self.tour_path: list = []
        
        # Track coordinates history for the current run
        self.history_coords = []
        
        self.fig, self.ax = None, None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Resets the environment. Will only create new random cities if coords is None
        or if options={'new_graph': True} is passed. This preserves the graph across
        episodes for RL training.
        """
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
            
        generate_new = True
        if self.coords is not None:
            generate_new = False
        if options and options.get('new_graph', False):
            generate_new = True
            
        if generate_new:
            self.coords = self._np_random.uniform(0, self.grid_size, size=(self.n_cities, 2))
            self.distance_matrix = calculate_distance_matrix(self.coords)
            
        self.visited = np.zeros(self.n_cities, dtype=np.int8)
        self.visited[0] = 1
        self.current_city = 0
        self.tour_path = [0]
        
        self.history_coords = [self.coords.copy()]

        return self._get_obs(), self._get_info()

    def _get_obs(self) -> Dict[str, Any]:
        return {
            "distances": self.distance_matrix.astype(np.float32),
            "visited": self.visited.copy(),
            "current_city": self.current_city,
        }

    def _get_info(self) -> Dict[str, Any]:
        return {"coords": self.coords.copy(), "tour": list(self.tour_path)}

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        done = False
        truncated = False
        
        # Penalize selecting an already visited city, unless it's an error state
        if self.visited[action] and action != 0:
            reward = -10.0 * self.grid_size  # Invalid action penalty
        else:
            reward = -float(self.distance_matrix[self.current_city, action])
            self.current_city = action
            if not self.visited[action]:
                self.visited[action] = 1
            self.tour_path.append(action)

        # Check for completion (all cities visited)
        if np.all(self.visited) and not done:
            # Must return to origin 0
            if self.current_city != self.tour_path[0]:
                reward -= float(self.distance_matrix[self.current_city, self.tour_path[0]])
                self.current_city = self.tour_path[0]
                self.tour_path.append(self.tour_path[0])
            done = True
            
            if self.auto_relocate:
                self.relocate_cities()

        return self._get_obs(), reward, done, truncated, self._get_info()

    def relocate_cities(self):
        """Randomly places a fraction of cities somewhere else."""
        if self.coords is None:
            return
            
        n_relocate = int(self.n_cities * self.dynamism_rate)
        if n_relocate == 0:
            return
            
        indices_to_move = self._np_random.choice(self.n_cities, size=n_relocate, replace=False)
        new_coords = self._np_random.uniform(0, self.grid_size, size=(n_relocate, 2))
        self.coords[indices_to_move] = new_coords
        self.distance_matrix = calculate_distance_matrix(self.coords)
        self.history_coords.append(self.coords.copy())

    def get_distance_matrix(self) -> np.ndarray:
        """Returns the current distance matrix."""
        if self.distance_matrix is None:
            raise ValueError("Distance matrix is not initialized. Call reset() first.")
        return self.distance_matrix

    def render(self):
        """Renders the current tour using matplotlib."""
        if self.fig is None or self.ax is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(6, 6))

        plot_tour(self.coords, self.tour_path, self.ax, title="Dynamic TSP Environment")
        self.fig.canvas.draw()
        plt.pause(0.01)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig, self.ax = None, None
