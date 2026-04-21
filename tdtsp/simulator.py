from dataclasses import dataclass
import numpy as np

@dataclass
class TDTSPInstance:
    coords: np.ndarray          # [n, 2]
    travel_tensor: np.ndarray   # [T, n, n]
    time_horizon: float
    dt: float
    depot: int = 0

class TDTSPSimulator:
    def __init__(self, instance: TDTSPInstance):
        self.inst = instance
        self.n = len(instance.coords)
        self.reset()

    def reset(self):
        self.current = self.inst.depot
        self.time = 0.0
        self.cost = 0.0
        self.visited = np.zeros(self.n, dtype=bool)
        self.visited[self.inst.depot] = True
        self.tour = [self.inst.depot]
        self.done = False
        return self.observe()

    def _time_bin(self, t):
        idx = int((t % self.inst.time_horizon) // self.inst.dt)
        return min(max(0, idx), self.inst.travel_tensor.shape[0] - 1)

    def travel_time(self, i, j, departure_time):
        b = self._time_bin(departure_time)
        return float(self.inst.travel_tensor[b, i, j])

    def step(self, j):
        if self.visited[j] and j != self.inst.depot:
            return self.observe(), 0.0, self.done, {"error": "Already visited"}

        tt = self.travel_time(self.current, j, self.time)

        step_cost = tt
        self.time += step_cost
        self.cost += step_cost
        self.current = j
        self.tour.append(j)

        if j == self.inst.depot and self.visited.all():
            # Episode terminates only when agent explicitly returns to depot
            self.done = True
        else:
            self.visited[j] = True

        reward = -step_cost

        if self.done:
            # Bonus for completing the tour
            B = 2000.0
            reward = B - self.cost

        return self.observe(), reward, self.done, {}

    def observe(self):
        return {
            "current_node": self.current,
            "current_time": self.time,
            "visited": self.visited.copy(),
            "tour": np.array(self.tour.copy(), dtype=np.int64),
        }

    def get_action_mask(self):
        mask = (~self.visited).astype(np.int8)
        # Unmask the depot only if all other nodes are visited
        if self.visited.sum() == self.n:
            mask[self.inst.depot] = 1
        else:
            mask[self.inst.depot] = 0
        return mask

    def evaluate_tour(self, perm):
        self.reset()
        for node in perm:
            if node != self.inst.depot:
                self.step(node)
        return self.cost
