from dataclasses import dataclass
import numpy as np

@dataclass
class TDTSPInstance:
    coords: np.ndarray          # [n, 2]
    service_times: np.ndarray   # [n]
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
        service_time = self.inst.service_times[j]
        
        # Step cost is incremental route duration
        step_cost = tt + service_time
        
        self.time += step_cost
        self.cost += step_cost
        self.current = j
        self.visited[j] = True
        self.tour.append(j)

        if self.visited.all() and self.current != self.inst.depot:
            # Need to return to depot
            tt_return = self.travel_time(self.current, self.inst.depot, self.time)
            self.time += tt_return
            self.cost += tt_return
            step_cost += tt_return
            self.current = self.inst.depot
            self.tour.append(self.inst.depot)
            self.done = True
        elif self.visited.all() and self.current == self.inst.depot:
            self.done = True

        reward = -step_cost
        return self.observe(), reward, self.done, {}

    def observe(self):
        return {
            "current_node": self.current,
            "current_time": self.time,
            "visited": self.visited.copy(),
            "tour": np.array(self.tour.copy(), dtype=np.int64),
        }

    def get_action_mask(self):
        # 1 if action is valid, 0 otherwise
        mask = (~self.visited).astype(np.int8)
        if self.visited.all() and self.current != self.inst.depot:
            # Allow step back to depot, though `step` handles it automatically too.
            mask[self.inst.depot] = 1 
        return mask

    def evaluate_tour(self, perm):
        self.reset()
        for node in perm:
            if node != self.inst.depot:
                self.step(node)
        return self.cost
