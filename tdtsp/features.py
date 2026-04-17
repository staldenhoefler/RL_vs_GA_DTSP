import numpy as np

def extract_global_features(simulator):
    """
    Returns global features vector:
    - normalized current time
    - fraction of nodes visited
    - base route duration elapsed
    - current node X coordinate
    - current node Y coordinate
    """
    n = simulator.inst.coords.shape[0]
    frac_visited = simulator.visited.sum() / n
    norm_time = (simulator.time % simulator.inst.time_horizon) / simulator.inst.time_horizon
    
    curr_x, curr_y = simulator.inst.coords[simulator.current]
    
    return np.array([norm_time, frac_visited, simulator.cost, curr_x, curr_y], dtype=np.float32)

def extract_node_features(simulator):
    """
    Returns [n, num_features] matrix of node properties:
    - x, y coords
    - service time
    - visited flag
    """
    inst = simulator.inst
    n = inst.coords.shape[0]
    
    features = np.zeros((n, 4), dtype=np.float32)
    features[:, 0:2] = inst.coords
    features[:, 2] = inst.service_times
    features[:, 3] = simulator.visited.astype(np.float32)
    
    return features
