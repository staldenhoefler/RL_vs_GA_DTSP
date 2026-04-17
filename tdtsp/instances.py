import numpy as np
from .simulator import TDTSPInstance

def make_coords(n, seed=None):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 100, size=(n, 2))

def make_static_instance(n=20, depot=0, speed=10.0, seed=42):
    """Static TSP baseline: Euclidean distance / speed."""
    coords = make_coords(n, seed)
    service_times = np.zeros(n)
    
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    travel_time = dist / speed
    
    travel_tensor = np.expand_dims(travel_time, axis=0) # [1, n, n]
    
    return TDTSPInstance(
        coords=coords,
        service_times=service_times,
        travel_tensor=travel_tensor,
        time_horizon=24.0,
        dt=24.0,
        depot=depot
    )

def make_piecewise_instance(n=20, t_bins=4, time_horizon=24.0, depot=0, seed=42):
    """Piecewise-constant TD edges."""
    rng = np.random.default_rng(seed)
    coords = make_coords(n, seed)
    
    service_times = rng.uniform(0.5, 1.5, size=(n,))
    service_times[depot] = 0.0
    
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    
    dt = time_horizon / t_bins
    travel_tensor = np.zeros((t_bins, n, n))
    
    for b in range(t_bins):
        speed_factor = rng.uniform(0.5, 1.5, size=(n, n))
        # Ensure reasonable speeds bounds
        speed = 10.0 * speed_factor
        travel_tensor[b] = dist / speed
        np.fill_diagonal(travel_tensor[b], 0.0)
        
    return TDTSPInstance(
        coords=coords,
        service_times=service_times,
        travel_tensor=travel_tensor,
        time_horizon=time_horizon,
        dt=dt,
        depot=depot
    )

def make_periodic_instance(n=20, t_bins=24, time_horizon=24.0, depot=0, num_peaks=2, seed=42):
    """Smooth periodic edges (sinusoidal rush hour profile)."""
    rng = np.random.default_rng(seed)
    coords = make_coords(n, seed)
    
    service_times = rng.uniform(0.5, 1.5, size=(n,))
    service_times[depot] = 0.0
    
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    
    dt = time_horizon / t_bins
    travel_tensor = np.zeros((t_bins, n, n))
    
    times = np.linspace(0, time_horizon, t_bins, endpoint=False)
    phases = rng.uniform(0, 2*np.pi, size=(n, n))
    
    for b in range(t_bins):
        # amplitude bounded in [0.6, 1.4] vs base speed mapping
        wave = np.sin(2 * np.pi * num_peaks * (times[b] / time_horizon) + phases)
        speed_multiplier = 1.0 + 0.4 * wave
        
        speed = 10.0 * speed_multiplier
        travel_tensor[b] = dist / speed
        np.fill_diagonal(travel_tensor[b], 0.0)
        
    return TDTSPInstance(
        coords=coords,
        service_times=service_times,
        travel_tensor=travel_tensor,
        time_horizon=time_horizon,
        dt=dt,
        depot=depot
    )

def make_city_instance(n_nodes=20, t_bins=24, time_horizon=24.0, depot=0, center_x=50.0, center_y=50.0,
                       spatial_scale=15.0, morning_peak=8.0, evening_peak=17.0, peak_width=2.5,
                       dir_penalty_min=1.3, dir_penalty_max=2.0, congestion_factor=6.0, seed=42):
    """
    Simulates a realistic city layout and traffic dynamics natively.
    Nodes are clustered in the center via normal distribution.
    Morning commute: higher travel delays (weights) traveling IN to the center.
    Evening commute: higher travel delays (weights) traveling OUT of the center.
    """
    n = n_nodes
    rng = np.random.default_rng(seed)
    center = (center_x, center_y)
    
    # 1. Normal distribution for coordinates
    coords = rng.normal(loc=center[0], scale=spatial_scale, size=(n, 2))
    coords = np.clip(coords, 0, 100) # Keep within bounded box
    
    # Fix depot directly to the epicenter 
    coords[depot] = [center[0], center[1]]
    
    service_times = rng.uniform(0.5, 1.5, size=(n,))
    service_times[depot] = 0.0
    
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    
    # Calculate radial distances to measure inward/outward movement
    radial_dist = np.linalg.norm(coords - np.array(center), axis=-1)
    
    # Generates structural directionality
    # For every pair of nodes, one direction will permanently have a heavily penalized base cost
    directional_multiplier = np.ones((n, n))
    for i in range(n):
        for j in range(i+1, n):
            penalty = rng.uniform(dir_penalty_min, dir_penalty_max)
            if rng.random() > 0.5:
                directional_multiplier[i, j] = penalty
            else:
                directional_multiplier[j, i] = penalty
                
    dt = time_horizon / t_bins
    # ensure at least 1 bin for static fallback edge cases
    actual_bins = max(1, t_bins)
    travel_tensor = np.zeros((actual_bins, n, n))
    
    times = np.linspace(0, time_horizon, actual_bins, endpoint=False)
    
    # Baseline peaks around 8:00 AM and 5:00 PM (17:00)
    morning_peak = 8.0
    evening_peak = 17.0
    peak_width = 2.5 
    
    for b in range(actual_bins):
        t = times[b]
        
        # Bell curves for rush hour intensities
        morning_intensity = np.exp(-0.5 * ((t - morning_peak) / peak_width)**2)
        evening_intensity = np.exp(-0.5 * ((t - evening_peak) / peak_width)**2)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                
                speed = 10.0
                
                # Difference in radius: >0 means moving inward, <0 means moving outward
                diff_radius = radial_dist[i] - radial_dist[j]
                
                # Normalize intensity constraint
                dir_factor = diff_radius / 20.0 
                
                if dir_factor > 0:
                    # Assured traffic congestion going inward during the morning
                    speed -= congestion_factor * morning_intensity * np.clip(dir_factor, 0, 1)
                elif dir_factor < 0:
                    # Assured traffic congestion going outward during the evening
                    speed -= congestion_factor * evening_intensity * np.clip(-dir_factor, 0, 1)
                
                speed_multiplier = rng.uniform(0.85, 1.15)
                final_speed = max(1.5, speed * speed_multiplier) 
                
                # Apply baseline physics + structural directonal multiplier
                travel_tensor[b, i, j] = (dist[i, j] / final_speed) * directional_multiplier[i, j]
                
        np.fill_diagonal(travel_tensor[b], 0.0)
        
    return TDTSPInstance(
        coords=coords,
        service_times=service_times,
        travel_tensor=travel_tensor,
        time_horizon=time_horizon,
        dt=dt,
        depot=depot
    )
