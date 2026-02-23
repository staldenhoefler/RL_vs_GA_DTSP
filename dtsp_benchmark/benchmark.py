import time
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from dtsp_env import DynamicTSP
from ga_solver import GASolver
from rl_solver import QLearningSolver
from utils import plot_tour

def run_benchmark(k_rounds: int = 3, seed: int = 42):
    """
    Runs the benchmark comparing GA and RL across K dynamic rounds.
    """
    print(f"Starting DTSP Benchmark with K={k_rounds} rounds.")
    env = DynamicTSP(n_cities=15, grid_size=100, dynamism_rate=0.9, seed=seed)
    env.auto_relocate = False  # Manual control over relocation for benchmarking
    
    # Initialize the base environment configuration without automatically regenerating cities
    env.reset(options={"new_graph": True})
    
    ga_dists = []
    rl_dists = []
    ga_times = []
    rl_times = []
    
    # Store Round 0 data for plotting
    ga_hist_0 = None
    rl_hist_0 = None
    ga_route_0 = None
    rl_route_0 = None
    coords_0 = None
    
    # Store all environment coordinates and routes for animation
    all_coords = []
    ga_routes = []
    rl_routes = []
    
    print(f"{'Round':<6} | {'GA Dist':<10} | {'RL Dist':<10} | {'GA Time(s)':<12} | {'RL Time(s)':<12}")
    print("-" * 62)
    
    with open('benchmark_results.csv', mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Round", "GA_Distance", "RL_Distance", "GA_Time_s", "RL_Time_s"])
        
        for r in range(k_rounds):
            if r > 0:
                env.relocate_cities()
            
            # Save the coordinates before any solver runs, representing the state for round r
            all_coords.append(env.unwrapped.coords.copy())
                
            dist_matrix = env.get_distance_matrix()
            
            # --- GA execution ---
            ga = GASolver(pop_size=500, n_generations=1000, mutation_rate=0.01, elite_size=10, tournament_size=5)
            start_ga = time.time()
            ga_route, ga_dist, ga_hist = ga.solve(dist_matrix)
            ga_time = time.time() - start_ga
            
            # --- RL execution ---
            # Instantiate fresh to compare fair training convergence from scratch per graph
            rl = QLearningSolver(alpha=0.2, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9999, n_episodes=100000)
            start_rl = time.time()
            rl_history = rl.train(env)
            rl_route, rl_dist = rl.solve(env)
            rl_time = time.time() - start_rl
            
            # Store first round for isolated plots
            if r == 2:
                ga_hist_0 = ga_hist
                rl_hist_0 = rl_history
                ga_route_0 = ga_route
                rl_route_0 = rl_route
                coords_0 = env.unwrapped.coords.copy()
                
            # Recording summary stats
            ga_dists.append(ga_dist)
            rl_dists.append(rl_dist)
            ga_times.append(ga_time)
            rl_times.append(rl_time)
            ga_routes.append(ga_route)
            rl_routes.append(rl_route)
            
            print(f"{r:<6} | {ga_dist:<10.2f} | {rl_dist:<10.2f} | {ga_time:<12.4f} | {rl_time:<12.4f}")
            writer.writerow([r, round(ga_dist, 2), round(rl_dist, 2), round(ga_time, 4), round(rl_time, 4)])

    print("\nBenchmark completed. Results saved to 'benchmark_results.csv'.")
    
    # Save the coordinates history
    np.save('environment_states.npy', np.array(all_coords))
    print("Environment states saved to 'environment_states.npy'.")
    
    generate_plots(k_rounds, ga_dists, rl_dists, ga_hist_0, rl_hist_0, coords_0, ga_route_0, rl_route_0)
    generate_animation(k_rounds, all_coords, ga_routes, rl_routes)

def generate_plots(k_rounds, ga_dists, rl_dists, ga_hist_0, rl_hist_0, coords_0, ga_route_0, rl_route_0):
    """Generates and saves the required matplotlib charts."""
    print("Generating and saving plots...")
    rounds = np.arange(k_rounds)
    width = 0.35
    
    # 1. Bar chart: Tour distances comparison
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(rounds - width/2, ga_dists, width, label='GA')
    ax1.bar(rounds + width/2, rl_dists, width, label='RL (Q-learning)')
    ax1.set_xlabel('Dynamic Round')
    ax1.set_ylabel('Best Tour Distance')
    ax1.set_title('Tour Distance per Round: GA vs RL')
    ax1.set_xticks(rounds)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig('distance_comparison.png')
    
    # 2. Convergence curves (Using round 0 data)
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))
    ax2a.plot(ga_hist_0, color='blue')
    ax2a.set_title('GA Convergence (Round 0)')
    ax2a.set_xlabel('Generation')
    ax2a.set_ylabel('Routing Distance')
    
    ax2b.plot(rl_hist_0, color='orange')
    ax2b.set_title('RL Training Reward (Round 0)')
    ax2b.set_xlabel('Episode')
    ax2b.set_ylabel('Total Reward')
    fig2.tight_layout()
    fig2.savefig('convergence_curves.png')
    
    # 3. Visual side-by-side tour plots for Round 0
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
    plot_tour(coords_0, ga_route_0, ax3a, title=f"GA Route (Dist: {ga_dists[2]:.2f})")
    plot_tour(coords_0, rl_route_0, ax3b, title=f"RL Route (Dist: {rl_dists[2]:.2f})")
    fig3.tight_layout()
    fig3.savefig('tour_visuals.png')

    print("Plots saved ('distance_comparison.png', 'convergence_curves.png', 'tour_visuals.png').")

def generate_animation(k_rounds, all_coords, ga_routes, rl_routes):
    """Generates a GIF animation of the TSP tours across rounds."""
    print("Generating animation (may take a moment)...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    def update(frame):
        plot_tour(all_coords[frame], ga_routes[frame], ax1, title=f"GA Route (Round {frame})")
        plot_tour(all_coords[frame], rl_routes[frame], ax2, title=f"RL Route (Round {frame})")
        
    anim = animation.FuncAnimation(fig, update, frames=k_rounds, interval=1000)
    
    try:
        # PillowWriter is usually available with matplotlib for generating gifs
        anim.save('tsp_evolution.gif', writer='pillow')
        print("Animation saved as 'tsp_evolution.gif'.")
    except Exception as e:
        print(f"Failed to save animation: {e}. You may need to run 'pip install pillow'.")
    finally:
        plt.close(fig)

def run_single_animation_tracking(seed: int = 42):
    """
    Runs a single RL episode explicitly capturing every dynamic change
    to the environment that occurs mid-episode or at the end.
    """
    print("Running single episode detailed tracking...")
    env = DynamicTSP(n_cities=15, grid_size=100, dynamism_rate=0.9, seed=seed)
    
    # Enable automatic relocation to see the true dynamic nature
    env.auto_relocate = True
    
    rl = QLearningSolver(alpha=0.2, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9999, n_episodes=1000)
    
    # Train rl normally without auto-relocation shifting underneath it constantly
    # just for a bit so it isn't completely random
    print("Pre-training RL agent...")
    rl.train(env)
    
    # Now evaluate it with auto_relocate ON
    print("Evaluating and capturing mid-episode relocations...")
    env.auto_relocate = True
    env.reset(options={"new_graph": True})
    
    # We will just run the agent in the environment until done,
    # capturing the state step by step. Note that auto_relocate triggers ON DONE.
    # If the user meant "relocations DURING the run", TSP usually defines it between rounds.
    # The environment currently records every relocation in `env.history_coords`.
    # Let's run a few full cycles in the same 'evaluate' session.
    
    total_cycles = 10
    all_routes = []
    
    for cycle in range(total_cycles):
        route, _ = rl.solve(env)
        all_routes.append(route)
        # env.solve() naturally triggers env.relocate_cities() at the end of the tour
        # if auto_relocate is True. Which pushes a new coord set to history_coords.

    print(f"Captured {len(env.unwrapped.history_coords)} coordinate states.")
    
    # Generate an animation focusing JUST on the environment change over time.
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # We only plot the RL route and the coordinates to show how the cities moved
    def update(frame):
        # Frame represents the state at the Start of the evaluation cycle
        coords = env.unwrapped.history_coords[frame]
        route = all_routes[frame] if frame < len(all_routes) else all_routes[-1]
        plot_tour(coords, route, ax, title=f"Dynamic City Relocation (State {frame})")
        
    anim = animation.FuncAnimation(fig, update, frames=len(env.unwrapped.history_coords)-1, interval=800)
    
    try:
        anim.save('single_run_dynamic.gif', writer='pillow')
        print("Animation saved as 'single_run_dynamic.gif'.")
    except Exception as e:
        print(f"Failed to save animation: {e}")
    finally:
        plt.close(fig)

if __name__ == "__main__":
    import os
    # Ensure plots don't block headless execution completely, just save them.
    # plt.show() can pause the CI or agent.
    plt.switch_backend('Agg')  
    run_benchmark(k_rounds=3, seed=42)
    run_single_animation_tracking(seed=42)
