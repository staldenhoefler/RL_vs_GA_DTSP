import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time
import wandb
import yaml
from tdtsp.instances import make_city_instance
from tdtsp.simulator import TDTSPSimulator
from tdtsp.ga_eval import evaluate_population
from tdtsp.baselines.ga import GeneticAlgorithm
from tdtsp.visualization import plot_instance_dynamics, plot_tour, get_figure_image, create_edge_weight_gif

def run_ga_experiment():
    print("--- Training GA on TD-TSP (Piecewise) ---")
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '../../config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    wandb.init(project="tdtsp-ga", config=config)
    
    env_cfg = config['env']
    ga_cfg = config['ga']
    
    e_cfg = env_cfg.copy()
    seed = e_cfg.pop('seed')
    _ = e_cfg.pop('test_seed', None)
    
    instance = make_city_instance(**e_cfg, seed=seed)
    simulator = TDTSPSimulator(instance)
    
    ga = GeneticAlgorithm(pop_size=ga_cfg['pop_size'], 
                          elite_size=ga_cfg['elite_size'], 
                          mutation_rate=ga_cfg['mutation_rate'],
                          max_generations=ga_cfg['max_generations'])
    
    start_time = time.time()
    best_tour, best_cost = ga.solve(simulator, evaluate_population, use_wandb=True)
    end_time = time.time()
    
    print(f"GA Completed in {end_time - start_time:.2f} seconds.")
    print(f"Best Cost: {best_cost:.2f}")
    print(f"Best Tour: {best_tour}")
    
    # Log visuals to WandB
    print("Generating visualizations...")
    fig_dyn = plot_instance_dynamics(instance)
    fig_tour = plot_tour(instance, best_tour, title=f"GA Best Tour (Cost: {best_cost:.2f})")
    print("Generating dynamics GIF...")
    gif_path = create_edge_weight_gif(instance, "ga_network_dynamics.gif")
    
    wandb.log({
        "environment/dynamics": wandb.Image(get_figure_image(fig_dyn)),
        "tours/best_tour": wandb.Image(get_figure_image(fig_tour)),
        "environment/edge_weight_gif": wandb.Video(gif_path, fps=2, format="gif")
    })
    
    wandb.finish()

if __name__ == "__main__":
    run_ga_experiment()
