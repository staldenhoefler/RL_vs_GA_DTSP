import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time
import wandb
import yaml
from tdtsp.instances import make_city_instance
from tdtsp.simulator import TDTSPSimulator
from tdtsp.gym_env import TDTSPGymEnv
from tdtsp.ga_eval import evaluate_population
from tdtsp.baselines.ga import GeneticAlgorithm
from tdtsp.baselines.rl_policy import TDTSPPolicy
from tdtsp.visualization import plot_instance_dynamics, plot_tour, get_figure_image, create_edge_weight_gif, create_tour_gif, plot_performance_comparison
import torch

def run_evaluation():
    print("=== Evaluating RL and GA on Periodic TD-TSP ===")
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '../../config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    wandb.init(project="tdtsp-eval", config=config)
    
    env_cfg = config['env']
    ga_cfg = config['ga']
    
    # Use realistic city generation with config mappings
    e_cfg = env_cfg.copy()
    test_seed = e_cfg.pop('test_seed')
    _ = e_cfg.pop('seed', None)
    
    instance = make_city_instance(**e_cfg, seed=test_seed)
    simulator = TDTSPSimulator(instance)
    
    # 1. Run GA Evaluation
    print("\n[1] GA Evaluation")
    ga = GeneticAlgorithm(pop_size=ga_cfg['pop_size'], 
                          elite_size=ga_cfg['elite_size'], 
                          mutation_rate=ga_cfg['mutation_rate'],
                          max_generations=ga_cfg['eval_max_generations'])
    t0 = time.time()
    best_tour_ga, best_cost_ga = ga.solve(simulator, evaluate_population, use_wandb=True)
    t_ga = time.time() - t0
    
    print(f"GA Test Cost: {best_cost_ga:.2f} | Time: {t_ga:.2f}s")
    
    # 2. Run RL Validation
    print("\n[2] RL Validation")
    env = TDTSPGymEnv(simulator)
    rl_cfg = config['rl']
    policy = TDTSPPolicy(node_dim=4, global_dim=5, hidden_dim=rl_cfg['hidden_dim'])
    
    model_path = os.path.join(os.path.dirname(__file__), '../../rl_policy.pth')
    if os.path.exists(model_path):
        policy.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"Loaded trained RL policy from {model_path}")
    else:
        print(f"WARNING: {model_path} not found! Evaluating an untrained random baseline policy.")

    t0 = time.time()
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = policy.get_action(obs, deterministic=True)
        # Properly unpack and evaluate both terminated and truncated signals
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    t_rl = time.time() - t0
    
    cost_rl = info['cost']
    print(f"RL Test Cost: {cost_rl:.2f} | Time: {t_rl:.2f}s")
    
    print("\n=== Summary ===")
    print(f"GA Cost: {best_cost_ga:.2f}   | RL Cost: {cost_rl:.2f}")
    print(f"GA Inference Time: {t_ga:.2f}s | RL Inference Time: {t_rl:.2f}s")
    
    # Generate and log visualizations
    print("Generating visualizations...")
    fig_dyn = plot_instance_dynamics(instance)
    fig_tour_ga = plot_tour(instance, best_tour_ga, title=f"GA Test Tour (Cost: {best_cost_ga:.2f})")
    
    # RL Constructive policy tour is in info['tour']
    rl_tour = info['tour'] if 'tour' in info else None
    fig_tour_rl = plot_tour(instance, rl_tour, title=f"RL Test Tour (Cost: {cost_rl:.2f})")
    
    # Create the edge weight dynamics GIF
    print("Generating dynamics GIF...")
    gif_path = create_edge_weight_gif(instance, "eval_network_dynamics.gif")

    print("Generating tour GIFs...")
    ga_gif_path = create_tour_gif(instance, best_tour_ga, "eval_ga_tour.gif", title="GA Tour Step-by-Step")
    rl_gif_path = None
    if rl_tour:
        rl_gif_path = create_tour_gif(instance, rl_tour, "eval_rl_tour.gif", title="RL Tour Step-by-Step")
        
    fig_comp = plot_performance_comparison(best_cost_ga, cost_rl, t_ga, t_rl)
    
    summary_table = wandb.Table(columns=["Algorithm", "Final Cost", "Inference Time (s)"])
    summary_table.add_data("GA", best_cost_ga, t_ga)
    if rl_tour:
        summary_table.add_data("RL", cost_rl, t_rl)
        
    log_dict = {
        "eval/ga_final_cost": best_cost_ga,
        "eval/rl_final_cost": cost_rl,
        "eval/ga_time": t_ga,
        "eval/rl_time": t_rl,
        "eval/summary_table": summary_table,
        "environment/dynamics": wandb.Image(get_figure_image(fig_dyn)),
        "tours/ga_tour": wandb.Image(get_figure_image(fig_tour_ga)),
        "tours/rl_tour": wandb.Image(get_figure_image(fig_tour_rl)),
        "performance/comparison": wandb.Image(get_figure_image(fig_comp)),
        "environment/edge_weight_gif": wandb.Video(gif_path, fps=2, format="gif")
    }
    
    if ga_gif_path:
        log_dict["tours/ga_tour_step_video"] = wandb.Video(ga_gif_path, fps=2, format="gif")
    if rl_gif_path:
        log_dict["tours/rl_tour_step_video"] = wandb.Video(rl_gif_path, fps=2, format="gif")
        
    wandb.log(log_dict)
    
    wandb.finish()
    print("Eval complete.")

if __name__ == "__main__":
    run_evaluation()
