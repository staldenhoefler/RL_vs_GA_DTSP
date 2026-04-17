import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time
import torch
import torch.optim as optim
import wandb
import yaml
from tdtsp.instances import make_city_instance
from tdtsp.simulator import TDTSPSimulator
from tdtsp.gym_env import TDTSPGymEnv
from tdtsp.baselines.rl_policy import TDTSPPolicy
from tdtsp.visualization import plot_instance_dynamics, plot_tour, get_figure_image, create_edge_weight_gif

def run_rl_experiment():
    print("--- Training RL on TD-TSP (Piecewise) ---")
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '../../config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    wandb.init(project="tdtsp-rl", config=config)
    
    env_cfg = config['env']
    rl_cfg = config['rl']
    
    e_cfg = env_cfg.copy()
    seed = e_cfg.pop('seed')
    _ = e_cfg.pop('test_seed', None)
    
    instance = make_city_instance(**e_cfg, seed=seed)
    simulator = TDTSPSimulator(instance)
    env = TDTSPGymEnv(simulator)
    
    policy = TDTSPPolicy(node_dim=4, global_dim=5, hidden_dim=rl_cfg['hidden_dim'])
    optimizer = optim.Adam(policy.parameters(), lr=rl_cfg['lr'])
    
    num_episodes = rl_cfg['num_episodes']
    
    start_time = time.time()
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        log_probs = []
        rewards = []
        
        while not done:
            action, log_prob = policy.get_action(obs, deterministic=False)
            obs, reward, done, _, info = env.step(action)
            
            log_probs.append(log_prob)
            rewards.append(reward)
            
        # Compute returns (REINFORCE)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + G  # gamma = 1 for episodic routing
            returns.insert(0, G)
            
        returns = torch.tensor(returns)
        # Normalize returns (baseline)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
        loss = 0
        for log_prob, R in zip(log_probs, returns):
            loss -= log_prob * R
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        wandb.log({
            "episode": ep + 1,
            "cost": info['cost'],
            "loss": loss.item()
        })
        
        if (ep + 1) % 100 == 0:
            print(f"Episode {ep+1}/{num_episodes} - Last Cost: {info['cost']:.2f}")

    end_time = time.time()
    print(f"RL Training Completed in {end_time - start_time:.2f} seconds.")
    
    # Save the model
    model_path = os.path.join(os.path.dirname(__file__), '../../rl_policy.pth')
    torch.save(policy.state_dict(), model_path)
    print(f"Saved trained policy to {model_path}")
    
    # Extract last tour for visualization
    last_tour = info['tour'] if 'tour' in info else None
    last_cost = info['cost'] if 'cost' in info else 0.0
    
    # Log visuals to WandB
    print("Generating visualizations...")
    fig_dyn = plot_instance_dynamics(instance)
    fig_tour = plot_tour(instance, last_tour, title=f"RL Final Tour (Cost: {last_cost:.2f})")
    print("Generating dynamics GIF...")
    gif_path = create_edge_weight_gif(instance, "rl_network_dynamics.gif")
    
    wandb.log({
        "environment/dynamics": wandb.Image(get_figure_image(fig_dyn)),
        "tours/final_tour": wandb.Image(get_figure_image(fig_tour)),
        "environment/edge_weight_gif": wandb.Video(gif_path, fps=2, format="gif")
    })
    
    wandb.finish()

if __name__ == "__main__":
    run_rl_experiment()
