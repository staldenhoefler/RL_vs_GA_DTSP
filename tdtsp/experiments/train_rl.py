import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time
import torch
import torch.optim as optim
import torch.distributions as dist
import numpy as np
import wandb
import yaml
from tdtsp.instances import make_city_instance, make_static_instance
from tdtsp.simulator import TDTSPSimulator
from tdtsp.gym_env import TDTSPGymEnv
from tdtsp.baselines.rl_policy import TDTSPPolicy
from tdtsp.visualization import plot_instance_dynamics, plot_tour, get_figure_image, create_edge_weight_gif


def normalize_observation(obs):
    """
    Applies zero-mean, unit-variance normalization to the feature vectors.
    """
    nf = obs["node_features"]
    gf = obs["global_features"]

    nf_std = np.std(nf, axis=0) + 1e-8
    gf_std = np.std(gf, axis=0) + 1e-8

    obs["node_features"] = (nf - np.mean(nf, axis=0)) / nf_std
    obs["global_features"] = (gf - np.mean(gf, axis=0)) / gf_std

    return obs

def run_rl_experiment():
    config_path = os.path.join(os.path.dirname(__file__), '../../config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    wandb.init(project="tdtsp-rl", config=config)

    env_cfg = config['env']
    rl_cfg = config['rl']

    e_cfg = env_cfg.copy()
    seed = e_cfg.pop('seed')
    test_seed = e_cfg.pop('test_seed', 99)

    num_instances = rl_cfg.get('num_instances', 1)
    envs = []
    instances = []
    
    for i in range(num_instances):
        ## inst = make_city_instance(**e_cfg, seed=seed + i)
        inst = make_static_instance(n_nodes=env_cfg['n_nodes'], seed=seed + i)
        sim = TDTSPSimulator(inst)
        instances.append(inst)
        envs.append(TDTSPGymEnv(sim))

    val_inst = make_static_instance(n_nodes=env_cfg['n_nodes'], seed=test_seed)
    val_simulator = TDTSPSimulator(val_inst)
    val_env = TDTSPGymEnv(val_simulator)

    policy = TDTSPPolicy(node_dim=4, global_dim=5, edge_dim=env_cfg['n_nodes'], hidden_dim=rl_cfg['hidden_dim'], num_layers=rl_cfg['num_layers'])
    optimizer = optim.SGD(policy.parameters(), lr=rl_cfg['lr'], weight_decay=rl_cfg.get('weight_decay', 0.0))
    print(f"Initialized policy with {sum(p.numel() for p in policy.parameters())} parameters.")
    for layer in policy.children():
        for param in layer.parameters():
            print(f"Layer {layer.__class__.__name__} parameter shape: {param.shape}")
    num_episodes = rl_cfg['num_episodes']
    batch_size = rl_cfg.get('batch_size', 16)
    entropy_coef = rl_cfg.get('entropy_coef', 0.01)
    temperature = rl_cfg.get('temperature', 1.5)

    batch_loss_list = []
    global_min_cost = float('inf')

    start_time = time.time()
    for ep in range(num_episodes):
        env_idx = np.random.randint(num_instances)
        env = envs[env_idx]
        
        obs, _ = env.reset()
        obs = normalize_observation(obs)

        done = False
        log_probs = []
        rewards = []
        entropies = []

        while not done:
            # Direct policy evaluation to extract entropy
            logits = policy(obs)
            scaled_logits = logits / temperature
            m = dist.Categorical(logits=scaled_logits)
            action = m.sample()

            log_prob = m.log_prob(action)
            entropy = m.entropy()

            action_item = action.item()
            obs, reward, terminated, truncated, info = env.step(action_item)
            obs = normalize_observation(obs)
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)

        # Return computation
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)

        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Loss formulation with entropy regularization
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)

        episode_loss = torch.stack(policy_loss).sum() - entropy_coef * torch.stack(entropies).sum()
        batch_loss_list.append(episode_loss)

        # Gradient update per batch
        if (ep + 1) % batch_size == 0 or (ep + 1) == num_episodes:
            optimizer.zero_grad()
            batch_loss_tensor = torch.stack(batch_loss_list).mean()
            batch_loss_tensor.backward()
            optimizer.step()

            wandb.log({
                "episode": ep + 1,
                "loss": batch_loss_tensor.item()
            })
            batch_loss_list = []

        # Evaluate on validation environment
        val_obs, _ = val_env.reset()
        val_obs = normalize_observation(val_obs)
        val_done = False
        val_info = {}
        
        with torch.no_grad():
            while not val_done:
                val_action, _ = policy.get_action(val_obs, deterministic=True)
                val_obs, _, val_term, val_trunc, val_info = val_env.step(val_action)
                val_obs = normalize_observation(val_obs)
                val_done = val_term or val_trunc

        val_cost = val_info['cost']

        if val_cost < global_min_cost:
            global_min_cost = val_cost
            current_tour = val_info.get('tour', [])
            if hasattr(current_tour, 'tolist'):
                current_tour = current_tour.tolist()
            print(f"[RL] New minima on validation set: {global_min_cost:.2f} | Tour list: {current_tour}")
            wandb.log({
                "new_minima/cost": global_min_cost,
                "new_minima/tour_list": str(current_tour)
            }, commit=False)

        wandb.log({
            "episode": ep + 1,
            "cost": val_cost
        })

        if (ep + 1) % 100 == 0:
            print(f"Episode {ep + 1}/{num_episodes} - Val Cost: {val_cost:.2f} - Val Path: {val_info.get('tour', [])}")
        entropy_coef *= rl_cfg.get('entropy_decay', 0.99)
        temperature *= rl_cfg.get('temperature_decay', 0.99)
        temperature = np.max([temperature, 1])
        if (ep+1) % 2000 == 0:
            entropy_coef = rl_cfg.get('entropy_coef', 0.01)
            temperature = rl_cfg.get('temperature', 1.5)
    end_time = time.time()

    model_path = os.path.join(os.path.dirname(__file__), '../../rl_policy.pth')
    torch.save(policy.state_dict(), model_path)

    last_tour = val_info['tour'] if 'val_info' in locals() and 'tour' in val_info else None
    last_cost = val_info['cost'] if 'val_info' in locals() and 'cost' in val_info else 0.0

    # For visualizations, use the fixed validation instance
    instance = val_inst

    fig_dyn = plot_instance_dynamics(instance)
    fig_tour = plot_tour(instance, last_tour, title=f"RL Final Val Tour (Cost: {last_cost:.2f})")
    gif_path = create_edge_weight_gif(instance, "rl_val_network_dynamics.gif")

    wandb.log({
        "environment/dynamics": wandb.Image(get_figure_image(fig_dyn)),
        "tours/final_tour": wandb.Image(get_figure_image(fig_tour)),
        "environment/edge_weight_gif": wandb.Video(gif_path, fps=2, format="gif")
    })

    wandb.finish()


if __name__ == "__main__":
    run_rl_experiment()