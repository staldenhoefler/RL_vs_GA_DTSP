import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(input_dim, hidden_dim, num_layers):
    """Constructs a sequential multi-layer perceptron."""
    layers = []
    current_dim = input_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.ReLU())
        current_dim = hidden_dim
    return nn.Sequential(*layers)


class TDTSPPolicy(nn.Module):
    def __init__(self, node_dim=4, global_dim=3, edge_dim=24, hidden_dim=128, num_layers=4):
        super().__init__()

        # Note: This architecture requires a fixed graph size where N = edge_dim.
        self.node_embed = _build_mlp(node_dim, hidden_dim, num_layers)
        self.global_embed = _build_mlp(global_dim, hidden_dim, num_layers)
        self.edge_embed = _build_mlp(edge_dim * edge_dim, hidden_dim, num_layers)

        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.key_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, obs_dict):
        """
        Receives batched or unbatched observation dictionary:
        - node_features: [B, N, node_dim] or [N, node_dim]
        - global_features: [B, global_dim] or [global_dim]
        - edge_features: [B, N, N] or [N, N]
        - action_mask: [B, N] or [N]
        """
        node_feats = torch.as_tensor(obs_dict["node_features"], dtype=torch.float32)
        global_feats = torch.as_tensor(obs_dict["global_features"], dtype=torch.float32)
        edge_feats = torch.as_tensor(obs_dict["edge_features"], dtype=torch.float32)
        action_mask = torch.as_tensor(obs_dict["action_mask"], dtype=torch.int8)

        unbatched = node_feats.dim() == 2
        if unbatched:
            node_feats = node_feats.unsqueeze(0)
            global_feats = global_feats.unsqueeze(0)
            edge_feats = edge_feats.unsqueeze(0)
            action_mask = action_mask.unsqueeze(0)

        # Flatten edge features per batch: [B, N, N] -> [B, N * N]
        edge_feats = edge_feats.flatten(start_dim=1)

        # Extract features
        n_emb = self.node_embed(node_feats)  # [B, N, H]
        g_emb = self.global_embed(global_feats)  # [B, H]
        e_emb = self.edge_embed(edge_feats)  # [B, H]

        # Combine structural node embeddings with global edge context.
        # e_emb is expanded to [B, 1, H] to broadcast across the N nodes.
        combined_n_emb = n_emb + e_emb.unsqueeze(1)  # [B, N, H]

        # Query and Key projections
        q = self.query_proj(g_emb).unsqueeze(1)  # [B, 1, H]
        k = self.key_proj(combined_n_emb)  # [B, N, H]

        # Batched matrix multiplication for logits
        logits = torch.bmm(q, k.transpose(1, 2)).squeeze(1)  # [B, N]

        mask_bool = action_mask.bool()
        logits = logits.masked_fill(~mask_bool, -1e9)

        if unbatched:
            logits = logits.squeeze(0)

        return logits

    def get_action(self, obs_dict, deterministic=False):
        logits = self.forward(obs_dict)

        if deterministic:
            probs = F.softmax(logits, dim=-1)
            action = torch.argmax(probs, dim=-1).item()
            log_prob = torch.log(probs[action] + 1e-8)
        else:
            m = torch.distributions.Categorical(logits=logits)
            action = m.sample()
            log_prob = m.log_prob(action)
            action = action.item()

        return action, log_prob