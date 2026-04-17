import torch
import torch.nn as nn
import torch.nn.functional as F

class TDTSPPolicy(nn.Module):
    def __init__(self, node_dim=4, global_dim=5, hidden_dim=64):
        super().__init__()
        self.node_embed1 = nn.Linear(node_dim, hidden_dim)
        self.node_embed2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.global_embed1 = nn.Linear(global_dim, hidden_dim)
        self.global_embed2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, obs_dict):
        """
        Receives unbatched observation dictionary:
        - node_features: [n, 4]
        - global_features: [3]
        - action_mask: [n]
        """
        node_feats = torch.as_tensor(obs_dict["node_features"], dtype=torch.float32)
        global_feats = torch.as_tensor(obs_dict["global_features"], dtype=torch.float32)
        action_mask = torch.as_tensor(obs_dict["action_mask"], dtype=torch.int8)
        
        n_emb = F.relu(self.node_embed1(node_feats))
        n_emb = F.relu(self.node_embed2(n_emb))           # [n, H]
        
        g_emb = F.relu(self.global_embed1(global_feats))
        g_emb = F.relu(self.global_embed2(g_emb))         # [H]
        
        # Simple attention query mechanism
        q = self.query_proj(g_emb).unsqueeze(0)           # [1, H]
        k = self.key_proj(n_emb)                          # [n, H]
        
        # Logits distribution across nodes
        logits = torch.matmul(q, k.transpose(0, 1)).squeeze(0) # [n]
        
        # Mask visited nodes
        mask_bool = action_mask.bool()
        logits = logits.masked_fill(~mask_bool, -1e9)
        
        return logits
        
    def get_action(self, obs_dict, deterministic=False):
        logits = self.forward(obs_dict)
        probs = F.softmax(logits, dim=0)
        
        if deterministic:
            action = torch.argmax(probs).item()
            log_prob = torch.log(probs[action] + 1e-8)
        else:
            m = torch.distributions.Categorical(probs)
            action = m.sample().item()
            log_prob = m.log_prob(torch.tensor(action))
            
        return action, log_prob
