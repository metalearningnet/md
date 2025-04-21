import torch
import torch.nn as nn
from utils import info
import torch.nn.functional as F
from titans import MemoryAsContextTransformer
from torch.distributions import Normal, kl_divergence, Categorical

class SkillMemory(nn.Module):
    def __init__(self,
                 num_tokens: int = 4,
                 action_dim: int = 64,
                 hidden_dim: int = 32,
                 mac_persistent_mem_tokens: int = 64,
                 mac_longterm_mem_tokens: int = 64,
                 mac_depth: int = 4,
                 mac_segment_len: int = 32,
                 mi_coeff: float = 1.0,
                 entropy_coeff: float = 0.1,
                 adv_coeff: float = 0.5,
                 kl_coeff: float = 0.01):
        super().__init__()
        info(f"Skill memory (action_dim: {action_dim} hidden_dim: {hidden_dim})")
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # ========== Memory-Augmented Context Transformer ==========
        self.mac = MemoryAsContextTransformer(
            num_tokens=num_tokens,
            dim=hidden_dim,
            depth=mac_depth,
            segment_len=mac_segment_len,
            num_persist_mem_tokens=mac_persistent_mem_tokens,
            num_longterm_mem_tokens=mac_longterm_mem_tokens,
            neural_memory_segment_len=mac_segment_len,
            token_emb=None
        )
        self.mac_output_proj = nn.Linear(num_tokens, hidden_dim)
        self.memory_var = nn.Parameter(torch.zeros(1, hidden_dim))

        # ========== Policy Network ==========
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # ========== Skill-Conditioned Prior ==========
        self.prior_net = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.prior_mean = nn.Linear(hidden_dim, hidden_dim)
        self.prior_std = nn.Linear(hidden_dim, hidden_dim)

        # ========== MD Objective Components ==========
        self.disc_gru = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True)
        self.disc_linear = nn.Linear(hidden_dim, 1)

        self.mmi_discriminator = nn.Sequential(
            nn.Linear(action_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

        # ========== Loss Coefficients ==========
        self.mi_coeff = mi_coeff
        self.entropy_coeff = entropy_coeff
        self.adv_coeff = adv_coeff
        self.kl_coeff = kl_coeff

    def get_action_logits(self, states, m):
        policy_input = torch.cat([states, m], dim=-1)
        action_logits = self.policy(policy_input)
        return action_logits, m

    def forward(self, states):        
        # ===== Process Through MAC =====
        mac_output = self.mac(states)
        m = self.mac_output_proj(mac_output)

        # ===== Skill-Conditioned Prior =====
        prior_out, _ = self.prior_net(m)
        prior_mean = self.prior_mean(prior_out)
        prior_std = F.softplus(self.prior_std(prior_out)) + 1e-4

        # ===== Memory Distribution =====
        mem_std = F.softplus(self.memory_var) + 1e-4
        mem_dist = Normal(m, mem_std.expand_as(m))
        
        action_logits, m = self.get_action_logits(states, m)
        return {
            'm': m,
            'states': states,
            'mem_dist': mem_dist,
            'action_logits': action_logits,
            'prior_dist': Normal(prior_mean, prior_std)
        }

    def compute_losses(self, states):
        outputs = self.forward(states)
        action_logits = outputs['action_logits']
        prior_dist = outputs['prior_dist']
        mem_dist = outputs['mem_dist']
        states = outputs['states']
        m = outputs['m']
        
        # ===== I(S;M) =====
        pos_pairs = torch.cat([states, m], dim=-1)
        neg_m = m[torch.randperm(m.size(0))]
        neg_pairs = torch.cat([states, neg_m], dim=-1)
        pos_scores = self.disc_linear(self.disc_gru(pos_pairs)[0][:, -1])
        neg_scores = self.disc_linear(self.disc_gru(neg_pairs)[0][:, -1])
        mi_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores)) + \
                  F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))

        # ===== Action Entropy =====s
        entropy = Categorical(logits=action_logits).entropy().mean()
        
        # ===== Adversarial Loss =====s
        real_logits = self.mmi_discriminator(torch.cat([action_logits, m.detach()], dim=-1))
        fake_logits = self.mmi_discriminator(torch.cat([action_logits.detach(), m], dim=-1))
        adv_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits)) + \
                   F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))

        # ===== KL Regularization =====
        kl_loss = kl_divergence(mem_dist, prior_dist).mean()

        total_loss = (
            self.mi_coeff * mi_loss +
            self.entropy_coeff * -entropy +
            self.adv_coeff * adv_loss +
            self.kl_coeff * kl_loss
        )
        
        return {
            'total_loss': total_loss,
            'loss_components': {
                'mi_loss': mi_loss.item(),
                'entropy': entropy.item(),
                'adv_loss': adv_loss.item(),
                'kl_loss': kl_loss.item()
            }
        }

    def generate(self, states):
        with torch.no_grad():
            # ===== MAC Processing =====
            mac_output = self.mac(states)
            m = self.mac_output_proj(mac_output)
        
            # ===== Action Generation =====
            action_logits, _ = self.get_action_logits(states, m)
            return action_logits