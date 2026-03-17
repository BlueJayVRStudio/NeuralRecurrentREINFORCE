import numpy as np
import scipy.signal
from gymnasium.spaces import Box, Discrete
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import Uniform
from torch.distributions.categorical import Categorical
from torch.nn.utils import spectral_norm
import math

def get_autograd_depth(tensor):
    if not hasattr(tensor, 'grad_fn') or tensor.grad_fn is None:
        return 0
    
    def walk(fn):
        if not fn or not hasattr(fn, 'next_functions'):
            return 0
        depths = [walk(next_f[0]) for next_f in fn.next_functions if next_f[0] is not None]
        return 1 + (max(depths) if depths else 0)

    return walk(tensor.grad_fn)

def get_autograd_depth_iterative(tensor):
    if tensor.grad_fn is None:
        return 0
    
    queue = [(tensor.grad_fn, 1)]
    max_depth = 0
    visited = set()

    while queue:
        fn, depth = queue.pop(0)
        if fn in visited:
            continue
        visited.add(fn)
        
        max_depth = max(max_depth, depth)
        
        if hasattr(fn, 'next_functions'):
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    queue.append((next_fn, depth + 1))
    return max_depth

def soft_update(target, source, tau=0.005):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def binary_gumbel_sigmoid(logits, tau=1.0, training=False, eps=1e-10):
    if training:
        unif = torch.rand_like(logits)
        noise = torch.log(unif + eps) - torch.log(1.0 - unif + eps)
        y_soft = torch.sigmoid((logits + noise) / tau)
        # STE
        y_hard = (y_soft > 0.5).float()
        return (y_hard - y_soft).detach() + y_soft
    else:
        return (logits > 0).float()

class RNN_Actor(nn.Module):
    def __init__(self, action_dim=4, state_dim=8, hidden_size=64, latent_size=8, cell_size=8, disk_size=8, n_tokens=10, n_queries=10, embed_dim=256):
        super().__init__()

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.cell_size = cell_size
        self.disk_size = disk_size

        self.n_tokens = n_tokens
        self.n_queries = n_queries
        self.embed_dim = embed_dim

        # learnable query embeddings
        self.E_q = nn.Parameter(torch.randn(n_queries, embed_dim))

        # input head
        self.to_k_input = nn.Linear(state_dim, n_tokens * embed_dim)
        self.to_v_input = nn.Linear(state_dim, n_tokens * embed_dim)
        self.input_head = nn.Sequential(
            nn.Linear(n_queries * embed_dim, state_dim),
            nn.Tanh()
        )

        # internal recurrent components
        self.null_h = torch.nn.Parameter(torch.randn(latent_size)) #.requires_grad_(False)
        self._state = self.null_h.clone()
        self.cell = torch.randn(cell_size)
        self.init_disk = nn.Parameter(torch.randn(disk_size))
        self.disk = self.init_disk.clone()
        
        # recurrent node
        self.fake_rnn = nn.Sequential(
            nn.Linear(state_dim + latent_size + cell_size + disk_size, hidden_size),
            nn.Tanh()
        )
        # intermediate logits to respective logits
        self.fake_rnn_hidden_state = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        self.fake_rnn_cell = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        self.fake_rnn_disk = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        self.fake_rnn_action = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

        self.to_k_cell = nn.Linear(hidden_size, n_tokens * embed_dim)
        self.to_v_cell = nn.Linear(hidden_size, n_tokens * embed_dim)
        self.cell_head = nn.Sequential(
            nn.Linear(n_queries * embed_dim, cell_size),
            nn.Tanh()
        )
        
        self.to_k_disk = nn.Linear(hidden_size, n_tokens * embed_dim)
        self.to_v_disk = nn.Linear(hidden_size, n_tokens * embed_dim)
        self.disk_head = nn.Sequential(
            nn.Linear(n_queries * embed_dim, disk_size),
            nn.Tanh()
        )
        
        self.to_k_hidden = nn.Linear(hidden_size, n_tokens * embed_dim)
        self.to_v_hidden = nn.Linear(hidden_size, n_tokens * embed_dim)
        self.hidden_head = nn.Sequential(
            nn.Linear(n_queries * embed_dim, latent_size),
            nn.Tanh()
        )

        self.to_k_action = nn.Linear(hidden_size, n_tokens * embed_dim)
        self.to_v_action = nn.Linear(hidden_size, n_tokens * embed_dim)
        self.action_head = nn.Sequential(
            nn.Linear(n_queries * embed_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim)
        )

        # scaling factor for attention
        self.scale = math.sqrt(embed_dim)
        # layer norms
        self.ln_state_input = nn.LayerNorm(n_queries * embed_dim)
        self.ln_cell = nn.LayerNorm(n_queries * embed_dim)
        self.ln_disk = nn.LayerNorm(n_queries * embed_dim)
        self.ln_hidden_state = nn.LayerNorm(n_queries * embed_dim)
        self.ln_action = nn.LayerNorm(n_queries * embed_dim)

    def reset_internal_state(self):
        self._state = self.null_h.detach().clone()
        self.cell = torch.randn(self.cell_size)
        self.disk =  self.init_disk.detach().clone()
        # print(self.E_q)
        # print("instruction: ", self.null_h)
        # print("disk: ", self.disk)

    def forward(self, state, tau=1.0, hard=True):
        # transform input
        k = self.to_k_input(state).view(self.n_tokens, self.embed_dim)
        v = self.to_v_input(state).view(self.n_tokens, self.embed_dim)
        attn_scores = torch.matmul(self.E_q, k.transpose(0, 1)) / self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attended_v = torch.matmul(attn_probs, v).view(self.n_queries * self.embed_dim)
        attended_v = self.ln_state_input(attended_v)
        transformed_state_input = self.input_head(attended_v)
        
        # run transformed input on fake rnn
        intermediate_logits = self.fake_rnn(torch.cat([transformed_state_input, self._state, self.cell, self.disk], dim=-1).squeeze(0))
        next_state_logits = self.fake_rnn_hidden_state(intermediate_logits)
        next_cell_logits = self.fake_rnn_cell(intermediate_logits)
        next_disk_logits = self.fake_rnn_disk(intermediate_logits)
        action_logits = self.fake_rnn_action(intermediate_logits)

        # transform cell
        k = self.to_k_cell(next_cell_logits).view(self.n_tokens, self.embed_dim)
        v = self.to_v_cell(next_cell_logits).view(self.n_tokens, self.embed_dim)
        attn_scores = torch.matmul(self.E_q, k.transpose(0, 1)) / self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attended_v = torch.matmul(attn_probs, v).view(self.n_queries * self.embed_dim)
        attended_v = self.ln_cell(attended_v)
        ##
        self.cell = self.cell_head(attended_v)

        # transform disk
        k = self.to_k_disk(next_disk_logits).view(self.n_tokens, self.embed_dim)
        v = self.to_v_disk(next_disk_logits).view(self.n_tokens, self.embed_dim)
        attn_scores = torch.matmul(self.E_q, k.transpose(0, 1)) / self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attended_v = torch.matmul(attn_probs, v).view(self.n_queries * self.embed_dim)
        attended_v = self.ln_disk(attended_v)
        ##
        self.disk = self.disk_head(attended_v)

        # transform hidden state
        k = self.to_k_hidden(next_state_logits).view(self.n_tokens, self.embed_dim)
        v = self.to_v_hidden(next_state_logits).view(self.n_tokens, self.embed_dim)
        attn_scores = torch.matmul(self.E_q, k.transpose(0, 1)) / self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attended_v = torch.matmul(attn_probs, v).view(self.n_queries * self.embed_dim)
        attended_v = self.ln_hidden_state(attended_v)
        ##
        self._state = binary_gumbel_sigmoid(self.hidden_head(attended_v), training=False)

        # transform action
        k = self.to_k_action(action_logits).view(self.n_tokens, self.embed_dim)
        v = self.to_v_action(action_logits).view(self.n_tokens, self.embed_dim)
        attn_scores = torch.matmul(self.E_q, k.transpose(0, 1)) / self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attended_v = torch.matmul(attn_probs, v).view(self.n_queries * self.embed_dim)
        attended_v = self.ln_action(attended_v)
        logits = self.action_head(attended_v)
        dist = Categorical(logits=logits)
        return dist.sample()
    
    def evaluate(self, state, action):
        # transform input
        k = self.to_k_input(state).view(self.n_tokens, self.embed_dim)
        v = self.to_v_input(state).view(self.n_tokens, self.embed_dim)
        attn_scores = torch.matmul(self.E_q, k.transpose(0, 1)) / self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attended_v = torch.matmul(attn_probs, v).view(self.n_queries * self.embed_dim)
        attended_v = self.ln_state_input(attended_v)
        transformed_state_input = self.input_head(attended_v)
        
        # run transformed input on fake rnn
        intermediate_logits = self.fake_rnn(torch.cat([transformed_state_input, self._state, self.cell, self.disk], dim=-1).squeeze(0))
        next_state_logits = self.fake_rnn_hidden_state(intermediate_logits)
        next_cell_logits = self.fake_rnn_cell(intermediate_logits)
        next_disk_logits = self.fake_rnn_disk(intermediate_logits)
        action_logits = self.fake_rnn_action(intermediate_logits)

        # transform cell
        k = self.to_k_cell(next_cell_logits).view(self.n_tokens, self.embed_dim)
        v = self.to_v_cell(next_cell_logits).view(self.n_tokens, self.embed_dim)
        attn_scores = torch.matmul(self.E_q, k.transpose(0, 1)) / self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attended_v = torch.matmul(attn_probs, v).view(self.n_queries * self.embed_dim)
        attended_v = self.ln_cell(attended_v)
        ##
        self.cell = self.cell_head(attended_v)

        # transform disk
        k = self.to_k_disk(next_disk_logits).view(self.n_tokens, self.embed_dim)
        v = self.to_v_disk(next_disk_logits).view(self.n_tokens, self.embed_dim)
        attn_scores = torch.matmul(self.E_q, k.transpose(0, 1)) / self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attended_v = torch.matmul(attn_probs, v).view(self.n_queries * self.embed_dim)
        attended_v = self.ln_disk(attended_v)
        ##
        self.disk = self.disk_head(attended_v)

        # transform hidden state
        k = self.to_k_hidden(next_state_logits).view(self.n_tokens, self.embed_dim)
        v = self.to_v_hidden(next_state_logits).view(self.n_tokens, self.embed_dim)
        attn_scores = torch.matmul(self.E_q, k.transpose(0, 1)) / self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attended_v = torch.matmul(attn_probs, v).view(self.n_queries * self.embed_dim)
        attended_v = self.ln_hidden_state(attended_v)
        ##
        self._state = binary_gumbel_sigmoid(self.hidden_head(attended_v), training=True)

        # transform action
        k = self.to_k_action(action_logits).view(self.n_tokens, self.embed_dim)
        v = self.to_v_action(action_logits).view(self.n_tokens, self.embed_dim)
        attn_scores = torch.matmul(self.E_q, k.transpose(0, 1)) / self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attended_v = torch.matmul(attn_probs, v).view(self.n_queries * self.embed_dim)
        attended_v = self.ln_action(attended_v)
        logits = self.action_head(attended_v)
        
        dist = Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy()

