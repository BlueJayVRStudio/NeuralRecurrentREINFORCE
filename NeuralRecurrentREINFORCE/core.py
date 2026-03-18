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


def binary_gumbel_sigmoid(logits, tau=1.0, training=False, eps=1e-10):
    if training:
        unif = torch.rand_like(logits)
        noise = torch.log(unif + eps) - torch.log(1.0 - unif + eps)
        y_soft = torch.sigmoid((logits + noise) / tau)
        # Straight-Through Estimator (STE)
        y_hard = (y_soft > 0.5).float()
        return (y_hard - y_soft).detach() + y_soft
    else:
        return (logits > 0).float()

# you do it on the loss
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


class ConcertedAttentionHead(nn.Module):
    def __init__(self, query_embeddings, input_dim, output_dim, n_tokens, embed_dim):
        super().__init__()
        self.E_q = query_embeddings

        self.n_queries = query_embeddings.size()[0]
        self.n_tokens = n_tokens
        self.embed_dim = embed_dim

        self.to_k = nn.Linear(input_dim, n_tokens * embed_dim)
        self.to_v = nn.Linear(input_dim, n_tokens * embed_dim)
        
        self.output_head = nn.Sequential(
            nn.Linear(self.n_queries * embed_dim, output_dim),
            nn.Tanh()
        )
        
        self.scale = math.sqrt(embed_dim)
        self.ln = nn.LayerNorm(self.n_queries * embed_dim)
    
    def forward(self, _input):
        k = self.to_k(_input).view(self.n_tokens, self.embed_dim)
        v = self.to_v(_input).view(self.n_tokens, self.embed_dim)
        attn_scores = torch.matmul(self.E_q, k.transpose(0, 1)) / self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attended_v = torch.matmul(attn_probs, v).view(self.n_queries * self.embed_dim)
        attended_v = self.ln(attended_v)
        return self.output_head(attended_v)

class RNN_Actor(nn.Module):
    def __init__(self, action_dim=4, state_dim=8, hidden_size=64, latent_size=8, cell_size=8, disk_size=8, n_tokens=10, n_queries=10, embed_dim=128):
        super().__init__()

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.cell_size = cell_size
        self.disk_size = disk_size

        self.combined_input_dim = state_dim + latent_size + cell_size + disk_size

        self.n_tokens = n_tokens
        self.n_queries = n_queries
        self.embed_dim = embed_dim

        # learnable query embeddings
        self.E_q = nn.Parameter(torch.randn(n_queries, embed_dim))

        # input head
        self.input_head = ConcertedAttentionHead(self.E_q, state_dim, state_dim, n_tokens, embed_dim)

        # internal recurrent components
        self.null_h = torch.nn.Parameter(torch.randn(latent_size)) #.requires_grad_(False)
        self._state = self.null_h.clone()
        self.cell = torch.randn(cell_size)
        self.init_disk = nn.Parameter(torch.randn(disk_size))
        self.disk = self.init_disk.clone()
        
        # recurrent node
        self.fake_rnn = nn.Sequential(
            nn.Linear(self.combined_input_dim, hidden_size),
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

        self.cell_head = ConcertedAttentionHead(self.E_q, hidden_size, cell_size, n_tokens, embed_dim)
        self.disk_head = ConcertedAttentionHead(self.E_q, hidden_size, disk_size, n_tokens, embed_dim)
        self.latent_head = ConcertedAttentionHead(self.E_q, hidden_size, latent_size, n_tokens, embed_dim)
        self.action_head = ConcertedAttentionHead(self.E_q, hidden_size, action_dim, n_tokens, embed_dim)

        self.reward_critic = Critic(state_dim, 1, hidden_size)

    def reset_internal_state(self):
        self._state = self.null_h.detach().clone()
        self.cell = torch.randn(self.cell_size)
        self.disk =  self.init_disk.detach().clone()
        # print(self.E_q)
        # print("instruction: ", self.null_h)
        # print("disk: ", self.disk)

    def forward(self, state, tau=1.0, hard=True):
        # transform input
        transformed_state_input = self.input_head(state)
        
        # run transformed input on fake rnn
        intermediate_logits = self.fake_rnn(torch.cat([transformed_state_input, self._state, self.cell, self.disk], dim=-1).squeeze(0))
        next_latent_logits = self.fake_rnn_hidden_state(intermediate_logits)
        next_cell_logits = self.fake_rnn_cell(intermediate_logits)
        next_disk_logits = self.fake_rnn_disk(intermediate_logits)
        action_logits = self.fake_rnn_action(intermediate_logits)

        # transform cell
        self.cell = self.cell_head(next_cell_logits)
        # transform disk
        self.disk = self.disk_head(next_disk_logits)
        # transform hidden state
        self._state = binary_gumbel_sigmoid(self.latent_head(next_latent_logits), training=False)
        # transform action
        logits = self.action_head(action_logits)
        dist = Categorical(logits=logits)
        return dist.sample()
    
    def evaluate(self, state, action):
        # transform input
        transformed_state_input = self.input_head(state)
        
        # run transformed input on fake rnn
        intermediate_logits = self.fake_rnn(torch.cat([transformed_state_input, self._state, self.cell, self.disk], dim=-1).squeeze(0))
        next_latent_logits = self.fake_rnn_hidden_state(intermediate_logits)
        next_cell_logits = self.fake_rnn_cell(intermediate_logits)
        next_disk_logits = self.fake_rnn_disk(intermediate_logits)
        action_logits = self.fake_rnn_action(intermediate_logits)

        # transform cell
        self.cell = self.cell_head(next_cell_logits)
        # transform disk
        self.disk = self.disk_head(next_disk_logits)
        # transform hidden state
        self._state = binary_gumbel_sigmoid(self.latent_head(next_latent_logits), training=True)
        # transform action
        logits = self.action_head(action_logits)
        
        dist = Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy(), transformed_state_input.clone().detach()

    def critic_evaluate(self, state, action, reward):
        # transform input
        transformed_state_input = self.input_head(state)
        
        # run transformed input on fake rnn
        combined_input = torch.cat([transformed_state_input, self._state, self.cell, self.disk], dim=-1).squeeze(0)
        intermediate_logits = self.fake_rnn(combined_input)
        next_latent_logits = self.fake_rnn_hidden_state(intermediate_logits)
        next_cell_logits = self.fake_rnn_cell(intermediate_logits)
        next_disk_logits = self.fake_rnn_disk(intermediate_logits)
        action_logits = self.fake_rnn_action(intermediate_logits)

        # transform cell
        self.cell = self.cell_head(next_cell_logits)
        # transform disk
        self.disk = self.disk_head(next_disk_logits)
        # transform hidden state
        self._state = binary_gumbel_sigmoid(self.latent_head(next_latent_logits), training=False)
        
        critic_loss = F.mse_loss(self.reward_critic(transformed_state_input, torch.tensor(action).unsqueeze(0)), torch.tensor(reward).float())
        return critic_loss

class Critic(nn.Module):
    def __init__(self, state_dim=8, action_dim=8, hidden_size=8):
        super().__init__()

        self.MLP = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size), 
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
    def forward(self, state, action):
        evaluation = self.MLP(torch.cat([state, action], dim=-1))
        return evaluation