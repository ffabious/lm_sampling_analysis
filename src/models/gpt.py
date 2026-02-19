from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class GPTConfig:
    vocab_size: int
    max_seq_len: int = 256
    n_layers: int = 4
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    bias: bool = True
    layer_norm_eps: float = 1e-5

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.config = config
        self.head_dim = config.d_model // config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.out = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        b, t, c = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(c, dim=2)

        q = q.view(b, t, self.config.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.config.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.config.n_heads, self.head_dim).transpose(1, 2)

        past_len = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.size(2)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        attn_mask = None
        is_causal = True
        if past_len > 0:
            total_k = k.size(2)
            q_positions = past_len + torch.arange(t, device=x.device).unsqueeze(1)
            k_positions = torch.arange(total_k, device=x.device).unsqueeze(0)
            attn_mask = k_positions <= q_positions
            is_causal = False

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.config.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.resid_dropout(self.out(y))

        present_kv = (k, v) if use_cache else None
        return y, present_kv
    
class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        attn_out, present_kv = self.attn(self.ln1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, present_kv
    
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor = None,
        past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
    ):
        b, t = input_ids.shape
        assert t <= self.config.max_seq_len

        if past_key_values is None:
            past_key_values = [None] * len(self.blocks)

        past_len = 0
        if past_key_values[0] is not None:
            past_len = past_key_values[0][0].size(2)
        assert past_len + t <= self.config.max_seq_len

        pos = torch.arange(past_len, past_len + t, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(pos)[None, :, :]
        x = self.drop(x)

        new_past_key_values = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            x, present_kv = block(x, past_kv=past_key_values[i], use_cache=use_cache)
            if use_cache:
                new_past_key_values.append(present_kv)
        
        x = self.ln_f(x)
        logits = self.head(x)

        out = {"logits": logits}
        if use_cache:
            out["past_key_values"] = new_past_key_values
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            out["loss"] = loss
        return out