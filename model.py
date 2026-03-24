"""
Orion Base Model Architecture
Mini-Llama style: RoPE, GQA, SwiGLU, RMSNorm, Flash Attention
Designed for context extension via YaRN from day 1.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

@dataclass
class OrionConfig:
    vocab_size: int = 32000
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4  # GQA: fewer KV heads than Q heads
    max_seq_len: int = 4096
    hidden_dim: int = None  # Auto-calculated for SwiGLU
    dropout: float = 0.0
    rope_theta: float = 10000.0
    # YaRN parameters for context extension
    yarn_scale: float = 1.0
    yarn_original_max_seq_len: int = 4096
    
    def __post_init__(self):
        if self.hidden_dim is None:
            # SwiGLU hidden dim: 2/3 * 4 * dim, rounded to multiple of 256
            self.hidden_dim = int(2 * self.dim * 4 / 3)
            self.hidden_dim = ((self.hidden_dim + 255) // 256) * 256


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0, yarn_scale: float = 1.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    if yarn_scale != 1.0:
        # NTK-aware scaling for context extension
        freqs = freqs / yarn_scale
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rope(x, cos, sin):
    B, H, T, D = x.shape
    cos = cos[:T].unsqueeze(0).unsqueeze(0)  # [1, 1, T, D//2]
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    x1 = x[..., :D//2]
    x2 = x[..., D//2:]
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.cat([out1, out2], dim=-1)


class GQAAttention(nn.Module):
    """Grouped Query Attention with RoPE."""
    def __init__(self, config: OrionConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads  # How many Q heads per KV head
        
        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, cos, sin, mask=None):
        B, T, C = x.shape
        
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to Q and K
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        
        # Expand KV heads to match Q heads (GQA)
        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.n_heads, T, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.n_heads, T, self.head_dim)
        
        # Try Flash Attention, fall back to standard
        try:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout.p if self.training else 0.0, is_causal=mask is None)
        except Exception:
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = (q @ k.transpose(-2, -1)) * scale
            if mask is not None:
                attn = attn + mask
            else:
                causal = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
                attn = attn + causal
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = attn @ v
        
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""
    def __init__(self, config: OrionConfig):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)  # Gate
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)  # Down
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)  # Up
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class OrionBlock(nn.Module):
    def __init__(self, config: OrionConfig):
        super().__init__()
        self.attention = GQAAttention(config)
        self.ffn = SwiGLU(config)
        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)
    
    def forward(self, x, cos, sin, mask=None):
        x = x + self.attention(self.norm1(x), cos, sin, mask)
        x = x + self.ffn(self.norm2(x))
        return x


class OrionModel(nn.Module):
    def __init__(self, config: OrionConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([OrionBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Weight tying
        self.output.weight = self.tok_emb.weight
        
        # Precompute RoPE frequencies
        cos, sin = precompute_rope_freqs(
            config.dim // config.n_heads,
            config.max_seq_len,
            config.rope_theta,
            config.yarn_scale
        )
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)
        
        # Init weights
        self.apply(self._init_weights)
        # Scale residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('wo.weight') or pn.endswith('w2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin)
        
        x = self.norm(x)
        logits = self.output(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# Preset configs
CONFIGS = {
    "10M": OrionConfig(dim=384, n_layers=6, n_heads=6, n_kv_heads=2),
    "50M": OrionConfig(dim=512, n_layers=12, n_heads=8, n_kv_heads=2),
    "125M": OrionConfig(dim=768, n_layers=12, n_heads=12, n_kv_heads=4),
    "500M": OrionConfig(dim=1024, n_layers=24, n_heads=16, n_kv_heads=4),
    "1B": OrionConfig(dim=2048, n_layers=24, n_heads=16, n_kv_heads=4),
    "7B": OrionConfig(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        hidden_dim=14336,  # Explicit override — auto formula gives 11008 (~5.8B); 14336 targets ~7.1B
        max_seq_len=8192,
        rope_theta=500000.0,  # Llama 3 style long-range RoPE
        yarn_original_max_seq_len=8192,
    ),
}


def count_params_from_config(cfg: OrionConfig) -> int:
    """
    Calculate total parameter count from an OrionConfig without instantiating the model.
    Useful for large models (7B+) where CPU RAM would be exhausted.

    Breakdown:
      - Token embedding:  vocab_size × dim  (shared with output head via weight tying)
      - Per transformer layer:
          Attention:  wq (dim × (n_heads × head_dim))
                      wk (dim × (n_kv_heads × head_dim))
                      wv (dim × (n_kv_heads × head_dim))
                      wo ((n_heads × head_dim) × dim)
          FFN:        w1 (dim × hidden_dim)  [gate]
                      w2 (hidden_dim × dim)  [down]
                      w3 (dim × hidden_dim)  [up]
          RMSNorm ×2: 2 × dim               [norm1, norm2]
      - Final RMSNorm: dim
      - Output head: tied to embedding (0 extra params)
    """
    head_dim = cfg.dim // cfg.n_heads

    embed = cfg.vocab_size * cfg.dim

    attn = (
        cfg.dim * (cfg.n_heads * head_dim) +     # wq
        cfg.dim * (cfg.n_kv_heads * head_dim) +  # wk
        cfg.dim * (cfg.n_kv_heads * head_dim) +  # wv
        (cfg.n_heads * head_dim) * cfg.dim        # wo
    )
    ffn = (
        cfg.dim * cfg.hidden_dim +   # w1
        cfg.hidden_dim * cfg.dim +   # w2
        cfg.dim * cfg.hidden_dim     # w3
    )
    norms_per_layer = 2 * cfg.dim
    per_layer = attn + ffn + norms_per_layer

    final_norm = cfg.dim

    return embed + cfg.n_layers * per_layer + final_norm


if __name__ == "__main__":
    print("Orion Model Architecture Sizes:")
    for name, cfg in CONFIGS.items():
        if name == "7B":
            # Don't instantiate — too large for CPU RAM
            params = count_params_from_config(cfg)
            print(f"  {name}: {params:,} parameters ({params/1e9:.3f}B) | dim={cfg.dim} layers={cfg.n_layers} heads={cfg.n_heads} kv_heads={cfg.n_kv_heads} hidden={cfg.hidden_dim} [estimated, no instantiation]")
        else:
            model = OrionModel(cfg)
            params = model.count_params()
            print(f"  {name}: {params:,} parameters ({params/1e6:.1f}M) | dim={cfg.dim} layers={cfg.n_layers} heads={cfg.n_heads} kv_heads={cfg.n_kv_heads} hidden={cfg.hidden_dim}")
