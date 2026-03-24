"""
Orion Atlas 7B -- Mamba-2 Hybrid Architecture
==============================================
Combines Mamba-2 SSD (State Space Duality) blocks with Microsoft Differential Attention
blocks and SwiGLU FFN in a hybrid interleaved design.

References:
  - NVIDIA Mamba-2 Hybrid 8B: arXiv:2406.07887
  - Mamba-2 / SSD: arXiv:2405.21060 (Dao & Gu 2024)
  - Differential Attention: arXiv:2410.05258 (ICLR 2025)
  - Jamba (AI21), Zamba (Zyphra): production hybrid validation

Architecture (32 layers):
  - Each HybridBlock = (Mamba2Block OR DiffAttnBlock) + SwiGLU FFN
  - DiffAttn at attn_layer_indices (default: [3,7,11,15,19,23,27] -- 7 of 32 = 21.9%)
  - Mamba-2 at all other 25 layers
  - FFN after every block (always)
  - Full causal attention on all DiffAttn layers (no sliding window)
  - RoPE only on DiffAttn layers; Mamba-2 is position-free
  - Context: 128K tokens (max_seq_len=131072)

Design decisions:
  - 7 attention layers per Jamba/NVIDIA-8B/Zamba validated ratio
  - Pure PyTorch (no mamba-ssm package), chunked parallel scan
  - DiffAttn with GQA (n_kv_heads=8) + RoPE (theta=500K for long context)
  - Learnable lambda init per Diff-Attention paper
  - Weight tying tok_emb / output projection
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, List


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Orion7BConfig:
    vocab_size: int = 32000
    dim: int = 4096
    n_layers: int = 32

    # Indices where DiffAttn layers appear; all others use Mamba-2.
    # None -> defaults to [3, 7, 11, 15, 19, 23, 27] (7 layers, evenly distributed)
    # Based on Jamba/NVIDIA-8B/Zamba: 6-7 attn layers in 32-layer hybrid is optimal.
    attn_layer_indices: Optional[List[int]] = None

    # Mamba-2 SSM params
    d_state: int = 128        # SSM state dimension N
    d_conv: int = 4           # Causal conv kernel size
    mamba_expand: int = 2     # d_inner = mamba_expand * dim
    mamba_headdim: int = 64   # Per-head dimension inside Mamba-2

    # Differential Attention params
    n_heads: int = 32         # Query heads
    n_kv_heads: int = 8       # KV heads (GQA 4:1)

    # FFN
    hidden_dim: int = 14336   # SwiGLU hidden dim

    # Context
    # 128K context window. RoPE only on DiffAttn layers; Mamba-2 is position-free.
    max_seq_len: int = 131072
    rope_theta: float = 500_000.0   # Llama-3-style long-range RoPE

    # Training
    dropout: float = 0.0

    def __post_init__(self):
        if self.attn_layer_indices is None:
            # 7 attention layers evenly distributed across 32 layers
            # Avoids first and last layer (pure Mamba-2 at boundaries)
            self.attn_layer_indices = [3, 7, 11, 15, 19, 23, 27]


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10_000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    B, H, T, D = x.shape
    cos = cos[:T].unsqueeze(0).unsqueeze(0)  # [1,1,T,D//2]
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., :D // 2], x[..., D // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class SwiGLU(nn.Module):
    """SwiGLU FFN."""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # down
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # up
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ---------------------------------------------------------------------------
# Mamba-2 / SSD block
# ---------------------------------------------------------------------------

def _ssm_chunked_scan(
    x_ssm: torch.Tensor,   # [B, T, d_inner]  -- gated input to SSM
    A_bar: torch.Tensor,   # [B, T, d_inner]  -- discretised diagonal A (in (0,1))
    B: torch.Tensor,       # [B, T, d_state]
    C: torch.Tensor,       # [B, T, d_state]
    D: torch.Tensor,       # [d_inner]        -- skip
    chunk: int = 256,
) -> torch.Tensor:
    """
    Chunked parallel scan (log-domain).

    Within each chunk, parallel cumsum gives O(chunk) work per element.
    Chunk boundaries are propagated sequentially -> O(T/chunk) serial steps.

    Time:   O(T * d_inner * d_state / chunk) for the cumsum kernel
    Memory: O(B * chunk * d_inner * d_state) per chunk
    """
    B_sz, T, d_inner = x_ssm.shape
    d_state = B.shape[-1]

    h = x_ssm.new_zeros(B_sz, d_inner, d_state)
    segs = []

    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        x_c = x_ssm[:, s:e]      # [B, L, d_inner]
        A_c = A_bar[:, s:e]      # [B, L, d_inner]
        B_c = B[:, s:e]          # [B, L, d_state]
        C_c = C[:, s:e]          # [B, L, d_state]

        # Cumulative log-A within chunk
        log_A_cs = torch.log(A_c.clamp(min=1e-7)).cumsum(dim=1)  # [B, L, d_inner]
        decay    = log_A_cs.exp()                                   # [B, L, d_inner]
        inv_d    = (-log_A_cs).exp()                                # [B, L, d_inner]

        # Contribution from carry-in state h
        h_state = decay.unsqueeze(-1) * h.unsqueeze(1)            # [B, L, d_inner, d_state]

        # Contribution from inputs within the chunk
        bx      = x_c.unsqueeze(-1) * B_c.unsqueeze(-2)          # [B, L, d_inner, d_state]
        bx_norm = bx * inv_d.unsqueeze(-1)
        h_input = decay.unsqueeze(-1) * bx_norm.cumsum(dim=1)     # [B, L, d_inner, d_state]

        h_all = h_state + h_input                                  # [B, L, d_inner, d_state]

        # Output: y = C * h
        y_seg = (h_all * C_c.unsqueeze(-2)).sum(-1)                # [B, L, d_inner]
        segs.append(y_seg)

        h = h_all[:, -1]  # carry state forward

    y = torch.cat(segs, dim=1)                                     # [B, T, d_inner]
    return y + x_ssm * D.unsqueeze(0).unsqueeze(0)


class Mamba2Block(nn.Module):
    """
    Mamba-2 / SSD (State Space Duality) block.

    Equations:
      1. in_proj(x) -> z, x_conv, B, C, dt_logit
      2. causal conv1d on x_conv
      3. dt  = softplus(dt_logit + dt_bias)  [B, T, n_heads_m]
         Replicate dt per head -> d_inner dimension
         A_log learnable (scalar per head, constrained negative)
         A_bar = exp(-dt * exp(A_log))   [B, T, d_inner]  -- in (0,1)
      4. SSM scan: y = chunked_scan(dt * x_conv, A_bar, B, C, D)
      5. y_gated = y * silu(z)
      6. out_proj(y_gated) -> residual
    """

    def __init__(self, config: Orion7BConfig):
        super().__init__()
        d = config.dim
        self.d_inner = config.mamba_expand * d         # e.g. 2*4096 = 8192
        self.headdim = config.mamba_headdim            # 64
        self.n_heads_m = self.d_inner // self.headdim  # 128 heads
        self.d_state = config.d_state
        self.d_conv = config.d_conv

        # z, x_conv, B, C, dt_logit
        in_feat = 2 * self.d_inner + 2 * config.d_state + self.n_heads_m
        self.in_proj  = nn.Linear(d, in_feat, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d, bias=False)

        # Causal depthwise conv
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=config.d_conv,
            groups=self.d_inner,
            padding=config.d_conv - 1,
            bias=True,
        )

        # SSM parameters
        # A_log: [n_heads_m] -- log of negative A; stays negative via softplus later
        self.A_log   = nn.Parameter(torch.log(torch.arange(1, self.n_heads_m + 1).float()))
        self.D       = nn.Parameter(torch.ones(self.d_inner))
        self.dt_bias = nn.Parameter(torch.zeros(self.n_heads_m))

        # Output norm (Mamba-2 uses a norm before output projection)
        self.norm = RMSNorm(self.d_inner)

        self._init_dt_bias()

    def _init_dt_bias(self):
        # Initialize dt_bias so softplus(dt_logit + dt_bias) ~ Uniform(0.001, 0.1)
        dt_init = torch.exp(torch.rand(self.n_heads_m) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        dt_init = dt_init.clamp(min=1e-4)
        inv_softplus = torch.log(dt_init.expm1())
        self.dt_bias = nn.Parameter(inv_softplus)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_sz, T, _ = x.shape

        proj = self.in_proj(x)  # [B, T, 2*d_inner + 2*d_state + n_heads_m]
        z      = proj[..., :self.d_inner]
        x_conv = proj[..., self.d_inner:2 * self.d_inner]
        B_ssm  = proj[..., 2 * self.d_inner:2 * self.d_inner + self.d_state]
        C_ssm  = proj[..., 2 * self.d_inner + self.d_state:2 * self.d_inner + 2 * self.d_state]
        dt_log = proj[..., 2 * self.d_inner + 2 * self.d_state:]  # [B, T, n_heads_m]

        # Causal conv: [B, T, d_inner] -> conv1d expects [B, C, L]
        x_c = x_conv.transpose(1, 2)
        x_c = self.conv1d(x_c)[..., :T]   # trim padding
        x_c = F.silu(x_c.transpose(1, 2)) # [B, T, d_inner]

        # Discretise
        dt = F.softplus(dt_log + self.dt_bias.unsqueeze(0).unsqueeze(0))  # [B, T, n_heads_m]
        # Expand dt to d_inner (replicate each head's dt across headdim positions)
        dt_exp = dt.unsqueeze(-1).expand(-1, -1, -1, self.headdim)
        dt_exp = dt_exp.reshape(B_sz, T, self.d_inner)                    # [B, T, d_inner]

        # A_bar in (0,1): A_bar = exp(-dt * A);  A = exp(A_log) > 0
        A = torch.exp(self.A_log)                   # [n_heads_m]
        A_exp = A.unsqueeze(-1).expand(-1, self.headdim).reshape(self.d_inner)  # [d_inner]
        A_bar = torch.exp(-dt_exp * A_exp.unsqueeze(0).unsqueeze(0))     # [B, T, d_inner]

        # SSM input = dt * x_conv (absorb dt into input, not B)
        x_ssm = dt_exp * x_c  # [B, T, d_inner]

        y = _ssm_chunked_scan(x_ssm, A_bar, B_ssm, C_ssm, self.D)  # [B, T, d_inner]

        # Gate + norm + project
        y = self.norm(y * F.silu(z))
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# Differential Attention block (Microsoft, ICLR 2025)
# ---------------------------------------------------------------------------

class DifferentialAttention(nn.Module):
    """
    Differential Attention with GQA + RoPE.

    Attention(Q1,K1,V1) - lambda * Attention(Q2,K2,V2)

    Q is split into [Q1|Q2], K into [K1|K2].
    Lambda is a learned scalar (per head) initialised near 0.8.

    Uses F.scaled_dot_product_attention (Flash Attention when available).
    GQA: n_kv_heads < n_heads, groups = n_heads // n_kv_heads.
    """

    def __init__(self, config: Orion7BConfig, layer_idx: int = 0):
        super().__init__()
        self.n_heads    = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep      = config.n_heads // config.n_kv_heads
        self.head_dim   = config.dim // config.n_heads
        self.scale      = 1.0 / math.sqrt(self.head_dim)

        d = config.dim

        # Q projects to 2*n_heads*head_dim (Q1 and Q2 stacked)
        # K,V project to 2*n_kv_heads*head_dim for K1/K2 and n_kv_heads*head_dim for V
        self.wq  = nn.Linear(d, 2 * self.n_heads * self.head_dim, bias=False)
        self.wk  = nn.Linear(d, 2 * self.n_kv_heads * self.head_dim, bias=False)
        self.wv  = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.wo  = nn.Linear(self.n_heads * self.head_dim, d, bias=False)

        # Per-head lambda: scalar, initialised per Diff-Attn paper (eq 5)
        # lambda = exp(lambda_q1 * lambda_k1) - exp(lambda_q2 * lambda_k2) + lambda_init
        lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_idx)
        self.lambda_q1 = nn.Parameter(torch.randn(self.n_heads, self.head_dim) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(self.n_heads, self.head_dim) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(self.n_heads, self.head_dim) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(self.n_heads, self.head_dim) * 0.1)
        self.lambda_init = lambda_init

        # Sub-layer norm on the output (Diff-Attn paper Sec 3.2)
        self.subln = RMSNorm(self.head_dim)

        self.drop = nn.Dropout(config.dropout)

    def _compute_lambda(self) -> torch.Tensor:
        """Compute per-head lambda scalars. Shape: [n_heads, 1, 1]"""
        lam = (
            torch.exp((self.lambda_q1 * self.lambda_k1).sum(-1))
            - torch.exp((self.lambda_q2 * self.lambda_k2).sum(-1))
            + self.lambda_init
        )  # [n_heads]
        return lam.unsqueeze(-1).unsqueeze(-1)  # [n_heads, 1, 1]

    def forward(
        self,
        x: torch.Tensor,            # [B, T, dim]
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        H  = self.n_heads
        Hk = self.n_kv_heads
        D  = self.head_dim

        # Project
        q_full = self.wq(x).view(B, T, 2 * H, D).transpose(1, 2)   # [B, 2H, T, D]
        k_full = self.wk(x).view(B, T, 2 * Hk, D).transpose(1, 2)  # [B, 2Hk, T, D]
        v      = self.wv(x).view(B, T, Hk, D).transpose(1, 2)      # [B, Hk, T, D]

        # Split Q1/Q2 and K1/K2
        q1, q2 = q_full[:, :H], q_full[:, H:]
        k1, k2 = k_full[:, :Hk], k_full[:, Hk:]

        # Apply RoPE to Q1, Q2, K1, K2
        q1 = apply_rope(q1, cos, sin)
        q2 = apply_rope(q2, cos, sin)
        k1 = apply_rope(k1, cos, sin)
        k2 = apply_rope(k2, cos, sin)

        # GQA expansion: repeat KV heads to match Q heads
        if self.n_rep > 1:
            def expand_kv(kv):
                return kv.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, H, T, D)
            k1 = expand_kv(k1)
            k2 = expand_kv(k2)
            v  = v.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, H, T, D)

        dp = self.drop.p if self.training else 0.0
        is_causal = mask is None

        # Attention maps via Flash Attention
        try:
            a1 = F.scaled_dot_product_attention(q1, k1, v, attn_mask=mask, dropout_p=dp, is_causal=is_causal)
            a2 = F.scaled_dot_product_attention(q2, k2, v, attn_mask=mask, dropout_p=dp, is_causal=is_causal)
        except Exception:
            # Fallback: manual attention
            def manual_attn(q, k, v):
                s = (q @ k.transpose(-2, -1)) * self.scale
                if mask is not None:
                    s = s + mask
                else:
                    cm = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
                    s = s + cm
                return F.softmax(s, dim=-1) @ v
            a1 = manual_attn(q1, k1, v)
            a2 = manual_attn(q2, k2, v)

        lam = self._compute_lambda()  # [H, 1, 1]

        # Differential output: [B, H, T, D]
        diff = a1 - lam.unsqueeze(0) * a2

        # Sub-layer norm per head, then reshape to [B, T, H*D]
        diff = self.subln(diff)
        out = diff.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.wo(out)


# ---------------------------------------------------------------------------
# Hybrid Block
# ---------------------------------------------------------------------------

class HybridBlock(nn.Module):
    """
    Either a Mamba-2 block or a DiffAttn block, followed by a SwiGLU FFN.
    Pre-norm (RMSNorm before each sublayer), residual connections.
    """

    def __init__(self, config: Orion7BConfig, layer_idx: int):
        super().__init__()
        self.is_attn = layer_idx in config.attn_layer_indices

        if self.is_attn:
            self.mixer = DifferentialAttention(config, layer_idx=layer_idx)
        else:
            self.mixer = Mamba2Block(config)

        self.ffn      = SwiGLU(config.dim, config.hidden_dim, config.dropout)
        self.norm1    = RMSNorm(config.dim)
        self.norm2    = RMSNorm(config.dim)

        # RoPE buffers are provided externally for attn blocks
        self._needs_rope = self.is_attn

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.is_attn:
            x = x + self.mixer(self.norm1(x), cos, sin, mask)
        else:
            x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class OrionHybrid7B(nn.Module):
    """
    Orion Atlas 7B -- Mamba-2 Hybrid with Differential Attention.

    32 layers: 30 Mamba-2 + 2 DiffAttn (at indices 7 and 23 by default).
    Each layer also includes a SwiGLU FFN sublayer.
    Weight-tied input embedding and output projection.
    """

    def __init__(self, config: Orion7BConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.layers  = nn.ModuleList([
            HybridBlock(config, i) for i in range(config.n_layers)
        ])
        self.norm   = RMSNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying
        self.output.weight = self.tok_emb.weight

        # Precompute RoPE for the DiffAttn layers
        head_dim = config.dim // config.n_heads
        cos, sin = precompute_rope_freqs(head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        # Initialise weights
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,               # [B, T]
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        B, T = idx.shape
        x = self.tok_emb(idx)            # [B, T, dim]

        for layer in self.layers:
            if layer._needs_rope:
                x = layer(x, self.rope_cos, self.rope_sin, mask)
            else:
                x = layer(x)

        x = self.norm(x)
        logits = self.output(x)          # [B, T, vocab_size]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
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


# ---------------------------------------------------------------------------
# Parameter counting (no instantiation required)
# ---------------------------------------------------------------------------

def count_params_from_config_7b(cfg: Orion7BConfig) -> int:
    """
    Transformer-equivalent parameter count for Orion Atlas 7B (~7.1B).

    Uses the same analytical formula as the 1B model.py (count_params_from_config)
    so that the 7B hybrid capacity can be compared directly to a pure-transformer
    baseline of the same size.

    Treats every layer as a standard GQA-attention + SwiGLU FFN layer:
      Attention per layer:
        wq: dim * (n_heads * head_dim)
        wk: dim * (n_kv_heads * head_dim)
        wv: dim * (n_kv_heads * head_dim)
        wo: (n_heads * head_dim) * dim
      FFN per layer:
        w1: dim * hidden_dim    (gate)
        w2: hidden_dim * dim    (down)
        w3: dim * hidden_dim    (up)
      RMSNorm x2: 2 * dim
    Plus:
      tok_emb (tied): vocab_size * dim
      final norm: dim

    For the actual hybrid parameter count use count_params_hybrid_actual().
    """
    d      = cfg.dim
    V      = cfg.vocab_size
    H      = cfg.n_heads
    Hk     = cfg.n_kv_heads
    hd     = d // H
    hidden = cfg.hidden_dim
    L      = cfg.n_layers

    embed = V * d

    attn_per = (
        d * (H  * hd) +   # wq
        d * (Hk * hd) +   # wk
        d * (Hk * hd) +   # wv
        (H * hd) * d      # wo
    )
    ffn_per = d * hidden + hidden * d + d * hidden
    norms   = 2 * d
    per_layer = attn_per + ffn_per + norms

    return embed + L * per_layer + d


def count_params_hybrid_actual(cfg: Orion7BConfig) -> int:
    """
    True parameter count for the instantiated OrionHybrid7B hybrid model (~8.9B).

    The hybrid is larger than the transformer-equivalent because:
    - Mamba-2 expand=2 means d_inner = 2*dim (8192), so in_proj + out_proj are ~2x attn.
    - DiffAttn has 2x Q and K projections + lambda params.
    - Extra SSM params: conv1d, A_log, D, dt_bias, inner norm.

    Breakdown:
    DiffAttn layer:
      wq: dim * 2*n_heads*head_dim
      wk: dim * 2*n_kv_heads*head_dim
      wv: dim * n_kv_heads*head_dim
      wo: n_heads*head_dim * dim
      subln: head_dim
      lambdas: 4 * n_heads * head_dim

    Mamba-2 layer:
      in_proj:  dim * (2*d_inner + 2*d_state + n_heads_m)
      conv1d:   d_inner * d_conv + d_inner (bias)
      out_proj: d_inner * dim
      A_log:    n_heads_m
      D:        d_inner
      dt_bias:  n_heads_m
      norm:     d_inner

    Shared per layer: SwiGLU FFN + 2 RMSNorm
    """
    d         = cfg.dim
    V         = cfg.vocab_size
    H         = cfg.n_heads
    Hk        = cfg.n_kv_heads
    hd        = d // H
    hidden    = cfg.hidden_dim
    L         = cfg.n_layers
    n_attn    = len(cfg.attn_layer_indices) if cfg.attn_layer_indices else 2
    n_ssm     = L - n_attn
    d_inner   = cfg.mamba_expand * d
    n_heads_m = d_inner // cfg.mamba_headdim
    d_state   = cfg.d_state
    d_conv    = cfg.d_conv

    diff_attn = (
        d * (2 * H  * hd) +
        d * (2 * Hk * hd) +
        d * (Hk * hd)     +
        (H * hd) * d      +
        hd                +
        4 * H * hd
    )
    in_feat = 2 * d_inner + 2 * d_state + n_heads_m
    mamba = (
        d * in_feat      +
        d_inner * d_conv +
        d_inner          +
        d_inner * d      +
        n_heads_m        +
        d_inner          +
        n_heads_m        +
        d_inner
    )
    per_layer_shared = d * hidden + hidden * d + d * hidden + 2 * d

    return (
        V * d +
        n_attn * (diff_attn + per_layer_shared) +
        n_ssm  * (mamba     + per_layer_shared) +
        d
    )


# ---------------------------------------------------------------------------
# Layer layout helper
# ---------------------------------------------------------------------------

def describe_layout(cfg: Orion7BConfig) -> str:
    lines = [f"Orion Atlas 7B -- {cfg.n_layers} layers:"]
    attn_set = set(cfg.attn_layer_indices or [])
    mamba_count = attn_count = 0
    for i in range(cfg.n_layers):
        kind = "DiffAttn" if i in attn_set else "Mamba-2"
        if i in attn_set:
            attn_count += 1
        else:
            mamba_count += 1
        lines.append(f"  [{i:02d}] {kind} + SwiGLU FFN")
    lines.append(f"\nSummary: {mamba_count} Mamba-2 + {attn_count} DiffAttn = {cfg.n_layers} total")
    lines.append(f"  ({mamba_count/cfg.n_layers*100:.1f}% SSM, {attn_count/cfg.n_layers*100:.1f}% Attention)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = Orion7BConfig()
    params = count_params_from_config_7b(cfg)
    print(f"Orion Atlas 7B Hybrid")
    print(f"  Params: {params:,}  ({params/1e9:.2f}B)")
    print()
    print(describe_layout(cfg))
