from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = -100

SONG_TOK = "<|song|>"
END_TOK = "<|end|>"
PAD_TOK = "<|pad|>"
SPECIALS = [SONG_TOK, END_TOK, PAD_TOK]


@dataclass
class Config:
    block_size: int = 128
    batch_size: int = 32
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    head_dim: int | None = None
    hidden_dim: int | None = None
    dropout: float = 0.2

    lr: float = 1e-3
    min_lr: float = 1e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    z_loss_weight: float = 0.0

    final_epochs: int = 150
    patience: int = 8
    seed: int = 1337
    compile_model: bool = False

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0
        self.head_dim = self.n_embd // self.n_head
        if self.hidden_dim is None:
            self.hidden_dim = int(round(8 * self.n_embd / 3 / 64) * 64)
        self.hidden_dim = int(self.hidden_dim)

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))   # no bias, no mean subtraction
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.g

def rotate(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1*cos - x2*sin,
                        x1*sin + x2*cos), dim=-1).flatten(-2)

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head, self.n_embd = cfg.n_head, cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv  = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_pdrop = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.q_norm = RMSNorm(cfg.head_dim)   # or L2-normalize with a learnable scale
        self.k_norm = RMSNorm(cfg.head_dim)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, cache=None, start_pos=0):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)          # norm BEFORE rotation

        t = torch.arange(start_pos, start_pos + T,
                        device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos = freqs.cos()[None, None].to(q.dtype)
        sin = freqs.sin()[None, None].to(q.dtype)
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)

        if cache is not None:                          # append to past
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        new_cache = (k, v)

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_pdrop if self.training else 0.0,
            is_causal=(T == k.size(2)),                # ← invariant B
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y)), new_cache

class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg.hidden_dim                      # ≈ 8/3 · n_embd, rounded
        self.w_gate = nn.Linear(cfg.n_embd, h, bias=False)
        self.w_up   = nn.Linear(cfg.n_embd, h, bias=False)
        self.w_down = nn.Linear(h, cfg.n_embd, bias=False)
        self.drop   = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.rmsnorm1, self.attn = RMSNorm(cfg.n_embd), CausalSelfAttention(cfg)
        self.rmsnorm2, self.swiglu  = RMSNorm(cfg.n_embd), SwiGLU(cfg)

    def forward(self, x, cache=None, start_pos=0):
        a, new_cache = self.attn(self.rmsnorm1(x), cache, start_pos)
        x = x + a
        x = x + self.swiglu(self.rmsnorm2(x))
        return x, new_cache

class TinyTransformer(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.cfg, self.vocab_size = cfg, vocab_size
        self.tok_emb = nn.Embedding(vocab_size, self.cfg.n_embd)
        self.drop = nn.Dropout(self.cfg.dropout)
        self.blocks = nn.ModuleList([Block(self.cfg) for _ in range(self.cfg.n_layer)])
        self.rmsnorm = RMSNorm(self.cfg.n_embd)
        self.head = nn.Linear(self.cfg.n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight            # weight tying
        self.apply(self._init_weights)
        self._scale_residual_projections()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
              nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self):
      s = 1.0 / math.sqrt(2 * self.cfg.n_layer)
      for b in self.blocks:
          b.attn.proj.weight.data.mul_(s)
          b.swiglu.w_down.weight.data.mul_(s)

    def forward(self, idx, targets=None, caches=None, start_pos=0):
        B, T = idx.shape
        assert start_pos + T <= self.cfg.block_size, "context overflow"
        if caches is not None:
            assert not self.training, "KV cache requires eval mode"
            assert len(caches) == self.cfg.n_layer

        x = self.drop(self.tok_emb(idx))
        for i, blk in enumerate(self.blocks):
            x, new_c = blk(x, None if caches is None else caches[i], start_pos)
            if caches is not None:
                caches[i] = new_c

        logits = self.head(self.rmsnorm(x))
        loss = None
        if targets is not None:
          loss = F.cross_entropy(
              logits.reshape(-1, self.vocab_size),
              targets.reshape(-1),
              ignore_index=cfg.ignore_index,
          )
          if self.cfg.z_loss_weight > 0:
            logZ = torch.logsumexp(logits.float(), dim=-1)
            loss = loss + self.cfg.z_loss_weight * logZ.pow(2).mean()
        return logits, loss

print("model classes ready")