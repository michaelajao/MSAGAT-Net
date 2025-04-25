import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from itertools import product

# Core Hyperparameters (modifiable via args or grid)
DEFAULTS = {
    'hidden_dim': 32,
    'attention_heads': 4,
    'dropout': 0.355,
    'num_temporal_scales': 2,
    'kernel_size': 3,
    'feature_channels': 16,
    'bottleneck_dim': 8,
}

class SpatialAttentionModuleLite(nn.Module):
    """
    Single-projection spatial attention with shared low-rank bias.
    """
    def __init__(self, hidden_dim, num_nodes, heads, bottleneck_dim, dropout):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.num_nodes = num_nodes
        self.bottleneck_dim = bottleneck_dim

        # Single linear for QKV
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Shared low-rank bias factors
        self.u = nn.Parameter(torch.Tensor(num_nodes, bottleneck_dim))
        self.v = nn.Parameter(torch.Tensor(bottleneck_dim, num_nodes))
        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.v)

    def forward(self, x):
        B, N, H = x.size()
        # QKV
        qkv = self.qkv(x).view(B, N, 3, self.heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # ELU stabilization
        q, k = F.elu(q) + 1, F.elu(k) + 1
        kv = torch.einsum('bhnd,bhne->bhde', k, v)
        ones = torch.ones(B, self.heads, N, 1, device=x.device)
        z = 1.0 / (torch.einsum('bhnd,bhno->bhn', k, ones) + 1e-8)
        attn_out = torch.einsum('bhnd,bhde,bhn->bhne', q, kv, z)

        # Scaled dot-product + bias
        bias = (self.u @ self.v).unsqueeze(0).unsqueeze(0)
        scores = torch.einsum('bhnd,bhmd->bhnm', q, k) / math.sqrt(self.head_dim)
        attn = F.softmax(scores + bias, dim=-1)
        out = torch.einsum('bhnm,bhme->bhne', attn, v)
        out = out.permute(0,2,1,3).reshape(B, N, H)
        out = self.out_proj(out)
        return out

class TemporalModuleLite(nn.Module):
    """
    Multi-scale dilated temporal conv with share and fuse.
    """
    def __init__(self, hidden_dim, num_scales, kernel_size, dropout):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size,
                      padding=(kernel_size//2)*(2**i), dilation=2**i)
            for i in range(num_scales)
        ])
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: [B, N, H]
        t = x.transpose(1,2)  # [B, H, N]
        outs = []
        for conv in self.convs:
            y = conv(t)
            y = self.bn(y)
            y = self.act(y)
            y = self.dropout(y)
            outs.append(y)
        fused = torch.stack(outs,0).mean(0)
        fused = fused.transpose(1,2)
        return self.layer_norm(fused + x)

class HorizonPredictorLite(nn.Module):
    """
    Simple MLP predictor (no gating).
    """
    def __init__(self, hidden_dim, horizon, bottleneck_dim, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, horizon)
        )

    def forward(self, x, last_step=None):
        return self.mlp(x)

class MSTAGATNetLite(nn.Module):
    """
    Lightweight MSTAGAT-Net for ablation and profiling.
    """
    def __init__(self, args, data,
                 use_bias=True, num_scales=None,
                 use_predictor_gate=False, hp=DEFAULTS):
        super().__init__()
        m, W, H = data.m, args.window, args.horizon
        # feature extractor
        self.fe = nn.Sequential(
            nn.Conv1d(1, hp['feature_channels'], hp['kernel_size'], padding=hp['kernel_size']//2),
            nn.BatchNorm1d(hp['feature_channels']), nn.ReLU(), nn.Dropout(hp['dropout'])
        )
        self.proj = nn.Sequential(
            nn.Linear(hp['feature_channels']*W, hp['bottleneck_dim']), nn.ReLU(),
            nn.Linear(hp['bottleneck_dim'], hp['hidden_dim'])
        )
        # modules with configurable options
        self.spatial = SpatialAttentionModuleLite(
            hp['hidden_dim'], m, hp['attention_heads'], hp['bottleneck_dim'], hp['dropout']
        )
        self.temporal = TemporalModuleLite(
            hp['hidden_dim'], num_scales or hp['num_temporal_scales'], hp['kernel_size'], hp['dropout']
        )
        # optionally swap predictor
        self.predictor = HaloPredictor = (
            HorizonPredictorLite(hp['hidden_dim'], H, hp['bottleneck_dim'], hp['dropout'])
        )

    def forward(self, x, idx=None):
        B, T, N = x.shape
        x0 = x.permute(0,2,1).reshape(B*N,1,T)
        f = self.fe(x0).view(B,N,-1)
        f = self.proj(f)
        s = self.spatial(f)
        t = self.temporal(s)
        p = self.predictor(t)
        return p.transpose(1,2)

# === Profiling Utility ===
def profile_model(model, window, nodes, batch=1):
    """Prints model summary via torchinfo"""
    summary(model, input_size=(batch, window, nodes))

# === Ablation Study Script ===
def ablation_study(args, data, device='cpu'):
    """
    Runs variants over use_bias and num_scales
    """
    results = {}
    grid = list(product([True, False], [1,2,4]))
    for use_bias, scales in grid:
        key = f"bias_{use_bias}_scales_{scales}"
        model = MSTAGATNetLite(args, data, use_bias=use_bias, num_scales=scales).to(device)
        # here you’d train & eval; placeholder:
        results[key] = {'params': sum(p.numel() for p in model.parameters()),
                        'flops': None}
        print(f"Variant {key}: params={results[key]['params']}")
    return results

# === Hyperparameter Tuning Skeleton ===
def hyperparameter_search(args, data, device='cpu'):
    """
    Simple grid search over selected hyperparameters
    """
    hp_grid = {
        'hidden_dim': [16,32,64],
        'attention_heads': [2,4,8],
        'num_temporal_scales': [1,2]
    }
    keys, values = zip(*hp_grid.items())
    best = {'score': float('-inf'), 'config': None}
    for combo in map(dict, [dict(zip(keys, v)) for v in product(*values)]):
        hp = DEFAULTS.copy(); hp.update(combo)
        model = MSTAGATNetLite(args, data, hp=hp).to(device)
        # train & validate; placeholder score:
        score = torch.rand(1).item()
        if score > best['score']:
            best = {'score': score, 'config': combo}
    print("Best config", best)
    return best
