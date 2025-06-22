import torch
import torch.nn as nn
import torch.nn.functional as F
import rff.layers as rff


class CoordEncoder(nn.Module):
    def __init__(
        self,
        fourier_sigma: float = 10.0,
        fourier_m: int = 16,
        coord_dim: int = 3,        # Δx, Δy, Δθ
        hidden_dim: int = 1024,
        output_dim: int = 512,
        depth: int = 6,
        use_layernorm: bool = True,
    ):
        super().__init__()
        # Fourier-features
        self.fourier = rff.PositionalEncoding(sigma=fourier_sigma, m=fourier_m)
        fourier_dim = coord_dim * 2 * fourier_m
        
        # Input hosszabbítás: Fourier + nyers coord
        self.input_dim = fourier_dim + coord_dim
        self.blocks = nn.ModuleList()
        for i in range(depth):
            in_dim = self.input_dim if i == 0 else hidden_dim
            lin = nn.Linear(in_dim, hidden_dim)
            norm = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
            act = nn.GELU()
            # Residual kapcsolat: blokk input + output
            block = nn.ModuleDict({'lin': lin, 'norm': norm, 'act': act})
            self.blocks.append(block)
        
        self.final = nn.Linear(hidden_dim, output_dim)

    def forward(self, coord: torch.Tensor):
        # coord: [batch, 3] – relatív x, y, θ (normalizált)
        x_fourier = self.fourier(coord)
        h = torch.cat([x_fourier, coord], dim=-1)
        
        for block in self.blocks:
            y = block['lin'](h)
            y = block['norm'](y)
            y = block['act'](y)
            # Residual kapcsolat (skip): h lehet input vagy hidden
            if y.shape == h.shape:
                h = h + y
            else:
                h = y  # csak első réteg esetén, ha dim változik
        return self.final(h)