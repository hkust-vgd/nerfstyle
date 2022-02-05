from math import pi
import torch
from torch import nn
import einops


class Embedder(nn.Module):
    def __init__(self, num_freqs: int, append_input: bool = True):
        super(Embedder, self).__init__()
        self.num_freqs = num_freqs
        self.append_input = append_input
        self.register_buffer(
            'basis', 2. ** torch.linspace(0., num_freqs - 1, steps=num_freqs))

    def forward(self, x: torch.Tensor):
        out = einops.repeat(x, 'n k -> n k c', c=self.num_freqs)
        out = out * self.basis
        out = torch.stack([torch.sin(out), torch.cos(out)], dim=2)  # (n,k,2,c)
        out = einops.rearrange(out, 'n k f c -> n (k c f)')
        if self.append_input:
            out = torch.cat([out, x], dim=1)
        return out
