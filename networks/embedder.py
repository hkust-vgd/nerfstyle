import torch
from torch import nn
import einops
import utils


class Embedder(nn.Module):
    def __init__(self, num_freqs: int, append_input: bool = True):
        super(Embedder, self).__init__()
        in_channels = 3
        self._out_channels = 2 * in_channels * num_freqs
        if append_input:
            self._out_channels += in_channels

        self.num_freqs = num_freqs
        self.append_input = append_input
        self.register_buffer(
            'basis', 2. ** torch.linspace(0., num_freqs - 1, steps=num_freqs))

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x: torch.Tensor):
        # B: batch size
        # K: no. of channels (= 3)
        # C: no. of frequencies
        # F: no. of functions (= 2)
        out = einops.repeat(x, 'b k -> b k c', c=self.num_freqs)
        out = out * self.basis
        out = einops.rearrange([torch.sin(out), torch.cos(out)],
                               'f b k c -> b (k c f)')
        if self.append_input:
            out = torch.cat([out, x], dim=1)
        return out

    def __repr__(self) -> str:
        attrs = ['num_freqs', 'append_input', 'out_channels']
        return utils.get_repr(self, attrs)


class MultiEmbedder(Embedder):
    def __init__(self, num_freqs: int, append_input: bool = True):
        super().__init__(num_freqs, append_input)

    def forward(self, x: torch.Tensor):
        # N: no. of networks
        out = einops.repeat(x, 'n b k -> n b k c', c=self.num_freqs)
        out = out * self.basis
        out = einops.rearrange([torch.sin(out), torch.cos(out)],
                               'f n b k c -> n b (k c f)')
        if self.append_input:
            out = torch.cat([out, x], dim=2)
        return out
