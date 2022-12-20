from functools import partial
from typing import Optional

from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F

from networks.fx import VGG16FeatureExtractor


class Upsample(nn.Module):
    def __init__(self, scale_factor):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor)


class ScaledDotProductAttn(nn.Module):
    def __init__(
        self,
        temperature: float
    ) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, v, mask):
        # (N_q, D) * (D, N_k) -> (N_q, N_k)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.temperature

        if mask is not None:
            attn = attn * mask

        # Apply softmax over keys dimension
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_head: int,  # No. of heads
        key_dim: int,
        val_dim: int,
        source_dim: int,
        cross_dim: Optional[int] = None,
        add_residual: bool = True,
        actv: str = 'none'
    ) -> None:
        super().__init__()
        if cross_dim is None:
            cross_dim = source_dim  # self attention

        self.n_head = n_head
        self.key_dim = key_dim
        self.val_dim = val_dim
        self.add_residual = add_residual

        self.wq = nn.Linear(source_dim, n_head * key_dim, bias=False)
        self.wk = nn.Linear(cross_dim, n_head * key_dim, bias=False)
        self.wv = nn.Linear(cross_dim, n_head * val_dim, bias=False)

        self.attn_fn = ScaledDotProductAttn(temperature=key_dim ** 0.5)
        self.fc = nn.Linear(n_head * val_dim, source_dim, bias=False)

        norm_layers = nn.ModuleDict({
            'none': nn.Identity(),
            'instance': nn.InstanceNorm2d(source_dim, eps=1e-6),
            'layer': nn.LayerNorm(source_dim, eps=1e-6)
        })
        assert actv in norm_layers.keys(), 'Invalid activation choice "{}"'.format(actv)
        self.actv = norm_layers[actv]

        self.expand_head = partial(rearrange, pattern='b n (h d) -> b h n d', h=n_head)
        self.contract_head = partial(rearrange, pattern='b h n d -> b n (h d)')

    # query:    (B, N_s, D_s)
    # key:      (B, N_c, D_c)
    # value:    (B, N_c, D_c)
    # mask:     (B, N_s, N_c)
    def forward(self, q, k, v, mask=None):
        residual = q

        q = self.expand_head(self.wq(q), d=self.key_dim)  # (B, H, N_s, D_k)
        k = self.expand_head(self.wk(k), d=self.key_dim)  # (B, H, N_c, D_k)
        v = self.expand_head(self.wv(v), d=self.val_dim)  # (B, H, N_c, D_v)

        if mask is not None:
            # (B, N_s, N_c) -> (B, H, N_s, N_c)
            mask = mask.unsqueeze(1)

        q = self.contract_head(self.attn_fn(q, k, v, mask))  # (B, N_s, H*D_v)
        q = self.fc(q)  # (B, N_s, D_s)

        if self.add_residual:
            q = q + residual
        return self.actv(q)


class BasicAttention(MultiHeadAttention):
    def __init__(
        self,
        key_dim: int,
        val_dim: int,
        source_dim: int,
        cross_dim: Optional[int] = None,
        add_residual: bool = True,
        actv: str = 'none'
    ) -> None:
        super().__init__(1, key_dim, val_dim, source_dim, cross_dim, add_residual, actv)
        self.expand_head = lambda x, d: x
        self.contract_head = lambda x: x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pad_size: int = 1,
        use_relu: bool = True
    ) -> None:
        super().__init__()
        self.refl_pad = nn.ReflectionPad2d((pad_size, pad_size, pad_size, pad_size))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.use_relu = use_relu

    def forward(self, x):
        x = self.refl_pad(x)
        out = self.conv(x)

        if self.use_relu:
            out = F.relu(out)

        return out


class PyramidEncoder(nn.Module):
    def __init__(self, enc_channels) -> None:
        super().__init__()

        downsample = nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)

        self.blocks = nn.ModuleDict({
            '1': nn.Sequential(
                ConvBlock(enc_channels, 64),
                ConvBlock(64, 64)
            ),
            '2': nn.Sequential(
                downsample,
                ConvBlock(64, 128),
                ConvBlock(128, 128)
            ),
            '3': nn.Sequential(
                downsample,
                ConvBlock(128, 256),
                ConvBlock(256, 256),
                ConvBlock(256, 256),
                ConvBlock(256, 256)
            ),
            '4': nn.Sequential(
                downsample,
                ConvBlock(256, 512),
                ConvBlock(512, 512),
                ConvBlock(512, 512),
                ConvBlock(512, 512)
            ),
            '5': nn.Sequential(
                downsample,
                ConvBlock(512, 512),
                ConvBlock(512, 512),
                ConvBlock(512, 512),
                ConvBlock(512, 512)
            )
        })

    def forward(self, x):
        out_dict = {0: x}
        for i in range(1, 6):
            out_dict[i] = self.blocks[str(i)](out_dict[i-1])
        return out_dict


class PyramidDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.upsample = Upsample(scale_factor=2)

        self.blocks = nn.ModuleDict({
            '4': nn.Sequential(
                ConvBlock(1024, 512),
                ConvBlock(512, 256)
            ),
            '3': nn.Sequential(
                ConvBlock(512, 256),
                ConvBlock(256, 256),
                ConvBlock(256, 256),
                ConvBlock(256, 128)
            ),
            '2': nn.Sequential(
                ConvBlock(256, 128),
                ConvBlock(128, 64)
            ),
            '1': nn.Sequential(
                ConvBlock(128, 64)
            )
        })

    def forward(self, enc_feats, feats5, feats4):
        x = self.upsample(feats5) + feats4
        for i in range(4, 0, -1):
            # print(enc_feats[i].shape, x.shape)
            x = torch.concat((enc_feats[i], x), dim=1)
            x = self.blocks[str(i)](x)
            if i != 1:
                x = self.upsample(x)
        return x


class NLCInstanceNorm(nn.Module):
    """
    Same as non-affine nn.InstanceNorm1d, but with layer and channel dimensions swapped.
    """
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, feats):
        feats_mean = feats.mean(dim=1, keepdim=True)
        feats_var = feats.var(dim=1, keepdim=True) + self.eps
        norm_feats = (feats - feats_mean) / feats_var.sqrt()
        return norm_feats


class AttentionPyramid(nn.Module):
    def __init__(self, input_channels, h, w) -> None:
        super().__init__()
        self.style_fx = VGG16FeatureExtractor(['relu4_1', 'relu5_1'])
        self.feats_encoder = PyramidEncoder(input_channels)
        self.decoder = PyramidDecoder()
        self.attn4 = BasicAttention(key_dim=512, val_dim=512, source_dim=512, add_residual=False)
        self.attn5 = BasicAttention(key_dim=512, val_dim=512, source_dim=512, add_residual=False)

        h4, h5 = h // 8, h // 16
        w4, w5 = w // 8, w // 16
        self.flatten = partial(rearrange, pattern='b c h w -> b (h w) c')
        self.unflatten5 = partial(rearrange, pattern='b (h w) c -> b c h w', h=h5, w=w5)
        self.unflatten4 = partial(rearrange, pattern='b (h w) c -> b c h w', h=h4, w=w4)
        self.norm = NLCInstanceNorm(eps=1e-6)

    # pix_feats: (F, H, W)
    # style_imgs: (N, 3, H, W)
    def forward(self, pix_feats, style_imgs):
        pix_feats = torch.tile(pix_feats, dims=(len(style_imgs), 1, 1, 1))
        enc_feats = self.feats_encoder(pix_feats)
        style_feats = self.style_fx(style_imgs)

        enc_feats4 = self.flatten(enc_feats[4])
        enc_feats5 = self.flatten(enc_feats[5])
        style_feats4 = self.flatten(style_feats['relu4_1'])
        style_feats5 = self.flatten(style_feats['relu5_1'])

        attn_feats5 = self.unflatten5(self.attn5(
            self.norm(enc_feats5), self.norm(style_feats5), style_feats5))
        attn_feats4 = self.unflatten4(self.attn4(
            self.norm(enc_feats4), self.norm(style_feats4), style_feats4))
        print(enc_feats5.dtype, style_feats5.dtype)

        out = self.decoder(enc_feats, attn_feats5, attn_feats4)
        return out
