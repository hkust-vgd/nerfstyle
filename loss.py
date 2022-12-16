from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, List


class GramStyleLoss(nn.Module):
    def __init__(self, keys: List[str]) -> None:
        super().__init__()
        self.keys = keys

    @staticmethod
    def _gram_mtx(feats: torch.Tensor):
        H, W = feats.shape[-2:]
        feats = rearrange(feats, pattern='n c h w -> n c (h w)')
        mtx = torch.matmul(feats, feats.transpose(-2, -1)) / (H * W)
        return mtx

    @torch.cuda.amp.autocast(dtype=torch.float32)
    def forward(
        self,
        feats1: Dict[str, torch.Tensor],
        feats2: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        matrices1 = [self._gram_mtx(feats1[k]) for k in self.keys]
        matrices2 = [self._gram_mtx(feats2[k]) for k in self.keys]
        losses = torch.stack([F.mse_loss(m1, m2) for m1, m2 in zip(matrices1, matrices2)])
        return torch.sum(losses)


class AdaINStyleLoss(nn.Module):
    def __init__(self, keys: List[str]) -> None:
        super().__init__()
        self.keys = keys

    @staticmethod
    def _mean(feats: torch.Tensor):
        return feats.mean(dim=(-2, -1))

    @staticmethod
    def _std(feats: torch.Tensor):
        return feats.var(dim=(-2, -1)).sqrt()

    @torch.cuda.amp.autocast(dtype=torch.float32)
    def forward(
        self,
        feats1: Dict[str, torch.Tensor],
        feats2: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        mean_l2 = [F.mse_loss(self._mean(feats1[k]), self._mean(feats2[k])) for k in self.keys]
        std_l2 = [F.mse_loss(self._std(feats1[k]), self._std(feats2[k])) for k in self.keys]
        losses = torch.stack([m + s for m, s in zip(mean_l2, std_l2)])
        return torch.sum(losses)


class MattingLaplacian(nn.Module):
    def __init__(
        self,
        device: torch.device,
        win_rad: int = 1,
        eps: float = 1E-7
    ) -> None:
        super().__init__()
        self.device = device
        self.win_rad = win_rad
        self.eps = eps

    def _compute_laplacian(
        self,
        image: torch.Tensor
    ) -> torch.Tensor:
        # Reference numpy implementation:
        # https://github.com/MarcoForte/closed-form-matting

        def eye(k):
            return torch.eye(k, device=self.device)

        win_size = (self.win_rad * 2 + 1) ** 2
        d, h, w = image.shape
        win_diam = self.win_rad * 2 + 1

        indsM = torch.arange(h * w, device=self.device).reshape((h, w))
        ravelImg = image.reshape((d, h * w)).T

        shape = (h - win_diam + 1, w - win_diam + 1) + (win_diam, win_diam)
        strides = indsM.stride() + indsM.stride()
        win_inds = torch.as_strided(indsM, shape, strides)
        win_inds = win_inds.reshape(-1, win_size)

        winI = ravelImg[win_inds]  # (P, K**2, 3)
        win_mu = torch.mean(winI, dim=1, keepdim=True)  # (P, 1, 3)
        win_var = torch.einsum('...ji,...jk ->...ik', winI, winI) / win_size - \
            torch.einsum('...ji,...jk ->...ik', win_mu, win_mu)  # (P, 3, 3)

        inv = torch.linalg.inv(win_var + (self.eps / win_size) * eye(3))
        X = torch.einsum('...ij,...jk->...ik', winI - win_mu, inv)
        vals = eye(win_size) - (1. / win_size) * \
            (1 + torch.einsum('...ij,...kj->...ik', X, winI - win_mu))

        nz_indsCol = torch.tile(win_inds, (win_size, )).ravel()
        nz_indsRow = torch.repeat_interleave(win_inds, win_size).ravel()
        nz_inds = torch.stack((nz_indsRow, nz_indsCol), dim=0)
        nz_indsVal = vals.ravel()
        L = torch.sparse_coo_tensor(indices=nz_inds, values=nz_indsVal, size=(h * w, h * w))
        return L

    def forward(
        self,
        target: torch.Tensor,
        style_map: torch.Tensor
    ) -> torch.Tensor:
        style_map = style_map.to(dtype=torch.float64)
        target = target.to(dtype=torch.float64)
        M = self._compute_laplacian(target)  # (HW, HW)
        V = style_map.reshape((3, -1))
        output = torch.trace(torch.mm(V, torch.sparse.mm(M, V.T)))
        return output
