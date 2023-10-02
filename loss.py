import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

from einops import rearrange
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F


def compute_centroid(mask):
    H, W = mask.shape
    N = torch.sum(mask)
    r_indices, c_indices = torch.arange(H, device=mask.device), torch.arange(W, device=mask.device)
    r_mean = torch.sum(torch.sum(mask, dim=1) * r_indices) / N / H
    c_mean = torch.sum(torch.sum(mask, dim=0) * c_indices) / N / W
    return torch.stack((r_mean, c_mean))


def labels_downscale(labels, new_dim):
    H, W = labels.shape
    NH, NW = new_dim
    r_indices = torch.linspace(0, H-1, NH).long()
    c_indices = torch.linspace(0, W-1, NW).long()
    return labels[r_indices[:, None], c_indices]


# (N1, C), (N2, C)
def cosine_dists(feats1, feats2):
    feats1_hat = feats1 / torch.linalg.norm(feats1, dim=1)[:, None]
    feats2_hat = feats2 / torch.linalg.norm(feats2, dim=1)[:, None]
    dists = 1.0 - torch.matmul(feats1_hat, feats2_hat.T)
    return dists


class StyleLoss(nn.Module):
    def __init__(self, keys: List[str]) -> None:
        super().__init__()
        self.keys = keys


class GramStyleLoss(StyleLoss):
    def __init__(self, keys: List[str]) -> None:
        super().__init__(keys)

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


class AdaINStyleLoss(StyleLoss):
    def __init__(self, keys: List[str]) -> None:
        super().__init__(keys)

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


class NNFMStyleLoss(StyleLoss):
    def __init__(self, keys: List[str]) -> None:
        super().__init__(keys)

    def forward(
        self,
        feats1: Dict[str, torch.Tensor],
        feats2: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        loss = 0
        for k in self.keys:
            feat1, feat2 = feats1[k].squeeze(0), feats2[k].squeeze(0)
            assert len(feat1) == len(feat2)
            feat1 = feat1.reshape((len(feat1), -1))  # (C, H*W)
            feat2 = feat2.reshape((len(feat2), -1))  # (C, H*W)

            feat1_hats = feat1 / torch.linalg.norm(feat1, dim=0)
            feat2_hats = feat2 / torch.linalg.norm(feat2, dim=0)
            min_dists = torch.amin(1.0 - torch.matmul(feat1_hats.T, feat2_hats), dim=1)
            loss += torch.mean(min_dists)
        return loss


class SemanticStyleLoss(StyleLoss):
    def __init__(
        self,
        keys: List[str],
        clusters_path: Path,
        matching: Optional[List[int]] = None
    ) -> None:
        super().__init__(keys)
        self.ready = False
        self.clusters = None
        self.matching = None
        self.use_matching = False

        if clusters_path is not None:
            self.use_matching = True

            self.clusters = np.load(str(clusters_path))['seg_map']
            clusters_list = np.unique(self.clusters)
            if clusters_list[0] < 0:
                clusters_list = clusters_list[1:]

            self.n_clusters = len(clusters_list)
            assert np.all(np.arange(self.n_clusters) == clusters_list)
            self.clusters = torch.tensor(self.clusters).cuda()
            self.matching = matching

    def _debug_matching(self):
        for i, j in enumerate(self.matching):
            mask = (self.clusters == j).detach().cpu().numpy()
            plt.imsave('tmp/c{:d}_s{:d}.png'.format(i, j), mask)

    @torch.no_grad()
    def init_feats(self, all_style_feats, num_classes):
        style_feats = all_style_feats[self.keys[0]].squeeze(0)
        self.style_feats = style_feats

        if not self.use_matching:
            self.ready = True
            return

        size1, size2 = style_feats.shape[1:], self.clusters.shape
        assert size1 == size2, \
            'Style features {} and style clusters {} ' \
            'are not same size'.format(tuple(size1), tuple(size2))

        self.style_feats_mean = torch.stack([
            torch.mean(style_feats[:, self.clusters == i], dim=1) for i in range(self.n_clusters)
        ])
        self.style_centroids = torch.stack([
            compute_centroid(self.clusters == i) for i in range(self.n_clusters)
        ])

        self.num_classes = num_classes
        self.ready = True

    def update_matching(self, image_feats, preds):
        preds_small = labels_downscale(preds, image_feats.shape[-2:])
        image_mean_feats = torch.stack([
            torch.mean(image_feats[:, preds_small == i], dim=1) for i in range(self.num_classes)
        ])
        image_centroids = torch.stack([
            compute_centroid(preds == i) for i in range(self.num_classes)
        ])

        feat_dists = cosine_dists(image_mean_feats, self.style_feats_mean)
        patch_dists = torch.linalg.norm(
            image_centroids[:, None] - self.style_centroids[None], dim=-1)
        cost_mtx = (feat_dists + patch_dists).detach().cpu().numpy()
        cost_mtx = np.nan_to_num(cost_mtx)
        self.matching = linear_sum_assignment(cost_mtx)[1]
        print(self.matching)
        # self._debug_matching()

    def forward(
        self,
        feats1: Dict[str, torch.Tensor],
        _,
        preds: torch.Tensor,
        iter: int
    ) -> torch.Tensor:
        assert self.ready
        image_feat = feats1[self.keys[0]].squeeze(0)
        if self.use_matching and self.matching is None:
            self.update_matching(image_feat, preds)

        preds_small = labels_downscale(preds, image_feat.shape[-2:])

        image_feat_nc = rearrange(image_feat, 'c h w -> (h w) c')
        style_feat_nc = rearrange(self.style_feats, 'c h w -> (h w) c')
        dists = cosine_dists(image_feat_nc, style_feat_nc)

        if self.use_matching and iter <= 150:
            for i in range(self.num_classes):
                image_mask = (preds_small == i).reshape(-1)
                style_mask = (self.clusters != self.matching[i]).reshape(-1)
                invalid_mask = torch.logical_and(*torch.meshgrid(image_mask, style_mask))
                dists[invalid_mask] = float('inf')

        min_dists = torch.amin(dists, dim=1)
        loss = torch.mean(min_dists)
        return loss


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


def get_style_loss(
    loss_name: str,
    keys: Union[List[str], str],
    **kwargs
) -> StyleLoss:
    module_ctor = getattr(sys.modules[__name__], loss_name)
    assert issubclass(module_ctor, StyleLoss)
    if isinstance(keys, str):
        return module_ctor([keys], **kwargs)
    return module_ctor(keys, **kwargs)
