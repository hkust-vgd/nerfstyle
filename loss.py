import torch
from torch import nn
import torch.nn.functional as F
from torchtyping import TensorType, patch_typeguard
import torchvision
from typeguard import typechecked

patch_typeguard()


@typechecked
def gram_mtx(
    feats: TensorType['C', 'H', 'W']
):
    c, h, w = feats.shape
    flat_feats = feats.view(c, h * w)
    gram_mtx = torch.mm(flat_feats, flat_feats.t())
    return gram_mtx.div(h * w)


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        net_type: str = 'vgg19'
    ) -> None:
        super().__init__()

        if net_type == 'vgg16':
            net_fn = torchvision.models.vgg16
            extract_layers = [3, 8, 15, 22, 29]
        elif net_type == 'vgg19':
            net_fn = torchvision.models.vgg19
            extract_layers = [3, 8, 17, 26, 35]
        else:
            raise NotImplementedError('Unrecognized net type "{}"'.format(net_type))

        net = net_fn(pretrained=True).features.eval()
        self.net = torchvision.models.feature_extraction.create_feature_extractor(
            net, return_nodes={str(k): f'layer{i}' for i, k in enumerate(extract_layers)})

        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406])[:, None, None])
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225])[:, None, None])

    @typechecked
    def forward(
        self,
        x: TensorType[3, 'H', 'W'],
        detach: bool = False
    ) -> dict:
        x_norm = (x - self.mean) / self.std
        feats_dict = self.net(x_norm.unsqueeze(0))
        for k in feats_dict.keys():
            feats_dict[k] = feats_dict[k].squeeze(0)
            if detach:
                feats_dict[k] = feats_dict[k].detach()

        return feats_dict


class StyleLoss(nn.Module):
    def __init__(
        self,
        target_feats: dict
    ) -> None:
        super().__init__()
        self.keys = list(target_feats.keys())
        self.matrices = {k: gram_mtx(v) for k, v in target_feats.items()}

    def forward(
        self,
        feats: dict
    ) -> TensorType[()]:
        assert list(feats.keys()) == self.keys
        matrices = {k: gram_mtx(v) for k, v in feats.items()}
        losses = torch.stack([F.mse_loss(self.matrices[k], matrices[k]) for k in self.keys])
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
        image: TensorType[3, 'H', 'W']
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

    @typechecked
    def forward(
        self,
        target: TensorType[3, 'H', 'W'],
        style_map: TensorType[3, 'H', 'W']
    ) -> TensorType[()]:
        M = self._compute_laplacian(target)  # (HW, HW)
        V = style_map.reshape((3, -1))
        output = torch.trace(torch.mm(V, torch.sparse.mm(M, V.T)))
        return output
