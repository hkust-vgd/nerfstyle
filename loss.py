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
