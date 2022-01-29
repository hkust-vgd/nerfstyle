import torch


class RayBatch:
    def __init__(self, origin, dests, near, far):
        self.origin = origin
        self.dests = dests
        self.near = near
        self.far = far

    def __len__(self):
        return len(self.dests)

    def viewdirs(self):
        norms = torch.norm(self.dests, dim=-1, keepdim=True)
        return self.dests / norms

    def lerp(self, coeffs):
        assert len(coeffs) == len(self)
        out = torch.einsum('nk, nc -> nkc', coeffs, self.dests) + self.origin
        return out
