import torch
import tinycudann as tcnn

from common import TensorModule


pos_encoding_config = {
    'otype': 'HashGrid',  # TODO: replace with 'DenseGrid'
    'n_levels': 16,
    'n_features_per_level': 2,
    'log2_hashmap_size': 19,
    'base_resolution': 16,
    'per_level_scale': 2.0,
    'interpolation': 'Linear'
}

dir_encoding_config = {
    'otype': 'SphericalHarmonics',
    'degree': 4
}

density_net_config = {
    'otype': 'FullyFusedMLP',
    'activation': 'ReLU',
    'output_activation': 'None',
    'n_neurons': 64,
    'n_hidden_layers': 1
}

rgb_net_config = {
    'otype': 'FullyFusedMLP',
    'activation': 'ReLU',
    'output_activation': 'None',
    'n_neurons': 64,
    'n_hidden_layers': 2
}


class TCNerf(TensorModule):
    def __init__(self) -> None:
        super(TCNerf, self).__init__()

        self.n_output_dims = 16

        self.x_embedder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config=pos_encoding_config
        )

        self.d_embedder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config=dir_encoding_config
        )

        self.density_net = tcnn.Network(
            n_input_dims=self.x_embedder.n_output_dims,
            n_output_dims=self.n_output_dims,
            network_config=density_net_config
        )

        rgb_net_input_dims = self.density_net.n_output_dims + self.d_embedder.n_output_dims
        self.rgb_net = tcnn.Network(
            n_input_dims=rgb_net_input_dims,
            n_output_dims=3,
            network_config=rgb_net_config
        )

    def forward(self, pts, dirs):
        x_embedded = self.x_embedder(pts)
        d_embedded = self.d_embedder(dirs)

        density_output = self.density_net(x_embedded)
        rgb_input = torch.concat((density_output, d_embedded), axis=-1)
        rgb_output = self.rgb_net(rgb_input)

        return rgb_output, density_output[:, 0:1]
