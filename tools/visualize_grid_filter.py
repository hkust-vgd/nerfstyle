"""
Visuzalizes cross-sections of occupancy maps and which submodules are non-empty, i.e. will be
trained in distillation stage.
"""

import argparse
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg

import __init__
from config import DatasetConfig

# Global constants
WIDTH = 7.5
HEIGHT = 9
UPDATE_RATE = 200
VOXEL_SIZE = 16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_cfg')
    parser.add_argument('occ_map_path')
    parser.add_argument('--sparsity-thres', type=float, default=0.0025)
    args = parser.parse_args()

    # Load stuff
    cfg = DatasetConfig.load(args.dataset_cfg)
    grid = np.load(args.occ_map_path)['map']

    grid_dims = grid.shape
    net_dims = cfg.net_res
    assert np.all(np.array(net_dims) * VOXEL_SIZE == np.array(grid_dims)), \
        'config and occ_map dims not match'
    x_coords = np.linspace(0, grid_dims[0], net_dims[0]+1).astype(int)
    y_coords = np.linspace(0, grid_dims[1], net_dims[1]+1).astype(int)
    z_coords = np.linspace(0, grid_dims[2], net_dims[2]+1).astype(int)

    indices = [
        (i, j, k)
        for i in range(net_dims[0])
        for j in range(net_dims[1])
        for k in range(net_dims[2])
    ]
    net_grid = np.zeros(net_dims)

    for i, j, k in indices:
        x1, x2 = x_coords[i], x_coords[i+1]
        y1, y2 = y_coords[j], y_coords[j+1]
        z1, z2 = z_coords[k], z_coords[k+1]
        net_grid[i, j, k] = np.sum(grid[x1:x2, y1:y2, z1:z2])

    threshold = 4096 * args.sparsity_thres
    print('Networks to be trained:', np.sum(net_grid >= threshold))
    print('Networks to be skipped:', np.sum(net_grid < threshold))

    # Set up / initalize plot
    plt.rcParams['keymap.quit'].clear()  # disable matplotlib default quit keys
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(WIDTH, HEIGHT))
    fig.tight_layout()

    bg_color = (1., 1., 1.)
    occ_color = (0.7, 0.7, 0.7)
    grid_color = (0.4, 0.4, 0.4)
    cmap = mcolors.ListedColormap([bg_color, occ_color], name='custom_cmap')

    def update(z):
        ax1.clear()
        ax1.imshow(grid[:, :, z], vmin=0., vmax=1., cmap=cmap)
        ax1.grid(color=grid_color, which='minor')
        ax1.set_xticks(np.arange(grid_dims[1], step=VOXEL_SIZE) + VOXEL_SIZE // 2)
        ax1.set_xticklabels(np.arange(net_dims[1]))
        ax1.set_xticks(np.arange(grid_dims[1], step=VOXEL_SIZE)[1:], minor=True)
        ax1.set_yticks(np.arange(grid_dims[0], step=VOXEL_SIZE) + VOXEL_SIZE // 2)
        ax1.set_yticklabels(np.arange(net_dims[0]))
        ax1.set_yticks(np.arange(grid_dims[0], step=VOXEL_SIZE)[1:], minor=True)
        ax1.tick_params(axis='both', which='minor', length=0)

        ax2.clear()
        ax2.imshow(net_grid[:, :, z // VOXEL_SIZE] > threshold, cmap=cmap, vmin=0., vmax=1.)
        ax2.grid(color=grid_color, which='minor')
        ax2.set_xticks(np.arange(net_dims[1]))
        ax2.set_xticks(np.arange(net_dims[1]+1)[1:] - 0.5, minor=True)
        ax2.set_yticks(np.arange(net_dims[0]))
        ax2.set_yticks(np.arange(net_dims[0]+1)[1:] - 0.5, minor=True)
        ax2.tick_params(axis='both', which='minor', length=0)

    update(0)  # Initial plot

    # Set up GUI
    sg.theme('LightGrey')
    dpi = fig.get_dpi()
    layout = [
        [sg.Canvas(size=(WIDTH * dpi, HEIGHT * dpi), key='canvas')],
        [sg.Slider(range=(0, grid_dims[2]-1), orientation='h',
                   size=(100, 20), default_value=0, key='z_value')]
    ]
    title = 'Cross section of "{}"'.format(args.occ_map_path)
    window = sg.Window(
        title, layout, finalize=True, resizable=True,
        element_justification='center', return_keyboard_events=True)
    window.bind("<Escape>", "-ESCAPE-")

    canvas = FigureCanvasTkAgg(fig, window['canvas'].TKCanvas)
    canvas.draw()
    canvas.get_tk_widget().pack(side='top', fill='both', expand=1)

    # Main loop
    prev_z_value = 0
    while True:
        event, values = window.read(timeout=UPDATE_RATE)
        if event in (sg.WIN_CLOSED, "-ESCAPE-"):
            break

        z_value = int(values['z_value'])
        if prev_z_value != z_value:
            prev_z_value = z_value
            update(z_value)
            canvas.draw()

    window.close()


if __name__ == '__main__':
    main()
