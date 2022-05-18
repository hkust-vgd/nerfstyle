"""
Visuzalizes cross-sections of occupancy maps and which submodules are non-empty, i.e. will be
trained in distillation stage.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg

import __init__
from config import DatasetConfig

# Global constants
WIDTH = 12
HEIGHT = 8
UPDATE_RATE = 200


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_cfg')
    parser.add_argument('occ_map_path')
    args = parser.parse_args()

    # Load stuff
    cfg = DatasetConfig.load(args.dataset_cfg)
    grid = np.load(args.occ_map_path)['map']

    grid_dims = grid.shape
    net_dims = cfg.net_res
    assert np.all(np.array(net_dims) * 16 == np.array(grid_dims)), \
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

    threshold = 4096 * 0.01
    print('Networks to be trained:', np.sum(net_grid >= threshold))
    print('Networks to be skipped:', np.sum(net_grid < threshold))

    # Set up / initalize plot
    plt.rcParams['keymap.quit'].clear()  # disable matplotlib default quit keys
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT))

    def update(z):
        ax1.clear()
        ax1.imshow(grid[:, :, z])
        ax2.clear()
        ax2.imshow(net_grid[:, :, z // 16] > 10)

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
