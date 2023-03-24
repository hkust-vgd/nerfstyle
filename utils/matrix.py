'''
Utility functions for matrix operations.
'''

import numpy as np

CANON_CAM_MTX = 'rdf'
CANON_WORLD_MTX = 'rfu'

# Directional vectors in canonical representation.
# Positive X/Y/Z corresponds to right / front / up.
coord_vectors = {
    'r': (1, 0, 0),
    'l': (-1, 0, 0),
    'f': (0, 1, 0),
    'b': (0, -1, 0),
    'u': (0, 0, 1),
    'd': (0, 0, -1)
}


def get_canonical_coord_mtx(coord_str: str):
    try:
        assert len(coord_str) == 3
        coord_mtx = np.array([coord_vectors[c] for c in coord_str.lower()]).T
        assert np.linalg.det(coord_mtx) == 1
    except:
        raise ValueError('Invalid coordinate system "{}"'.format(coord_str))

    return coord_mtx


def convert_poses(poses, w_coord, c_coord):
    # R'p + t' = B(RAp + t)
    # R' = BRA, t' = Bt

    can_cam_mtx = get_canonical_coord_mtx(CANON_CAM_MTX)
    dat_cam_mtx = get_canonical_coord_mtx(c_coord)

    can_world_mtx = get_canonical_coord_mtx(CANON_WORLD_MTX)
    dat_world_mtx = get_canonical_coord_mtx(w_coord)

    tf1 = np.matmul(can_cam_mtx.T, dat_cam_mtx)
    tf2 = np.matmul(dat_world_mtx.T, can_world_mtx)

    new_poses = np.copy(poses)
    new_poses[:, :3, :3] = np.matmul(tf2, np.matmul(poses[:, :3, :3], tf1))
    new_poses[:, :3, 3:4] = np.matmul(tf2, poses[:, :3, 3:4])

    return new_poses
