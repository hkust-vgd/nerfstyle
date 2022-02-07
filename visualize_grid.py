import sys
import numpy as np
import open3d as o3d


def main():
    grid = np.load(sys.argv[1])['map']
    height, width, depth = grid.shape
    pts = [[h, w, d]
           for h in range(height)
           for w in range(width)
           for d in range(depth)
           if grid[h, w, d]]
    pts = np.array(pts)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd, voxel_size=0.5)

    o3d.visualization.draw_geometries([voxel_grid])


if __name__ == '__main__':
    main()
