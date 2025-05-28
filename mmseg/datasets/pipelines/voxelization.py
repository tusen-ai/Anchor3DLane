import numpy as np

from ..builder import PIPELINES
from .voxel_generator import VoxelGenerator


@PIPELINES.register_module()
class Voxelization(object):
    def __init__(self, range, voxel_size, max_points_in_voxel, max_voxel_num, shuffle_points=False):
        self.range = range
        self.voxel_size = voxel_size
        self.max_points_in_voxel = max_points_in_voxel
        self.max_voxel_num = [max_voxel_num, max_voxel_num] if isinstance(max_voxel_num, int) else max_voxel_num
        self.shuffle_points = shuffle_points

        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num[0],
        )

    def __call__(self, results):
        voxel_size = self.voxel_generator.voxel_size
        pc_range = self.voxel_generator.point_cloud_range
        grid_size = self.voxel_generator.grid_size
        max_voxels = self.max_voxel_num[1]

        points = results['points']
        if self.shuffle_points:
            np.random.shuffle(points)

        voxels, coordinates, num_points = self.voxel_generator.generate(
            points, max_voxels=max_voxels 
        )
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

        results["voxels"] = dict(
            voxels=voxels,  # [N, 20, 5]
            coordinates=coordinates,  # [N, 3]
            num_points=num_points,  # 
            num_voxels=num_voxels,  # []
            shape=grid_size,
            range=pc_range,
            size=voxel_size
        )
        return results