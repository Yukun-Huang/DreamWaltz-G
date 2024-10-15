import numpy as np


class BasicPointCloud:
    def __init__(self, points=None, colors=None, normals=None, alphas=None) -> None:
        self.points: np.array = points if points else np.empty((0, 3))
        self.colors: np.array = colors if points else np.empty((0, 3))
        self.normals: np.array = normals if points else np.empty((0, 3))
        self.alphas: np.array = alphas if points else np.empty((0, 1))
    
    def __repr__(self) -> str:
        return f'Point cloud object with {len(self.points)} points.'

    def __len__(self):
        return self.points.shape[0]


def fetchPly(path):
    from plyfile import PlyData
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz=None, rgb=None, point_cloud:BasicPointCloud=None):
    from plyfile import PlyData, PlyElement
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    if xyz is None:
        xyz = point_cloud.points
    if rgb is None:
        rgb = point_cloud.colors
    
    normal = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normal, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
