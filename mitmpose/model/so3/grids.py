import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import RegularGridInterpolator


def fibonacci_sphere(samples=2):
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    i = np.linspace(0, samples, samples, endpoint=False)

    y = 1 - (i / (samples - 1)) * 2
    radius = np.sqrt(1 - y * y)
    theta = phi * i

    x = np.cos(theta) * radius
    z = np.sin(theta) * radius

    return np.stack((x, y, z), axis=1)

def fibonacci_sphere_eulers(samples=2):
    pts = fibonacci_sphere(samples)
    azimuth = np.arctan2(pts[:, 1], pts[:, 0])
    azimuth[azimuth < 0] = azimuth[azimuth < 0] + 2.0 * np.pi

    # Elevation from (-0.5 * pi, 0.5 * pi)
    a = np.linalg.norm(pts, axis=1)
    b = np.linalg.norm(pts[:, :2], axis=1)
    elev = np.arccos(b / a)
    elev[pts[:, 2] < 0] = -elev[pts[:, 2] < 0]
    eulers = np.stack((azimuth, elev, np.zeros_like(azimuth)), axis=1)
    return eulers

def fibonacci_sphere_rot(samples=2):
    return Rotation.from_euler('xyz', fibonacci_sphere_eulers(samples)).as_matrix()

def in_plane_rot(samples=2):
    eulers = np.zeros((samples, 3))
    eulers[:, 2] = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    return Rotation.from_euler('xyz', eulers).as_matrix()

def fibonacci_sphere_euler_in_plane(samples_sphere=2, samples_in_plane=2):
    grid_sphere = fibonacci_sphere_rot(samples_sphere)
    grid_in_plane = in_plane_rot(samples_in_plane)

    return np.array([gs * gp for gs in grid_sphere for gp in grid_in_plane])


class Grid:
    def __init__(self, samples_sphere, samples_in_plane):
        self.samples_sphere = samples_sphere
        self.samples_in_plane = samples_in_plane
        self._grid = None
        self._index_grid_sphere = None
        self._index_grid_in_plane = None
        self.grid_divider = 2

    @property
    def grid(self):
        if self._grid is None:
            grid_sphere = fibonacci_sphere_rot(self.samples_sphere)
            grid_in_plane = in_plane_rot(self.samples_in_plane)

            self._grid = np.array([gs * gp for gs in grid_sphere for gp in grid_in_plane])
        return self._grid

    def _min_distance(self):
        min_dist = np.zeros((self.grid.shape[0]))
        grid_euler = Rotation.from_matrix(self.grid).as_euler('xyz')
        for i, p in enumerate(grid_euler):
            d = np.linalg.norm(grid_euler - p, axis=1)
            d = d[d > 1e-4]
            min_dist[i] = np.min(d)
        return np.min(min_dist)

    def make_index_grid(self):
        self._index_grid_in_plane = np.zeros((self.samples_sphere, self.samples_in_plane), dtype=np.int16)
        xy_step = self._min_distance() / 2
        x = np.linspace(-np.pi / 2, np.pi / 2, int( np.pi / xy_step) + 1)
        y = np.linspace(0, 2 * np.pi , int(2 * np.pi / xy_step) + 1)
        z = np.linspace(0, 2 * np.pi, self.samples_in_plane + 1)
        grid_euler = Rotation.from_matrix(self.grid).as_euler('xyz')
        def nearest(px, py, pz):
            pass
        # self._index_grid_sphere =

    def index(self, rots):
        pass

if __name__ == '__main__':
    import time
    # print(fibonacci_sphere(4))
    # print(fibonacci_sphere_rot(4))
    grid = Rotation.from_matrix(fibonacci_sphere_rot(3000)).as_euler('xyz')
    st = time.time()
    min_dist = np.zeros((grid.shape[0]))
    for i, p in enumerate(grid):
        d = np.linalg.norm(grid - p, axis=1)
        d = d[d > 1e-4]
        # print(np.mean(d), np.median(d), np.max(d), np.min(d))
        min_dist[i] = np.min(d)

    print(np.min(min_dist))
    print(time.time() - st)
    st = time.time()
    g2d = np.tile(grid, (grid.shape[0], 1, 1))
    g2d_diff = np.linalg.norm(g2d - np.transpose(g2d, (1, 0, 2)), axis=2)
    print(np.min(g2d_diff[g2d_diff > 1e-4]))
    print(time.time() - st)