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


def fibonacci_sphere_in_plane_euler(samples_sphere=2, samples_in_plane=2):
    eulers_sphere = fibonacci_sphere_eulers(samples_sphere)
    z_in_plane = np.linspace(-np.pi, np.pi, samples_in_plane, endpoint=False)
    eulers = np.repeat(eulers_sphere, samples_in_plane, axis=0)
    zs = np.tile(z_in_plane, samples_sphere)
    eulers[:, 2] = zs
    return eulers


class Grid:
    def __init__(self, samples_sphere, samples_in_plane):
        self.samples_sphere = samples_sphere
        self.samples_in_plane = samples_in_plane
        self._grid = None
        self._index_grid_sphere = None
        self._index_grid_in_plane = None
        self.grid_divider = 2
        self._xy_step = None
        self.z_step = 2 * np.pi / samples_in_plane

    @property
    def grid(self):
        if self._grid is None:
            self._grid = Rotation.from_euler('xyz',
                                             fibonacci_sphere_in_plane_euler(self.samples_sphere, self.samples_in_plane)
                                             ).as_matrix()
        return self._grid

    def _min_distance(self):
        min_dist = np.zeros((self.grid.shape[0]))
        grid_euler = Rotation.from_matrix(self.grid).as_euler('xyz')
        for i, p in enumerate(grid_euler):
            d = np.linalg.norm(grid_euler - p, axis=1)
            d = d[d > 1e-4]
            min_dist[i] = np.min(d)
        return np.min(min_dist)

    @property
    def xy_step(self):
        if self._xy_step is None:
            self._xy_step = self._min_distance()
        return self._xy_step

    @property
    def index_grid(self):
        if self._index_grid_sphere is None:
            lx = int(2 * np.pi / self.xy_step)
            ly = int(2 * np.pi / self.xy_step)
            lz = self.samples_in_plane
            x = np.linspace(0, 2 * np.pi, lx)
            y = np.linspace(-np.pi, np.pi, ly)
            z = np.linspace(0, 2 * np.pi, lz)
            self._index_grid_sphere = np.zeros((lx, ly, lz), dtype=np.int16) - 1
            grid_euler_indices = self._index_on_index_grid(self.grid)
            gei = grid_euler_indices
            self._index_grid_sphere[gei[:, 0], gei[:, 1], gei[:, 2]] = np.arange(0, len(self.grid))
            d = 0
            any_unfilled = np.ones(len(self.grid), dtype=np.bool)
            while True:
                d += 1
                print(d)
                square_around = [(a, r) for r in range(-d, d + 1) for a in (-d, d)]
                square_around += [(b, a) for a, b in square_around]
                square_around = list(set(square_around))
                for idx in range(len(self.grid)):
                    if any_unfilled[idx]:
                        i, j, k = gei[idx, :]
                        any_filled = False
                        for di, dj in square_around:
                            i2, j2, = i + di, j + dj
                            if 0 <= i2 < lx and 0 <= j2 < ly:
                                if self._index_grid_sphere[i2, j2, k] == -1:
                                    any_filled = True
                                    self._index_grid_sphere[i2, j2, k] = self._index_grid_sphere[i, j, k]
                        if not any_filled:
                            any_unfilled[idx] = False
                if np.all(any_unfilled == False):
                    break
        return self._index_grid_sphere

    def _index_on_index_grid(self, rots):
        rots_euler = Rotation.from_matrix(rots).as_euler('xyz')
        rots_euler[:, :3] += np.pi
        rots_euler[:, :2] /= self.xy_step
        rots_euler[:, 2] /= self.z_step
        np.round_(rots_euler)
        return rots_euler.astype(np.int16)

    def nn_index(self, rots):
        indices = self._index_on_index_grid(rots)
        return self.index_grid[indices[:, 0], indices[:, 1], indices[:, 2]]


if __name__ == '__main__':
    import time

    # print(Rotation.from_euler('xyz', fse).as_euler('xyz'))
    # print(fibonacci_sphere_rot(4))
    g = Grid(4, 4)
    print(g.xy_step)
