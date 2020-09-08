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

    def __len__(self):
        return len(self.grid)

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
            lx = int(2 * np.pi / self.xy_step) + 1
            ly = int(2 * np.pi / self.xy_step) + 1
            lz = self.samples_in_plane + 1
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
        rots_euler += np.pi
        if len(rots.shape) > 2:
            rots_euler[:, :2] /= self.xy_step
            rots_euler[:, 2] /= self.z_step
        else:
            rots_euler[:2] /= self.xy_step
            rots_euler[2] /= self.z_step
        np.round_(rots_euler)
        return rots_euler.astype(np.int16)

    def nn_index(self, rots):
        indices = self._index_on_index_grid(rots)
        if len(indices.shape) > 1:
            return self.index_grid[indices[:, 0], indices[:, 1], indices[:, 2]]
        else:
            return self.index_grid[indices[0], indices[1], indices[2]]


import itertools


class GradUniformGrid(Grid):
    def __init__(self, samples_x, samples_y, samples_in_plane,
                 range_x=(-np.pi, np.pi), range_y=(-np.pi / 2, np.pi / 2), range_z=(-np.pi, np.pi), eulers_order='xyz', extra_rot=None):
        super().__init__(samples_x * samples_y, samples_in_plane)
        self.samples_x = samples_x
        self.samples_y = samples_y
        self._x_step = 2 * np.pi / samples_x
        self._y_step = np.pi / samples_y
        self.lx = np.linspace(range_x[0], range_x[1], self.samples_x)
        self.ly = np.linspace(range_y[0], range_y[1], self.samples_y)
        self.lz = np.linspace(range_z[0], range_z[1], self.samples_in_plane)
        self._interpolator = None

        self.extra_rot = extra_rot or Rotation.identity()

        self.eulers_order = eulers_order
        if self.eulers_order == 'yxz':
            self.lx, self.ly = self.ly, self.lx
            self.samples_x, self.samples_y = self.samples_y, self.samples_x
            self._x_step, self._y_step = self._y_step, self._x_step

    @property
    def grid(self):
        if self._grid is None:
            self._grid = (self.extra_rot * Rotation.from_euler(self.eulers_order,
                                             list(itertools.product(self.lx, self.ly, self.lz))
                                             )).as_matrix()
        return self._grid

    def make_interpolator(self, codebook):
        idcs = itertools.product(range(self.samples_x), range(self.samples_y), range(self.samples_in_plane))
        codebook3d = np.zeros((self.samples_x, self.samples_y, self.samples_in_plane, codebook.shape[1]),
                              dtype=np.float32)
        codebook_cpu = codebook.cpu()
        for i, (ix, iy, iz) in enumerate(idcs):
            codebook3d[ix, iy, iz, :] = codebook_cpu[i]
        self._interpolator = RegularGridInterpolator((self.lx, self.ly, self.lz), codebook3d, method='linear')

        def interp(rots):
            rrots = self.extra_rot.inv() * Rotation.from_matrix(rots)
            eulers = rrots.as_euler(self.eulers_order)
            return self._interpolator(eulers)

        return interp


class AxisSwapGrid(Grid):
    def __init__(self, samples_x, samples_y, samples_in_plane, delta_y=np.pi / 12):
        super().__init__(samples_x * samples_y, samples_in_plane)

        assert samples_y % 2 == 0
        self.grids = [GradUniformGrid(samples_x, samples_y // 2, samples_in_plane,
                                      range_y=(-np.pi / 4 - delta_y, np.pi / 4 + delta_y), extra_rot=erot)
                      for erot in (None, Rotation.from_euler('xyz', [np.pi/2, 0, 0]))]

        self._grid = None

    @property
    def grid(self):
        if self._grid is None:
            self._grid = np.concatenate((self.grids[0].grid, self.grids[1].grid))

        return self._grid

    def make_interpolator(self, codebook):
        half = len(self) // 2
        interpolators = [grid.make_interpolator(codebook[i * half: (i + 1) * half, :])
                         for i, grid in enumerate(self.grids)]

        def interp(rots):
            eulers = Rotation.from_matrix(rots).as_euler('xyz')
            if len(rots.shape) > 2:
                res = np.ones((rots.shape[0], codebook.shape[1]))
                mask_xyz = np.abs(eulers[:, 1]) <= np.pi / 4
                mask_yxz = np.logical_not(mask_xyz)
                if np.sum(mask_xyz) > 0:
                    res[mask_xyz] = interpolators[0](rots[mask_xyz])
                if np.sum(mask_yxz) > 0:
                    res[mask_yxz] = interpolators[1](rots[mask_yxz])
                return res
            else:
                if np.abs(eulers[1]) <= np.pi / 4:
                    return interpolators[0](rots)
                else:
                    return interpolators[1](rots)

        return interp


if __name__ == '__main__':
    import time

    # print(Rotation.from_euler('xyz', fse).as_euler('xyz'))
    # print(fibonacci_sphere_rot(4))
    # g = Grid(4, 4)
    # print(g.xy_step)
    ug = GradUniformGrid(3, 3, 3)
    print(ug.grid)
