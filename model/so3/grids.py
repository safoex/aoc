import numpy as np
from scipy.spatial.transform import Rotation


def fibonacci_sphere(samples=2):
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    i = np.linspace(0,samples, samples, endpoint=False)

    y = 1 - (i / (samples - 1)) * 2
    radius = np.sqrt(1 - y * y)
    theta = phi * i

    x = np.cos(theta) * radius
    z = np.sin(theta) * radius

    return np.stack((x, y, z), axis=1)


def fibonacci_sphere_rot(samples=2):
    pts = fibonacci_sphere(samples)
    azimuth = np.arctan2(pts[:,1], pts[:,0])
    azimuth[azimuth < 0] = azimuth[azimuth < 0] + 2.0 * np.pi

    # Elevation from (-0.5 * pi, 0.5 * pi)
    a = np.linalg.norm(pts, axis=1)
    b = np.linalg.norm(pts[:,:2], axis=1)
    elev = np.arccos(b / a)
    elev[pts[:, 2] < 0] = -elev[pts[:, 2] < 0]
    eulers = np.stack((azimuth, elev, np.zeros_like(azimuth)), axis=1)
    return Rotation.from_euler('xyz', eulers).as_matrix()


if __name__ == '__main__':
    print(fibonacci_sphere(4))
    print(fibonacci_sphere_rot(4))
