# Render offscreen -- make sure to set the PyOpenGL platform
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
import numpy as np
import trimesh
import pyrender
from PIL import Image
from mitmpose.model.pose.grids.grids import fibonacci_sphere_rot
import scipy

import torch
from torchvision import transforms


class ObjectRenderer:
    def __init__(self, path, camera_dist=0.5, res_side=640, intensity=(3,20), target_res=128, aae_scale_factor=1.2):
        tmesh = trimesh.load(path)
        self.mesh_size = scipy.linalg.norm(np.array(tmesh.bounding_box_oriented.extents))
        self.camera_dist_coefficient = 1.5
        mesh = pyrender.Mesh.from_trimesh(tmesh)
        self.scene = pyrender.Scene()
        self.node = pyrender.Node(mesh=mesh, matrix=np.eye(4))
        self.scene.add_node(self.node)

        self.camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        self.camera_dist = camera_dist or self.mesh_size * self.camera_dist_coefficient
        camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, self.camera_dist],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.cam_node = self.scene.add(self.camera, pose=camera_pose)

        # light = pyrender.SpotLight(color=np.ones(3), intensity=self.camera_dist * 500, innerConeAngle=np.pi / 16.0)
        # self.light = self.scene.add(light, pose=camera_pose)
        intensity_first = 20
        if isinstance(intensity, float) or isinstance(intensity, int):
            intensity_first = intensity

        dl = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=intensity_first)
        self.light = self.scene.add(dl)
        self.res_side = res_side
        self.target_res = target_res
        self.renderer = pyrender.OffscreenRenderer(self.res_side, self.res_side)

        self.directional_intensity = intensity

        self.aae_scale_factor = aae_scale_factor

    def set_camera_dist(self, camera_dist):
        camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, camera_dist],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.camera_dist = camera_dist
        self.scene.set_pose(self.cam_node, camera_pose)

        if isinstance(self.light, pyrender.SpotLight):
            light = pyrender.SpotLight(color=np.ones(3), intensity=self.camera_dist * 200, innerConeAngle=np.pi / 16.0)
            self.scene.remove_node(self.light)
            self.light = self.scene.add(light, pose=camera_pose)

    def find_optimal_camera_distance(self):
        N = 20
        grid = fibonacci_sphere_rot(N)
        sides = np.zeros((N, 2))
        while True:
            for i, rot in enumerate(grid):
                color, depth = self.render(rot)
                bbox = self.bbox(depth)
                sides[i, 0] = bbox[1] - bbox[0]
                sides[i, 1] = bbox[3] - bbox[2]

            print(np.max(sides), np.min(sides), np.median(sides))

            print(self.camera_dist)

            if np.any(sides[:, 0] + sides[:, 1] == 0):
                self.set_camera_dist(self.camera_dist * 2)
                continue
            if np.all(sides < self.res_side / 5):
                self.set_camera_dist(self.camera_dist * 1.2)
                continue
            if np.all(sides > 0) and np.any(sides > self.res_side / 3):
                self.set_camera_dist(self.camera_dist / 1.2)
                continue
            break


    def render(self, rot):
        pose = np.eye(4)
        pose[:3, :3] = rot

        if isinstance(self.directional_intensity, tuple):
            dif = self.directional_intensity[1] - self.directional_intensity[0]
            low = self.directional_intensity[0]
            self.light.light.intensity = np.random.random() * dif + low

        self.scene.set_pose(self.node, pose)
        return self.renderer.render(self.scene)

    @staticmethod
    def bbox(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)

        if np.all(rows == 0):
            rmin, rmax = 0, 0
        else:
            rmin, rmax = np.where(rows)[0][[0, -1]]

        if np.all(cols == 0):
            cmin, cmax = 0, 0
        else:
            cmin, cmax = np.where(cols)[0][[0, -1]]

        return rmin, rmax, cmin, cmax

    def crop_and_resize(self, color, bbox, target_res=None):
        target_res = target_res or self.target_res
        row_center = int(np.mean(bbox[:2]))
        col_center = int(np.mean(bbox[2:]))
        widest = max(bbox[1] - bbox[0], bbox[3] - bbox[2])
        half_side = int((widest * self.aae_scale_factor) / 2)
        left = row_center - half_side
        right = row_center + half_side
        top = col_center - half_side
        bottom = col_center + half_side

        final_box = (top, left, bottom, right)

        return np.array(Image.fromarray(color).crop(final_box).resize((target_res, target_res)), dtype=np.float32), \
               np.array(Image.fromarray((depth > 0).astype(np.uint8)).crop(final_box).resize((target_res, target_res)),
                        dtype=np.float32)

    def render_and_crop(self, rot, target_res=None):
        target_res = target_res or self.target_res
        color, depth = self.render(rot)
        bbox = self.bbox(depth)
        row_center = int(np.mean(bbox[:2]))
        col_center = int(np.mean(bbox[2:]))
        widest = max(bbox[1] - bbox[0], bbox[3] - bbox[2])
        half_side = int((widest * 1.2) / 2)
        left = row_center - half_side
        right = row_center + half_side
        top = col_center - half_side
        bottom = col_center + half_side

        # img = color.copy()
        #
        # img[:,top] = 0
        # img[:,bottom] = 0
        # img[:, int(col_center)] = 0
        # img[int(row_center), :] = 0
        # img[left,:] = 0
        # img[right,:] = 0

        final_box = (top, left, bottom, right)

        return np.array(Image.fromarray(color).crop(final_box).resize((target_res, target_res)), dtype=np.float32), \
               np.array(Image.fromarray((depth > 0).astype(np.uint8)).crop(final_box).resize((target_res, target_res)), dtype=np.float32)

    def test_show(self, color, depth):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.imshow(color)
        plt.subplot(1, 2, 2)
        plt.axis('off')
        plt.imshow(depth, cmap=plt.cm.gray_r)
        plt.show()


def cartesian_product(*arrays):
    ndim = len(arrays)
    return np.stack(np.meshgrid(*arrays), axis=-1).reshape(-1, ndim)


def sample_around(phi_0, phi_d, phi_n, r, z_0, z_d, z_n):
    phi = np.linspace(phi_0 - phi_d / 2, phi_0 + phi_d / 2, phi_n)
    z = np.linspace(z_0 - z_d / 2, z_0 + z_d / 2, z_n)
    phi_z = cartesian_product(phi, z)
    # print(phi_z)
    x_y_z = np.zeros((len(phi_z), 3))
    x_y_z[:, 0] = np.cos(phi_z[:, 0]) * r
    x_y_z[:, 1] = np.sin(phi_z[:, 0]) * r
    x_y_z[:, 2] = phi_z[:, 1]
    extra_rot = Rotation.from_euler('xyz', x_y_z).as_matrix()
    return phi_z, extra_rot

def render_and_save(objren, rot, path=None, rec_path=None, ae=None, target_res=128):
    img, _ = objren.render_and_crop(rot, target_res=target_res)
    img = np.moveaxis(img, 2, 0) / 255.0
    t_img = torch.tensor(img)
    if rec_path:
        t_img_rec = ae.forward(t_img[None, :, :, :].cuda()).cpu()
        transforms.ToPILImage()(t_img_rec[0, :, :, :]).convert("RGB").save(rec_path)
    if path:
        img = transforms.ToPILImage()(t_img).convert("RGB")
        img.save(path)


if __name__ == "__main__":
    from scipy.stats import special_ortho_group
    from scipy.spatial.transform import Rotation
    from tqdm import tqdm

    fuze_path = '/home/safoex/Downloads/cat_food/models_fixed/polpa.obj'
    fuze_path = '/home/safoex/Documents/libs/pyrender/examples/models/drill.obj'
    fuze_path = '/home/safoex/Documents/data/aae/models/scans/cleaner.obj'
    # fuze_path = '/home/safoex/Documents/data/aae/models/scans/tiramisu.obj'
    fuze_path = '/home/safoex/Documents/data/aae/models/textured.obj'
    # fuze_path = '/home/safoex/Downloads/006_mustard_bottle_berkeley_meshes/006_mustard_bottle/poisson/textured.obj'
    # fuze_path = '/home/safoex/Downloads/005_tomato_soup_can_berkeley_meshes/005_tomato_soup_can/poisson/textured.obj'

    models_dir = '/home/safoex/Documents/data/aae/models/scans/'
    model_name = 'meltacchin'
    model_path = models_dir + '%s.obj' % model_name
    objren = ObjectRenderer(model_path, None, 1024, intensity=(10, 10))
    # color, depth = objren.render(special_ortho_group.rvs(3))

    # rot = special_ortho_group.rvs(3)
    # phi_z, extra_rots = sample_around(0, 2*np.pi, 10, 5*np.pi/180, 0, 7*np.pi/180, 10)
    #
    # rots = np.zeros_like(extra_rots)
    # for i in range(extra_rots.shape[0]):
    #     rot_r = extra_rots[i].dot(rot)
    #     render_and_save(objren, rot_r, '/home/safoex/Documents/data/aae/test_rots/im%d.png'%i)

    N = 150
    eulers = np.zeros((N, 3))
    eulers[:, 1] = -0.3
    eulers[:, 0] = np.linspace(0, 2 * np.pi, N, endpoint=False)

    rots = Rotation.from_euler('xyz', eulers).as_matrix()
    draw_dir_root = '/home/safoex/Documents/data/aae/draw/'
    draw_dir = draw_dir_root + '%s/' % model_name

    if not os.path.exists(draw_dir_root):
        os.mkdir(draw_dir_root)
    if not os.path.exists(draw_dir):
        os.mkdir(draw_dir)

    # for i, rot in tqdm(enumerate(rots)):
    #     render_and_save(objren, rot, draw_dir + 'im%03d.png' % i, target_res=512)

    color, depth = objren.render(special_ortho_group.rvs(3))

    objren.test_show(color, depth)
