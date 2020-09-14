# Render offscreen -- make sure to set the PyOpenGL platform
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
import numpy as np
import trimesh
import pyrender
from PIL import Image
from mitmpose.model.so3.grids import fibonacci_sphere_rot


class ObjectRenderer:
    def __init__(self, path, camera_dist=0.5, res_side=640):
        mesh = pyrender.Mesh.from_trimesh(trimesh.load(path))
        self.scene = pyrender.Scene()
        self.node = pyrender.Node(mesh=mesh, matrix=np.eye(4))
        self.scene.add_node(self.node)

        self.camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        self.camera_dist = camera_dist
        camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, camera_dist],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.cam_node = self.scene.add(self.camera, pose=camera_pose)

        # light = pyrender.SpotLight(color=np.ones(3), intensity=self.camera_dist * 500, innerConeAngle=np.pi / 16.0)
        # self.light = self.scene.add(light, pose=camera_pose)
        dl = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
        self.light = self.scene.add(dl)
        self.res_side = res_side
        self.renderer = pyrender.OffscreenRenderer(self.res_side, self.res_side)

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

    def render_and_crop(self, rot, target_res=128):
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

        return [
            np.array(Image.fromarray(img).crop(final_box).resize((target_res, target_res)), dtype=np.float32)
            for img in (color, depth)
        ]

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


if __name__ == "__main__":
    from scipy.stats import special_ortho_group

    fuze_path = '/home/safoex/Downloads/t-less_v2_models_reconst/models_reconst/obj_04.ply'
    objren = ObjectRenderer(fuze_path, 240, 1280)
    # color, depth = objren.render(special_ortho_group.rvs(3))
    color, depth = objren.render(special_ortho_group.rvs(3))
    # Show the images
    import time

    start = time.clock()
    for i in range(10):
        color, depth = objren.render(special_ortho_group.rvs(3))
        # print(objren.bbox(depth))
    # objren.find_optimal_camera_distance()
    fin = time.clock()
    print(fin - start)
    objren.test_show(color, depth)
