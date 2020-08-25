
# Render offscreen -- make sure to set the PyOpenGL platform
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import numpy as np
import trimesh
import pyrender
from PIL import Image

class ObjectRenderer:
    def __init__(self, path, camera_dist=0.5):
        mesh = pyrender.Mesh.from_trimesh(trimesh.load(path))
        self.scene = pyrender.Scene()
        self.node = pyrender.Node(mesh=mesh, matrix=np.eye(4))
        self.scene.add_node(self.node)

        self.camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, camera_dist],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.scene.add(self.camera, pose=camera_pose)

        light = pyrender.SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle=np.pi/16.0)
        self.scene.add(light, pose=camera_pose)

        self.renderer = pyrender.OffscreenRenderer(640, 640)

    def render(self, rot):
        pose = np.eye(4)
        pose[:3, :3] = rot
        self.scene.set_pose(self.node, pose)
        return self.renderer.render(self.scene)

    @staticmethod
    def bbox(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return rmin, rmax, cmin, cmax

    def render_and_crop(self, rot, target_res=128):
        color, depth = self.render(rot)
        bbox = self.bbox(depth)
        row_center = int(np.mean(bbox[:2]))
        col_center = int(np.mean(bbox[2:]))
        widest = max(bbox[1]-bbox[0], bbox[3] - bbox[2])
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

        img = Image.fromarray(color)

        crop = img.crop((top, left, bottom, right))
        crop = crop.resize((target_res, target_res))
        return np.array(crop), depth

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

    fuze_path = '/home/safoex/Documents/libs/pyrender/examples/models/fuze.obj'
    objren = ObjectRenderer(fuze_path, 0.5)
    # color, depth = objren.render(special_ortho_group.rvs(3))
    color, depth = objren.render_and_crop(special_ortho_group.rvs(3))
    # Show the images
    import  time
    start = time.clock()
    for i in range(1000):
        color, depth = objren.render_and_crop(special_ortho_group.rvs(3))
    fin = time.clock()
    print(fin - start)