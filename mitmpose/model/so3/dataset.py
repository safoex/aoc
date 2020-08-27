import numpy as np
import torch
from torch.utils.data import Dataset
from mitmpose.model.so3 import fibonacci_sphere_rot
from mitmpose.model.so3 import ObjectRenderer


class RenderedDataset(Dataset):
    def __init__(self, model_path=None, grid_generator=fibonacci_sphere_rot, size=50000, res=128):
        self.size = size
        self.res = res
        self.model_path = model_path
        self.inputs = None
        self.masks = None
        self.grid_generator = grid_generator

    def render_dataset(self):
        if self.model_path is None:
            raise RuntimeError('you did not set model_path for rendering!')

        objren = ObjectRenderer(self.model_path)
        grid = self.grid_generator(self.size)
        self.inputs = np.zeros((self.size, 3, self.res, self.res))
        self.masks = np.zeros((self.size, 1, self.res, self.res))

        for i, rot in enumerate(grid):
            color, depth = objren.render_and_crop(rot, self.res)

            self.inputs[i, :, :, :] = np.moveaxis(color, 2, 0)
            self.masks[i, :, :, :] = np.reshape(depth > 0, (1, depth.shape[0], depth.shape[1]))

    def save_dataset(self, folder_path):
        inputs_path = folder_path + '/inputs.npy'
        masks_path = folder_path + '/masks.npy'
        np.save(inputs_path, self.inputs)
        np.save(masks_path, self.masks)

    def load_dataset(self, folder_path):
        inputs_path = folder_path + '/inputs.npy'
        masks_path = folder_path + '/masks.npy'
        self.inputs = np.load(inputs_path).astype(dtype=np.float32) / 255.0
        self.masks = np.load(masks_path).astype(dtype=np.uint8)
        self.size = self.inputs.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.inputs[idx], self.masks[idx]


if __name__ == '__main__':
    fuze_path = '/home/safoex/Documents/libs/pyrender/examples/models/fuze.obj'
    ds = RenderedDataset(fuze_path, fibonacci_sphere_rot, 5000)
    ds.render_dataset()
    ds.save_dataset('test_save')
