import numpy as np
import torch
from torch.utils.data import Dataset
from mitmpose.model.so3 import fibonacci_sphere_rot
from mitmpose.model.so3 import ObjectRenderer
from tqdm import tqdm
import os


class RenderedDataset(Dataset):
    def __init__(self, size, res, model_path=None, grid_generator=fibonacci_sphere_rot):
        self.size = size
        self.res = res
        self.model_path = model_path
        self.inputs = None
        self.masks = None
        self.grid_generator = grid_generator

    def create_dataset(self, folder):
        if self.model_path is None:
            raise RuntimeError('you did not set model_path for rendering!')

        objren = ObjectRenderer(self.model_path)
        grid = self.grid_generator(self.size)

        if not os.path.exists(folder):
            os.makedirs(folder)

        self.inputs = np.memmap(folder + '/inputs.npy', dtype=np.float32, mode='w+', shape=(self.size, 3, self.res, self.res))
        self.masks = np.memmap(folder + '/masks.npy', dtype=np.uint8, mode='w+', shape=(self.size, 1, self.res, self.res))

        for i in tqdm(range(len(grid))):
            rot = grid[i]
            color, depth = objren.render_and_crop(rot, self.res)

            self.inputs[i, :, :, :] = np.moveaxis(color, 2, 0)
            self.masks[i, :, :, :] = np.reshape(depth > 0, (1, depth.shape[0], depth.shape[1]))

    def load_dataset(self, folder):
        self.inputs = np.memmap(folder + '/inputs.npy', dtype=np.float32, mode="r+", shape=(self.size, 3, self.res, self.res))
        self.masks = np.memmap(folder + '/masks.npy', dtype=np.uint8, mode='r+', shape=(self.size, 1, self.res, self.res))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.inputs[idx], self.masks[idx]


if __name__ == '__main__':
    fuze_path = '/home/safoex/Documents/libs/pyrender/examples/models/fuze.obj'
    ds = RenderedDataset(5000, 128, fuze_path)
    ds.create_dataset('test_save')