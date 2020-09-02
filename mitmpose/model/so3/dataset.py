import numpy as np
import torch
from torch.utils.data import Dataset
from mitmpose.model.so3 import fibonacci_sphere_rot
from mitmpose.model.so3 import ObjectRenderer
from tqdm import tqdm
import os


class RenderedDataset(Dataset):
    def __init__(self, size, res, model_path=None, grid_generator=fibonacci_sphere_rot, camera_dist=0.5, render_res=640):
        self.size = size
        self.res = res
        self.model_path = model_path
        self.inputs = None
        self.masks = None
        self.rots = None
        self.grid_generator = grid_generator
        self.camera_dist = camera_dist
        self.render_res = render_res

    def create_memmaps(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

        self.inputs = np.memmap(folder + '/inputs.npy', dtype=np.float32, mode='w+', shape=(self.size, 3, self.res, self.res))
        self.masks = np.memmap(folder + '/masks.npy', dtype=np.uint8, mode='w+', shape=(self.size, 1, self.res, self.res))

    def create_dataset(self, folder):
        if self.model_path is None:
            raise RuntimeError('you did not set model_path for rendering!')

        objren = ObjectRenderer(self.model_path, res_side=self.render_res, camera_dist=self.camera_dist)
        grid = self.grid_generator(self.size)

        self.create_memmaps(folder)

        self.rots = grid.astype(np.float32)
        np.save(folder + '/rots.npy', self.rots)


        for i in tqdm(range(len(grid))):
            rot = grid[i]
            color, depth = objren.render_and_crop(rot, self.res)

            self.inputs[i, :, :, :] = np.moveaxis(color, 2, 0) / 255.0
            self.masks[i, :, :, :] = np.reshape(depth > 0, (1, depth.shape[0], depth.shape[1]))

    def load_dataset(self, folder):
        self.inputs = torch.from_numpy(np.memmap(folder + '/inputs.npy', dtype=np.float32, mode="r+", shape=(self.size, 3, self.res, self.res)))
        self.masks = torch.from_numpy(np.memmap(folder + '/masks.npy', dtype=np.uint8, mode='r+', shape=(self.size, 1, self.res, self.res)))
        self.rots = torch.from_numpy(np.load(folder + '/rots.npy'))

    def to_torch(self):
        self.inputs = torch.from_numpy(self.inputs)
        self.masks = torch.from_numpy(self.masks)
        self.rots = torch.from_numpy(self.rots)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if not isinstance(self.inputs, torch.Tensor):
            self.inputs = torch.from_numpy(self.inputs)
            self.masks = torch.from_numpy(self.masks)

        return self.inputs[idx], self.masks[idx], self.rots[idx]


if __name__ == '__main__':
    fuze_path = '/home/safoex/Documents/libs/pyrender/examples/models/fuze.obj'
    ds = RenderedDataset(500, 128, fuze_path)
    ds.create_dataset('test_save')