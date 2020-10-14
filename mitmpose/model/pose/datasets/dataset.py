import numpy as np
import torch
from torch.utils.data import Dataset
from mitmpose.model.pose.grids.grids import Grid
from mitmpose.model.pose import ObjectRenderer
from tqdm.auto import tqdm
import os


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        items = self.dataset.__getitem__(idx)
        return items, idx


class OnlineRenderDataset(Dataset):
    def __init__(self, grider: Grid, model_path=None, augmenter=None, res=128, camera_dist=None, render_res=640,
                 directional_light_intensity=5):
        self.size = grider.samples_in_plane * grider.samples_sphere
        self.res = res
        self.model_path = model_path
        self.grider = grider
        self.camera_dist = camera_dist
        self.render_res = render_res
        self._objren = None
        self.directional_light_intensity = directional_light_intensity
        self.augmenter = augmenter

    @property
    def objren(self):
        if self._objren is None:
            self._objren = ObjectRenderer(self.model_path, res_side=self.render_res, camera_dist=self.camera_dist,
                                          intensity=self.directional_light_intensity)
        return self._objren

    def create_dataset(self, workdir):
        pass

    def load_dataset(self, workdir):
        pass

    def load_or_create_dataset(self, workdir):
        pass

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, list):
            return [self.__getitem__(i) for i in idx]
        else:
            rot = self.grider.grid[idx]
            color, depth = self.objren.render_and_crop(rot, self.res)

            img = np.moveaxis(color, 2, 0) / 255.0
            mask = np.reshape(depth > 0, (1, depth.shape[0], depth.shape[1]))

            if self.augmenter is not None:
                img = self.augmenter((img, mask, rot))

            return img, mask, rot

    def test_show(self, img, mask):
        color = np.moveaxis(img, 0, 2)
        binary_depth = np.reshape(mask, (mask.shape[1], mask.shape[2]))
        self.objren.test_show(color, binary_depth)


class RenderedDataset(OnlineRenderDataset):
    def __init__(self, grider: Grid, model_path=None, augmenter=None, res=128, camera_dist=None, render_res=640,
                 directional_light_intensity=5, image_path='inputs.npy', masks_path='masks.npy', rots_path='rots.npy'):
        super().__init__(grider, model_path, augmenter, res, camera_dist, render_res, directional_light_intensity)
        self.inputs = None
        self.masks = None
        self.rots = None
        self.image_path = image_path
        self.masks_path = masks_path
        self.rots_path = rots_path

    def create_memmaps(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

        if self.image_path:
            self.inputs = np.memmap(folder + '/' + self.image_path, dtype=np.float32, mode='w+',
                                    shape=(self.size, 3, self.res, self.res))

        if self.masks_path:
            self.masks = np.memmap(folder + '/' + self.masks_path, dtype=np.uint8, mode='w+',
                                   shape=(self.size, 1, self.res, self.res))

    def create_dataset(self, folder):
        if self.model_path is None:
            raise RuntimeError('you did not set model_path for rendering!')

        grid = self.grider.grid

        self.create_memmaps(folder)

        self.rots = grid.astype(np.float32)

        if self.rots_path:
            np.save(folder + '/' + self.rots_path, self.rots)

        for i in tqdm(range(len(grid))):
            img, mask, _ = super().__getitem__(i)

            if self.image_path:
                self.inputs[i, :, :, :] = img

            if self.masks_path:
                self.masks[i, :, :, :] = mask

        self.to_torch()

    def load_dataset(self, folder):
        if self.image_path:
            self.inputs = torch.from_numpy(np.memmap(folder + '/' + self.image_path, dtype=np.float32, mode="r+",
                                                 shape=(self.size, 3, self.res, self.res)))
        if self.masks_path:
            self.masks = torch.from_numpy(np.memmap(folder + '/' + self.masks_path, dtype=np.uint8, mode='r+',
                                                shape=(self.size, 1, self.res, self.res)))
        if self.rots_path:
            self.rots = torch.from_numpy(np.load(folder + '/' + self.rots_path))
        else:
            self.rots = self.grider.grid.astype(np.float32)

    def to_torch(self):
        if self.inputs is not None:
            self.inputs = torch.from_numpy(self.inputs)
        if self.masks is not None:
            self.masks = torch.from_numpy(self.masks)
        if self.rots is not None:
            self.rots = torch.from_numpy(self.rots)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.inputs[idx] if self.inputs is not None  else None
        mask = self.masks[idx] if self.masks is not None else None
        rot = self.rots[idx] if self.rots is not None else None

        return img, mask, rot


class AugmentedAndRenderedDataset(Dataset):
    def __init__(self, grider: Grid, model_path=None, augmenter=None, res=128, render_res=640, camera_dist=None,
                 intensity_reconstruction=10, intensity_augment=(2, 20),
                 aug_class=RenderedDataset, rec_class=RenderedDataset):
        self.aug_images_path = None
        self.rec_images_path = None
        if aug_class is RenderedDataset:
            self.aug_images_path = 'images_augmented.npy'
        if rec_class is RenderedDataset:
            self.rec_images_path = 'images.npy'

        self.augmented_dataset = aug_class(grider, model_path, augmenter, res, camera_dist, render_res,
                                           intensity_augment, self.aug_images_path, None, None)
        self.reconstruction_dataset = rec_class(grider, model_path, None, res, camera_dist, render_res,
                                                intensity_reconstruction, self.rec_images_path)

    def create_dataset(self, workdir):
        self.reconstruction_dataset.create_dataset(workdir)
        self.augmented_dataset.create_dataset(workdir)

    def load_dataset(self, workdir):
        self.reconstruction_dataset.load_dataset(workdir)
        self.augmented_dataset.load_dataset(workdir)

    def __len__(self):
        return len(self.augmented_dataset)

    def __getitem__(self, idx):
        img_aug, _, _ = self.augmented_dataset[idx]
        img_rec, mask, rot = self.reconstruction_dataset[idx]
        return img_rec, mask, rot, img_aug


if __name__ == '__main__':
    fuze_path = '/home/safoex/Documents/libs/pyrender/examples/models/fuze.obj'
    grider = Grid(10, 10)
    ds = RenderedDataset(grider, fuze_path, image_path='inputs_augmented.npy', masks_path=None, rots_path=None)
    ds.create_dataset('test_save2')
