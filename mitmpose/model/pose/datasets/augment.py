import numpy as np
import torch
from mitmpose.model.pose.grids.grids import Grid

from mitmpose.model.pose.datasets.dataset import RenderedDataset, OnlineRenderDataset

from imgaug import augmenters as iaa
from torchvision.datasets import VOCDetection
from torchvision import transforms
from tqdm.auto import tqdm
from joblib import Parallel, delayed, parallel_backend


def print_batch(x, x_hat, save_path, side=4):
    img_tensor_inputs = [torch.cat([x[i, :, :, :].cpu() for i in range(j * side, (j + 1) * side)], 1) for j
                         in range(2)]
    img_tensor_outputs = [torch.cat([x_hat[i, :, :, :].cpu() for i in range(j * side, (j + 1) * side)], 1)
                          for j in range(2)]

    img_tensor = torch.cat(img_tensor_inputs + img_tensor_outputs, 2)
    im = transforms.ToPILImage()(img_tensor).convert("RGB")
    im.save(save_path)


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


class ImgAugTransform:
    def __init__(self, prob=0.5):
        self.aug = iaa.Sequential([
            iaa.Sometimes(prob, iaa.Affine(scale=(1.0, 1.2))),
            # iaa.Sometimes(prob, iaa.Add((-8, 8), per_channel=0.3)),
            # iaa.Sometimes(prob, iaa.ContrastNormalization((0.5, 2.2), per_channel=0.3)),
            # iaa.Sometimes(prob, iaa.Multiply((0.9, 1.1), per_channel=0.5)),
            # iaa.Sometimes(prob, iaa.Multiply((0.9, 1.1))),
            # iaa.Sometimes(0.3, iaa.Invert(0.2, per_channel=True)),
            iaa.Sometimes(prob, iaa.GaussianBlur(sigma=(0, 1.2))),
            # iaa.Sometimes(prob, iaa.CoarseDropout(p=0.2, size_percent=0.05))
        ], random_order=False)

    def __call__(self, img):
        img = np.array(img)
        return self.aug(images=img)


class AddBackgroundImageTransform:
    def __init__(self, bg_image_dataset_folder, size=(128, 128)):
        self.dataset = VOCDetection(bg_image_dataset_folder)
        self.resize = transforms.Resize(size)

    def __call__(self, batch):
        idx = np.random.randint(0, len(self.dataset))
        bg_img, second = self.dataset[idx]
        bg_img = self.resize(bg_img)

        img, mask, p = batch
        new_img = img.copy()
        bg_img = np.moveaxis(np.array(bg_img), 2, 0) / 255.0
        mask3d = np.repeat(mask, 3, axis=0)
        new_img[mask3d == 0] = bg_img[mask3d == 0]
        return new_img


class AAETransform:
    def __init__(self, aug_prob=0.5, bg_image_dataset_folder=None, batch_size=None, size=(128, 128), ImgAugTransformClass=ImgAugTransform,
                 add_bg=True, add_aug=True, add_patches=True):
        self.imgaug = ImgAugTransformClass(aug_prob)
        self.bgimg = AddBackgroundImageTransform(bg_image_dataset_folder, size)
        self.patches = transforms.Compose([transforms.RandomErasing(p=0.5, ratio=(0.3, 3.3), scale=(0.05, 0.15))] * 3)
        self.add_bg = add_bg
        self.add_aug = add_aug
        self.add_patches = add_patches

    def __call__(self, batch):
        img = batch
        if self.add_bg:
            img = self.bgimg(img)
        if self.add_aug:
            img = self.imgaug(img * 255.0) / 255.0
        if self.add_patches:
            img = self.patches(torch.Tensor(img))
        else:
            img = torch.Tensor(img)
        return img


class AugmentedDataset(RenderedDataset):
    def __init__(self, grider:Grid,  model_path=None, res=128,
                 transform=None, render_res=640, camera_dist=0.5,
                 intensity_render=10, intensity_augment=(2, 20),
                 show_file=None, n_workers=4):
        super().__init__(grider, model_path, res, render_res=render_res, camera_dist=camera_dist,
                         directional_light_intensity=intensity_render)
        self.transform = transform
        self.inputs_augmented = None
        self.show_file = show_file
        self.n_workers = n_workers
        self.intensity_render = intensity_render
        self.intensity_augment = intensity_augment
        self.online_ds = None
        if self.intensity_augment != self.intensity_render:
            self.online_ds = OnlineRenderDataset(grider, model_path, res, camera_dist, render_res, intensity_augment)

    def augment(self):
        if self.inputs is None:
            raise RuntimeError("Attempt to augment empty dataset! Create or load dataset first")
        if self.transform is None:
            raise RuntimeError("Transform is None")

        if self.online_ds:
            grid = self.grider.grid
            for i in tqdm(range(len(grid))):
                rot = grid[i]
                color, depth = self.online_ds.objren.render_and_crop(rot, self.res)
                self.inputs_augmented[i, :, :, :] = np.moveaxis(color, 2, 0) / 255.0

        def aug(i):
            input_img = self.inputs[i] if self.online_ds is None else self.inputs_augmented[i]
            img = self.transform((input_img, self.masks[i], self.rots[i]))
            self.inputs_augmented[i] = np.array(img)

            if self.show_file and i > 0 and i % 8 == 0:
                print_batch(x=torch.tensor(self.inputs[i - 8:i]),
                            x_hat=torch.tensor(self.inputs_augmented[i - 8:i]),
                            save_path=self.show_file)

        with parallel_backend('threading', n_jobs=self.n_workers):
            ProgressParallel()(delayed(aug)(i) for i in range(len(self)))

    def create_dataset(self, folder):
        super().create_dataset(folder)
        self.inputs_augmented = np.memmap(folder + '/inputs_augmented.npy', dtype=np.float32, mode='w+',
                                          shape=(self.size, 3, self.res, self.res))
        self.augment()

    def load_dataset(self, folder):
        super().load_dataset(folder)
        self.inputs_augmented = np.memmap(folder + '/inputs_augmented.npy', dtype=np.float32, mode='r+',
                                          shape=(self.size, 3, self.res, self.res))

    def to_torch(self):
        super().to_torch()
        self.inputs_augmented = torch.from_numpy(self.inputs_augmented)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if not isinstance(self.inputs_augmented, torch.Tensor):
            self.inputs_augmented = torch.from_numpy(self.inputs_augmented)

        return self.inputs[idx], self.masks[idx], self.rots[idx], self.inputs_augmented[idx]


if __name__ == '__main__':
    fuze_path = '/home/safoex/Documents/libs/pyrender/examples/models/fuze.obj'
    t = AAETransform(0.5, '/home/safoex/Documents/data/VOCtrainval_11-May-2012')
    grid = Grid(100, 10)
    ds = AugmentedDataset(grid, fuze_path, transform=t, show_file='test_saveX/dafaq.png')
    ds.create_dataset('test_saveX')
