import numpy as np
import torch
from torch.utils.data import Dataset
from mitmpose.model.so3 import fibonacci_sphere_rot

from mitmpose.model.so3.dataset import RenderedDataset

from imgaug import augmenters as iaa
from torchvision.datasets import VOCDetection
from torchvision import transforms
from tqdm import tqdm


class ImgAugTransform:
    def __init__(self, prob=0.5):
        self.aug = iaa.Sequential([
            iaa.Sometimes(prob, iaa.Affine(scale=(1.0, 1.2))),
            # iaa.Sometimes(prob, iaa.Add((-25, 25), per_channel=0.3)),
            iaa.Sometimes(prob, iaa.ContrastNormalization((0.5, 2.2), per_channel=0.3)),
            iaa.Sometimes(prob, iaa.Multiply((0.6, 1.4), per_channel=0.5)),
            iaa.Sometimes(prob, iaa.Multiply((0.9, 1.1))),
            iaa.Sometimes(0.3, iaa.Invert(0.2, per_channel=True)),
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
    def __init__(self, aug_prob=0.5, bg_image_dataset_folder=None, batch_size=None, size=(128, 128)):
        self.imgaug = ImgAugTransform(aug_prob)
        self.bgimg = AddBackgroundImageTransform(bg_image_dataset_folder, size)
        self.patches = transforms.Compose([transforms.RandomErasing(p=0.5, ratio=(0.3, 3.3), scale=(0.05, 0.15))] * 3)

    def __call__(self, batch):
        img = self.bgimg(batch)
        img = self.imgaug(img * 255.0) / 255.0
        img = self.patches(torch.Tensor(img))
        # img = torch.Tensor(img)
        return img


class AugmentedDataset(RenderedDataset):
    def __init__(self, size_sphere, size_in_plane, model_path=None, res=128, grid_generator=fibonacci_sphere_rot,
                 transform=None, render_res=640, camera_dist=0.5):
        super().__init__(size_sphere, size_in_plane, model_path, res, grid_generator, render_res=render_res, camera_dist=camera_dist)
        self.transform = transform
        self.inputs_augmented = None

    def augment(self):
        if self.inputs is None:
            raise RuntimeError("Attempt to augment empty dataset! Create or load dataset first")
        if self.transform is None:
            raise RuntimeError("Transform is None")

        for i in tqdm(range(len(self))):
            img = self.transform((self.inputs[i], self.masks[i], self.rots[i]))
            self.inputs_augmented[i] = np.array(img)

    def create_dataset(self, folder):
        super().create_dataset(folder)
        self.inputs_augmented = np.memmap(folder + '/inputs_augmented.npy', dtype=np.float32, mode='w+', shape=(self.size, 3, self.res, self.res))
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
    ds = AugmentedDataset(100, 10, fuze_path, transform=t)
    ds.create_dataset('test_save3')
