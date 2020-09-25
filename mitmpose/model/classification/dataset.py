import numpy as np
import torch
from torch.utils.data import Dataset
from mitmpose.model.so3.grids import Grid
from mitmpose.model.so3 import ObjectRenderer
from mitmpose.model.so3.dataset import RenderedDataset
from mitmpose.model.so3.augment import AugmentedDataset
from tqdm import tqdm
import os
from torchvision import transforms


class ManyObjectsRenderedDataset(Dataset):
    default_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(236),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def __init__(self, grider: Grid, models: dict, aae_render_tranform, classification_transform=default_transform, res=128, camera_dist=0.5,
                 render_res=640, intensity_render=10, intensity_augment=(2, 20), n_aug_workers=4):
        self.grider = grider
        self.models = models
        self._datasets = None
        self.default_params = {
            'camera_dist': camera_dist,
            'render_res': render_res,
            'res': res,
            'intensity_render': intensity_render,
            'intensity_augment': intensity_augment,
            'n_workers': n_aug_workers
        }
        self.labels = {}
        self.transform = classification_transform
        self.aae_render_transform = aae_render_tranform

    @property
    def datasets(self):
        if self._datasets is None:
            self._datasets = {}
            label = 0
            for model_name, params in self.models.items():
                mparams = self.default_params.copy()
                mparams.update(params)
                mparams['transform'] = self.aae_render_transform
                self.datasets[model_name] = AugmentedDataset(self.grider, **mparams)
                self.labels[model_name] = label
                label += 1
        return self._datasets

    def create_dataset(self, folder):
        for model_name, params in self.models.items():
            self.datasets[model_name].create_dataset(folder + '/' + model_name)

    def load_dataset(self, folder):
        for model_name, dataset in self.datasets.items():
            dataset.load_dataset(folder + '/' + model_name)

    def __len__(self):
        return sum(len(ds) for ds in self.datasets.values())

    def __getitem__(self, idx):
        obj = None
        for obj, ds in self.datasets.items():
            if idx >= len(ds):
                idx -= len(ds)
            else:
                break

        assert obj is not None

        return self.transform(self.datasets[obj][idx][-1]), self.labels[obj]
