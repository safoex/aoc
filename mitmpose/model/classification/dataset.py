import numpy as np
import torch
from torch.utils.data import Dataset
from mitmpose.model.so3.grids import Grid
from mitmpose.model.so3 import ObjectRenderer
from mitmpose.model.so3.dataset import RenderedDataset
from mitmpose.model.so3.augment import AugmentedDataset
from tqdm import tqdm
import os


class ManyObjectRenderedDataset(Dataset):
    def __init__(self, grider: Grid, models: dict, DatasetClass=AugmentedDataset, res=128, camera_dist=0.5, render_res=640):
        self.models = models
        self.datasets = None
        self.default_params = {
            'camera_dist' : camera_dist,
            'render_res' : render_res,
            'res' : res
        }
        self.DatasetClass = DatasetClass

    def create_dataset(self, folder):
        self.datasets = {}
        for model_name, params in self.models.items():
            mparams = self.default_params.copy()
            mparams.update(params)