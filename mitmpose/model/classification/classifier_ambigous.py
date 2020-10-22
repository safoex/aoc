import torchvision
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from mitmpose.model.classification.dataset import ManyObjectsRenderedDataset
from mitmpose.model.classification.dataset_ambigous import ManyAmbigousObjectsLabeledRenderedDataset
from mitmpose.model.pose.grids.grids import Grid
from mitmpose.model.pose.datasets.augment import AAETransform

import pytorch_lightning as pl


class AmbioguousObjectClassifierDataHandler:
    def __init__(self, workdir, grider: Grid, grider_codebook: Grid, models: dict, joint_ae, aae_render_tranform, classification_transform=None, res=128, camera_dist=None,
                 render_res=640, intensity_render=10, intensity_augment=(2, 20),
                 fraction_step=0.05, fraction_val=0.05, create_ds=True):

        def refresh_dataset(fraction):
            ds = ManyAmbigousObjectsLabeledRenderedDataset(grider, grider_codebook, models, joint_ae, aae_render_tranform, classification_transform,
                                                            res,camera_dist, render_res, intensity_render, intensity_augment,
                                                            False, None, fraction)
            ds.labeler.load(workdir)
            ds.recalculate_fin_labels()
            ds.load_dataset(workdir)
            return ds


class AmbiguousObjectClassifierDataModule(pl.LightningDataModule):
    def __init__(self, ds: [ManyObjectsRenderedDataset, ManyAmbigousObjectsLabeledRenderedDataset],
                 batch_size=4, num_workers=4, val_part=0.1, val_begin_part=0.05, val_end_part=0.05):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        trains_num = int(len(ds) * (1 - val_part))
        vals_num = len(ds) - trains_num
        self.train_ds, self.val_ds = random_split(ds, [trains_num, vals_num])


    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)