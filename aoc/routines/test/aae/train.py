from aoc.model.pose.datasets.dataset import AugmentedAndRenderedDataset, OnlineRenderDataset, Grid
from aoc.model.pose.datasets.augment import AAETransform
from aoc.routines.test.settings import *
from aoc.model.pose.aae.aae import AAE, AEDataModule
import numpy as np
import os
import pytorch_lightning as pl
import torch


if __name__ == '__main__':
    wdir = workdir + '/datasets_augmented'
    if not os.path.exists(wdir):
        os.mkdir(wdir)

    grider = default_train_grider
    ards = AugmentedAndRenderedDataset(grider, example_model_path, AAETransform(bg_image_dataset_folder=voc_path),
                                       aug_class=OnlineRenderDataset)

    ards.create_dataset(wdir)

    ae = AAE(128, 128, (128, 256, 256, 512))

    dm = AEDataModule(ards, wdir, batch_size=16)
    pl.callbacks.early_stopping.EarlyStopping()
    trainer = pl.Trainer(gpus=1, max_epochs=60)
    trainer.fit(ae, datamodule=dm)
    torch.save(ae.state_dict(), wdir + '/ae128.pth')