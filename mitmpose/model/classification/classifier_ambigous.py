import torchvision
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from mitmpose.model.classification.dataset import ManyObjectsRenderedDataset
from mitmpose.model.classification.dataset_ambigous import ManyAmbigousObjectsLabeledRenderedDataset
from mitmpose.model.classification.classifier import ObjectClassifierDataModule, ObjectClassifier
from mitmpose.model.pose.grids.grids import Grid
from mitmpose.model.pose.datasets.augment import AAETransform

import pytorch_lightning as pl


class AmbiguousObjectClassifierDataModule(ObjectClassifierDataModule):
    def __init__(self, ds_args, workdir, fraction, batch_size=4, num_workers=0, val_split=0.1):
        ds_args.update({
            'keep_fraction': fraction
        })
        ds = ManyAmbigousObjectsLabeledRenderedDataset(**ds_args)
        ds.labeler.load(workdir, with_codebooks=False)
        ds.labeler.recalculate_fin_labels()
        ds.load_dataset(workdir, with_codebooks=False)
        super().__init__(ds, batch_size, num_workers, val_split)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)


class SortedUncertaintyExperiment:
    def __init__(self, ds_args, workdir, fraction_size=0.1, fraction_step=0.05, val_size=0.05, epochs=5, freeze_conv=False):
        self.ds_args = ds_args
        self.fraction_size = fraction_size
        self.fraction_step = fraction_step
        self.val_size = val_size
        self.workdir = workdir
        self.epochs = epochs
        self.freeze_conv = freeze_conv
        self.n_models = len(ds_args['models'])

    def create_dataset(self):
        ds = ManyAmbigousObjectsLabeledRenderedDataset(**self.ds_args)
        ds.labeler.load(self.workdir)
        ds.labeler.recalculate_fin_labels()
        ds.create_dataset(self.workdir)

    def single_experiment(self, fraction, val_before, val_after):
        train_dm = AmbiguousObjectClassifierDataModule(self.ds_args, self.workdir, fraction)
        val_before_dm = AmbiguousObjectClassifierDataModule(self.ds_args, self.workdir, val_before, val_split=0.95)
        val_after_dm = AmbiguousObjectClassifierDataModule(self.ds_args, self.workdir, val_after, val_split=0.95)

        oc = ObjectClassifier(self.n_models, with_labels=True, freeze_conv=self.freeze_conv)
        trainer = pl.Trainer(gpus=1, max_epochs=self.epochs)
        trainer.fit(oc, train_dm)
        # return trainer.test(test_dataloaders=[val_before_dm.val_dataloader(), val_after_dm.val_dataloader()])
        return {
            'begin': trainer.test(test_dataloaders=val_before_dm.val_dataloader()),
            'end': trainer.test(test_dataloaders=val_after_dm.val_dataloader())
        }


    def series_experiment(self):
        fraction_begin = self.val_size
        fraction_end = 1 - self.val_size - self.fraction_size - self.fraction_step
        results = []
        while fraction_begin <= fraction_end:
            fraction = (fraction_begin, fraction_begin + self.fraction_size)
            val_before = (fraction_begin - self.val_size, fraction_begin)
            val_after = (fraction[1], fraction[1] + self.val_size)

            res = self.single_experiment(fraction, val_before, val_after)
            res['fraction'] = fraction
            results.append(res)
            fraction_begin += self.fraction_step
        return results