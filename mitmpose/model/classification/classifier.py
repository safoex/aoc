import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as TF
from torch import optim
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.data.distributed import DistributedSampler

from mitmpose.model.classification.dataset import ManyObjectsRenderedDataset
from mitmpose.model.so3.grids import Grid
from mitmpose.model.so3.augment import AAETransform

from PIL import Image
import numpy as np
import pytorch_lightning as pl


class ObjectClassifier(pl.LightningModule):
    def __init__(self, num_target_classes):
        super().__init__()
        # init a pretrained resnet
        self.classifier = torchvision.models.resnet50(pretrained=True)

        # freeze weights of feature extractor
        for param in self.classifier.parameters():
            param.requires_grad = False

        # set a number of target_classes
        num_filters = self.classifier.fc.in_features
        self.classifier.fc = nn.Linear(num_filters, num_target_classes)

        # set input_size
        self.input_size = 224

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.classifier(x)

    def configure_optimizers(self):
        params_to_update = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
        optim = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        return optim

    def training_step(self, batch, batch_idx):
        x, l = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, l)

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, l = batch
        y_hat = self.forward(x)
        val_loss = self.loss(y_hat, l)
        _, preds = torch.max(y_hat, 1)
        correct = torch.sum(preds == l).type(torch.float32)
        return {'val_loss': val_loss, 'corrects': correct}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_corr = torch.stack([x['corrects'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'corrects': avg_corr}
        return {'avg_val_loss': avg_loss, 'avg_correct_preds': avg_corr, 'log': tensorboard_logs}


class ObjectClassifierDataModule(pl.LightningDataModule):
    def __init__(self, ds: ManyObjectsRenderedDataset, batch_size=4, num_workers=4, val_part=0.1):
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


if __name__ == '__main__':
    models = {
        'fuze': {
            'model_path': '/home/safoex/Documents/libs/pyrender/examples/models/fuze.obj'
        },
        'drill': {
            'model_path': '/home/safoex/Documents/libs/pyrender/examples/models/drill.obj'
        }
    }

    workdir = 'test_many'
    grider = Grid(300, 10)
    ds = ManyObjectsRenderedDataset(grider, models, aae_render_tranform=AAETransform(0.5, '/home/safoex/Documents/data/VOCtrainval_11-May-2012'))
    ds.load_dataset(workdir)

    trainer = pl.Trainer(gpus=1, max_epochs=20)

    ocmodel = ObjectClassifier(2)
    ocdm = ObjectClassifierDataModule(ds)

    trainer.fit(ocmodel, ocdm)
