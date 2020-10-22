import torchvision
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from mitmpose.model.classification.dataset import ManyObjectsRenderedDataset
from mitmpose.model.classification.dataset_ambigous import ManyAmbigousObjectsLabeledRenderedDataset
from mitmpose.model.pose.grids.grids import Grid
from mitmpose.model.pose.datasets.augment import AAETransform

import pytorch_lightning as pl


class ObjectClassifier(pl.LightningModule):
    def __init__(self, num_target_classes, with_labels=False):
        super().__init__()
        # init a pretrained resnet
        self.classifier = torchvision.models.resnet18(pretrained=True)

        # freeze weights of feature extractor
        # for param in self.classifier.parameters():
        #     param.requires_grad = False

        # set a number of target_classes
        num_filters = self.classifier.fc.in_features
        self.classifier.fc = nn.Linear(num_filters, num_target_classes)

        # set input_size
        self.input_size = 224

        self.with_labels = with_labels
        if self.with_labels:
            self.loss = nn.BCEWithLogitsLoss()
        else:
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
        if self.with_labels:
            corrects = torch.sum(torch.argmax(y_hat, dim=1) == torch.argmax(l, dim=1)).float()
        else:
            corrects = torch.sum(torch.argmax(y_hat, dim=1) == l).float()
        corrects /= y_hat.shape[0]
        tensorboard = self.logger.experiment
        preds_n = min(len(x), 3)
        imgs = torch.cat([x[i] for i in range(preds_n)], 2)
        tensorboard.add_image('images', imgs.cpu(),
                              self.current_epoch, dataformats="CHW")
        logs = {'val_loss': val_loss, 'corrects': corrects}
        return {'val_loss': val_loss, 'corrects': corrects, 'log': logs}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_corr = torch.stack([x['corrects'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'corrects': avg_corr}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}


class ObjectClassifierDataModule(pl.LightningDataModule):
    def __init__(self, ds: [ManyObjectsRenderedDataset, ManyAmbigousObjectsLabeledRenderedDataset],
                 batch_size=4, num_workers=4, val_part=0.1):
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
    models_dir = '/home/safoex/Downloads/cat_food/models_fixed/'
    models_names = ['tonno_low', 'pollo', 'polpa']
    models = {mname: {'model_path': models_dir + '/' + mname + '.obj', 'camera_dist': 140} for mname in models_names}
    workdir = 'test_many_reconstr'
    grider = Grid(300, 10)
    ds = ManyObjectsRenderedDataset(grider, models, aae_render_tranform=AAETransform(0.5, '/home/safoex/Documents/data/VOCtrainval_11-May-2012'))
    ds.create_dataset(workdir)

    trainer = pl.Trainer(gpus=1, max_epochs=20)

    ocmodel = ObjectClassifier(3)
    ocdm = ObjectClassifierDataModule(ds)

    trainer.fit(ocmodel, ocdm)
