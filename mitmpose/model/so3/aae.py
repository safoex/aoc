import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

from mitmpose.model.so3 import EncoderCNN
from mitmpose.model.so3.decoder import Decoder
from mitmpose.model.so3.dataset import RenderedDataset
from mitmpose.model.so3.augment import AugmentedDataset, AAETransform

from torch.utils.data import DataLoader
from torchvision import transforms

from logutil import TimeSeries
from mitmpose.model.so3.ae import AE, print_batch, AEDataModule
import pytorch_lightning as pl


class AAE(AE):
    def __init__(self, image_size=128, latent_size=128, filters=(128, 256, 256, 512), lae_inside=None):
        super().__init__(image_size, latent_size, filters, lae_inside)

    def training_step(self, batch, batch_idx):
        x, m, p, x_aug = batch
        x_hat = ae.forward(x_aug)
        loss = self.loss_aae(x, x_hat, bootstrap_ratio=4)
        result = pl.TrainResult(loss)
        result.log('AE reconstruction loss', loss, prog_bar=True)
        return result



if __name__ == "__main__":
    fuze_path = '/home/safoex/Documents/libs/pyrender/examples/models/fuze.obj'
    t = AAETransform(0.5, '/home/safoex/Documents/data/VOCtrainval_11-May-2012')
    ds = AugmentedDataset(1000, 128, fuze_path, transform=t)
    ds.create_dataset('test_save4')
    # dataset = AugmentedDataset(200, 128)
    # dataset.load_dataset('test_save3')

    ae = AAE(128, 32, (8, 16, 16, 32))
    # train_ae(ae, dataset, iters=3000, save_every=30, save_path='test_save4/recons.jpg', batch_size=128)
    dm = AEDataModule(ds, 'test_save4')

    trainer = pl.Trainer()
    trainer.fit(ae, dm)
