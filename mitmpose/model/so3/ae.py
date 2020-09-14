import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

from mitmpose.model.so3 import EncoderCNN
from mitmpose.model.so3.decoder import Decoder
from mitmpose.model.so3.dataset import RenderedDataset
from mitmpose.model.so3.augment import AugmentedDataset, print_batch

from torch.utils.data import DataLoader
from torchvision import transforms

from logutil import TimeSeries
import pytorch_lightning as pl
from torch.utils.data import random_split


class AE(pl.LightningModule):
    def __init__(self, image_size=128, latent_size=128, filters=(128, 256, 256, 512), lae_inside=None):
        super().__init__()
        self.latent_size = latent_size
        self.encoder = EncoderCNN(image_size, latent_size, filters)
        self.decoder = Decoder(image_size, latent_size, filters)
        self.lae = lae_inside
        self.lae_used = lae_inside is not None

    def use_lae(self, use=True):
        self.lae_used = use

    def forward(self, x):
        z = self.encoder(x)
        if self.lae is not None and self.lae_used:
            z = self.lae.forward(z)
        x_hat = self.decoder(z)
        return x_hat

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optim

    def loss_aae(self, x, x_hat, bootstrap_ratio=4):
        if bootstrap_ratio > 1:
            mse = torch.flatten((x_hat - x) ** 2)
            loss_aae = torch.mean(torch.topk(mse, mse.numel() // bootstrap_ratio)[0])
        else:
            loss_aae = F.mse_loss(x, x_hat)
        return loss_aae

    def training_step(self, batch, batch_idx):
        x, m, p = batch
        x_hat = ae.forward(x)
        loss = self.loss_aae(x, x_hat, bootstrap_ratio=4)
        result = pl.TrainResult(loss)
        result.log('AE reconstruction loss', loss, prog_bar=True)
        return result

    # def validation_step(self, batch, batch_idx):
    #     x, m, p = batch
    #     x_hat = ae.forward(x)
    #
    #     return pl.EvalResult(self.loss_aae(x, x_hat, bootstrap_ratio=4))


def train_ae(ae, dataset, iters=5000, batch_size=32, save_every=0, save_path=None, print_every_seconds=10):
    ts = TimeSeries('Training ae', iters)
    opt_ae = optim.Adam(ae.parameters(), lr=2e-4)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    i = 0
    print_numbers = 0
    while i < iters:
        for i_batch, batch in enumerate(dataloader):
            i += 1
            if i > iters:
                break
            ae.train()
            x, m, p = [o.cuda() for o in batch]
            # y = y.type(torch.cuda.FloatTensor)
            # x = torch.cat((y, y, y), dim=1)
            x_hat = ae.forward(x)

            bootstrap_ratio = 4
            if bootstrap_ratio > 1:
                mse = torch.flatten((x_hat - x) ** 2)
                loss_aae = torch.mean(torch.topk(mse, mse.numel() // bootstrap_ratio)[0])
            else:
                loss_aae = F.mse_loss(x, x_hat)
            ts.collect("Reconstruction AE loss", loss_aae)

            opt_ae.zero_grad()
            loss_aae.backward()

            opt_ae.step()
            ae.eval()
            ts.print_every(print_every_seconds)
            if save_every != 0 and save_path is not None and i % save_every == 0:
                print_batch(x, x_hat, save_path)


class AEDataModule(pl.LightningDataModule):
    def __init__(self, dataset, dataset_folder, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = dataset
        self.dataset_folder = dataset_folder

    def setup(self, stage=None):
        self.dataset.load_dataset(self.dataset_folder)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
    #
    # def val_dataloader(self) -> DataLoader:
    #     return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=True)


if __name__ == "__main__2":
    dataset = RenderedDataset(500, 128)
    dataset.load_dataset('test_save')
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=1)
    ae = AE(128, 32, (8, 16, 16, 32))
    ae.cuda()
    summary(ae, (3, 128, 128))
    train_ae(ae, dataset, iters=3000, save_every=30, save_path='test_save/recons.jpg', batch_size=128)

if __name__ == "__main__":
    dataset = RenderedDataset(500, 128)
    dataset_folder = 'test_save'
    ae = AE(128, 32, (8, 16, 16, 32))
    dm = AEDataModule(dataset, dataset_folder)

    trainer = pl.Trainer()
    trainer.fit(ae, dm)