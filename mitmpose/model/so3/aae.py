import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

from mitmpose.model.so3 import EncoderCNN
from mitmpose.model.so3.decoder import Decoder
from mitmpose.model.so3.dataset import RenderedDataset
from mitmpose.model.so3.augment import AugmentedDataset

from torch.utils.data import DataLoader
from torchvision import transforms

from logutil import TimeSeries
from mitmpose.model.so3.ae import AE, print_batch


class AAE(AE):
    def __init__(self, image_size=128, latent_size=128, filters=(128, 256, 256, 512), lae_inside=None):
        super().__init__(image_size, latent_size, filters, lae_inside)


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
            x, m, p, x_aug = [o.cuda() for o in batch]
            # y = y.type(torch.cuda.FloatTensor)
            # x = torch.cat((y, y, y), dim=1)
            x_hat = ae.forward(x_aug)

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
                print_batch(x_aug, x_hat, save_path)


if __name__ == "__main__":
    dataset = AugmentedDataset(200, 128)
    dataset.load_dataset('test_save3')
    ae = AAE(128, 32, (8, 16, 16, 32))
    ae.cuda()
    summary(ae, (3, 128, 128))
    train_ae(ae, dataset, iters=3000, save_every=30, save_path='test_save3/recons.jpg', batch_size=128)
