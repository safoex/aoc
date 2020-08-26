import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

from model.so3.encoder_cnn import EncoderCNN
from model.so3.decoder import Decoder
from model.so3.dataset import RenderedDataset

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image

from logutil import TimeSeries


class AE(nn.Module):
    def __init__(self, image_size=128, latent_size=128, filters=(128, 256, 256, 512)):
        super().__init__()
        self.latent_size = latent_size
        self.encoder = EncoderCNN(image_size, latent_size, filters)
        self.decoder = Decoder(image_size, latent_size, filters)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

def train_ae(ae, dataset, iters=5000, batch_size=32, save_every=0, save_path=None):
    ts = TimeSeries('Training ae', iters)
    opt_ae = optim.Adam(ae.parameters())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    i = 0
    print_numbers = 0
    while i < iters:
        for i_batch, batch in enumerate(dataloader):
            i += 1
            if i > iters:
                break
            ae.train()
            x, y = [o.cuda() for o in batch]

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
            ts.print_every(10)
            if save_every != 0 and save_path is not None and i % save_every == 0:
                img_tensor_output = x_hat[0,:,:,:].cpu()
                img_tensor_input = x[0,:,:,:].cpu()
                img_tensor = torch.cat((img_tensor_input, img_tensor_output), 2)
                im = transforms.ToPILImage()(255.0 * img_tensor).convert("RGB")
                im.save(save_path)
                if print_numbers == 0 and i > 1000:
                    print(img_tensor_output[0,42:86:4, 42:86:4])
                    print(img_tensor_output[1,42:86:4, 42:86:4])
                    print(img_tensor_output[2,42:86:4, 42:86:4])
                    print_numbers = 1


if __name__ == "__main__":
    dataset = RenderedDataset()
    dataset.load_dataset('test_save')
    ae = AE(128, 256, (16, 32, 32, 64))
    ae.cuda()
    summary(ae, (3, 128, 128))
    train_ae(ae, dataset, iters=30000, save_every=30, save_path='test_save/recons.jpg')
