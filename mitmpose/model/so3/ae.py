import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

from mitmpose.model.so3 import EncoderCNN
from mitmpose.model.so3.decoder import Decoder
from mitmpose.model.so3.dataset import RenderedDataset

from torch.utils.data import DataLoader
from torchvision import transforms

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
            x, y = [o.cuda() for o in batch]
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
                side = 4
                img_tensor_inputs = [torch.cat([x[i, :, :, :].cpu() for i in range(j * side, (j + 1) * side)], 1) for j
                                     in range(2)]
                img_tensor_outputs = [torch.cat([x_hat[i, :, :, :].cpu() for i in range(j * side, (j + 1) * side)], 1)
                                      for j in range(2)]

                img_tensor = torch.cat(img_tensor_inputs + img_tensor_outputs, 2)
                im = transforms.ToPILImage()(img_tensor).convert("RGB")
                im.save(save_path)


if __name__ == "__main__":
    dataset = RenderedDataset(500, 128)
    dataset.load_dataset('test_save')
    ae = AE(128, 32, (8, 16, 16, 32))
    ae.cuda()
    summary(ae, (3, 128, 128))
    train_ae(ae, dataset, iters=3000, save_every=30, save_path='test_save/recons.jpg', batch_size=128)
