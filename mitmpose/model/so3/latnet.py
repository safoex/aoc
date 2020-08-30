import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary


from mitmpose.model.so3 import EncoderCNN
from mitmpose.model.so3.decoder import Decoder
from mitmpose.model.so3.dataset import RenderedDataset
from mitmpose.model.so3.ae import AE, train_ae, print_batch

from torch.utils.data import DataLoader
from torchvision import transforms

from logutil import TimeSeries



class LatNet(nn.Module):
    def __init__(self, latent_size, gen_size=None, pose_size=1):
        super().__init__()
        self.latent_size = latent_size
        self.pose_size = pose_size
        self.gen_size = gen_size or latent_size // 2
        self.input_features = pose_size if isinstance(pose_size, int) else pose_size[0] * pose_size[1]
        self.fc1 = nn.Linear(self.input_features, self.gen_size)
        self.fc2 = nn.Linear(self.gen_size, self.gen_size)
        self.bn2 = nn.BatchNorm1d(self.gen_size)
        self.fc3 = nn.Linear(self.gen_size, self.gen_size)
        self.bn3 = nn.BatchNorm1d(self.gen_size)
        self.fc4 = nn.Linear(self.gen_size, latent_size)
        self.cuda()

    def forward(self, p):
        p = p.view((-1, self.input_features))
        x = self.fc1(p)

        x = F.leaky_relu(x, 0.2)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)

        x = self.fc3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2)

        x = self.fc4(x)
        x = torch.sigmoid(x)
        return x


class LinearList(nn.Module):
    def __init__(self, sizes, activations):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(ins, outs) for ins, outs in zip(sizes[:-1], sizes[1:])])
        self.activations = activations

    def forward(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
        return x


class LAE(nn.Module):
    def __init__(self, in_size, out_size, fcs):
        super().__init__()
        inputs = [in_size] + fcs
        outputs = fcs + [out_size]
        ins = [in_size] + fcs + [out_size]
        outs = list(reversed(ins))

        self.enc = LinearList(ins, [torch.sigmoid] * (len(ins) - 1))
        self.dec = LinearList(outs, [torch.sigmoid] * (len(ins) - 1))

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x


def train_lae(ae, lae, dataset, epochs=20, batch_size=128, print_every_seconds=2):
    ts = TimeSeries('Training ae', epochs * (len(dataset) // batch_size) + 1)
    opt = optim.Adam(lae.parameters(), lr=2e-2)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    ae.eval()
    for epoch in range(epochs):
        for i_batch, batch in enumerate(dataloader):
            lae.train()
            x, m, p = [o.cuda() for o in batch]
            # y = y.type(torch.cuda.FloatTensor)
            # x = torch.cat((y, y, y), dim=1)
            z = ae.encoder.forward(x)

            z_hat = lae.forward(z)

            loss = F.mse_loss(z, z_hat)
            ts.collect("LAE loss", loss)

            opt.zero_grad()
            loss.backward()

            opt.step()
            lae.eval()
            ts.print_every(print_every_seconds)


def train_latnet(ae, latnet, dataset, epochs=20, batch_size=128, save_every=0, save_path=None,
                 print_every_seconds=10, transform_pose=None):
    ts = TimeSeries('Training ae', epochs * (len(dataset) // batch_size) + 1)
    opt = optim.Adam(latnet.parameters(), lr=2e-4)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    ae.eval()
    for epoch in range(epochs):
        for i_batch, batch in enumerate(dataloader):
            latnet.train()
            x, m, p = [o.cuda() for o in batch]
            # y = y.type(torch.cuda.FloatTensor)
            # x = torch.cat((y, y, y), dim=1)
            z = ae.encoder.forward(x)
            if transform_pose:
                p = transform_pose(p)

            z_hat = latnet.forward(p)

            loss = F.mse_loss(z, z_hat)
            ts.collect("Reconstruction AE loss", loss)

            opt.zero_grad()
            loss.backward()

            opt.step()
            latnet.eval()
            ts.print_every(print_every_seconds)


from scipy.spatial.transform import Rotation
import numpy as np


def train_laeae(ae, lae, dataset, iters_together=2000, iters_splitted=5000, batch_size=32, save_every=0, save_path=None, print_every_seconds=10):
    ts = TimeSeries('Training ae', iters_together + iters_splitted)
    opt_ae = optim.Adam(ae.parameters(), lr=2e-4)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    opt_lae = optim.Adam(lae.parameters(), lr=2e-4)
    i = 0
    print_numbers = 0
    while i < iters_together:
        for i_batch, batch in enumerate(dataloader):
            i += 1
            if i > iters_together:
                break
            ae.train()
            x, m, p = [o.cuda() for o in batch]
            x_hat = ae.forward(x)

            bootstrap_ratio = 4
            if bootstrap_ratio > 1:
                mse = torch.flatten((x_hat - x) ** 2)
                loss_aae = torch.mean(torch.topk(mse, mse.numel() // bootstrap_ratio)[0])
            else:
                loss_aae = F.mse_loss(x, x_hat)
            z = ae.encoder.forward(x)
            z_hat = ae.lae.forward(z)
            loss_sim_latent = F.mse_loss(z, z_hat)
            loss = loss_aae + 0.01 * loss_sim_latent
            ts.collect("Reconstruction AE loss", loss_aae)
            ts.collect("Reconstruction LAE loss", loss_sim_latent)

            opt_ae.zero_grad()
            loss.backward()

            opt_ae.step()
            ae.eval()
            ts.print_every(print_every_seconds)
            if save_every != 0 and save_path is not None and i % save_every == 0:
                print_batch(x, x_hat, save_path)

    ae.use_lae(False)

    while i < iters_together + iters_splitted:
        for i_batch, batch in enumerate(dataloader):
            i += 1
            if i > iters_together + iters_splitted:
                break
            ae.train()
            x, m, p = [o.cuda() for o in batch]
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

            # ------------------ LAE ------------------------ #
            lae.train()
            z = ae.encoder.forward(x)
            z_hat = lae.forward(z)
            if bootstrap_ratio > 1:
                mse = torch.flatten((z_hat - z) ** 2)
                loss_lae = torch.mean(torch.topk(mse, mse.numel() // bootstrap_ratio)[0])
            else:
                loss_lae = F.mse_loss(z, z_hat)

            ts.collect("Reconstruction LAE loss", loss_aae)

            opt_lae.zero_grad()
            loss_lae.backward()

            opt_lae.step()
            lae.eval()

            ts.print_every(print_every_seconds)
            if save_every != 0 and save_path is not None and i % save_every == 0:
                print_batch(x, x_hat, save_path)

if __name__ == "__main__":

    lae = LAE(64, 6, [32])
    lae.cuda()
    ae = AE(128, 64, (32, 64, 64, 128), lae_inside=lae)
    # ae.load_state_dict(torch.load('../saved/fuze64.pth'))
    ae.cuda()
    summary(ae, (3, 128, 128))
    ae.eval()
    latnet = LatNet(64, 128, pose_size=3)
    latnet.cuda()
    dataset = RenderedDataset(5000, 128)
    dataset.load_dataset('test_save2')

    def to_euler(batch):
        b = np.array(batch.cpu())
        b_euler = Rotation.from_matrix(b).as_euler('xyz')
        return torch.Tensor(b_euler.copy()).cuda()

    # train_latnet(ae, latnet, dataset=dataset, epochs=200, transform_pose=to_euler)

    # train_lae(ae, lae, dataset, epochs=200)

    # train_ae(ae, dataset, iters=3000, save_every=30, save_path='test_save2/recons.jpg', batch_size=128)

    train_laeae(ae, lae, dataset, iters_together=2000, iters_splitted=2000, save_every=30, save_path='test_save2/recons.jpg',
                batch_size=128)