import time
import os
import json
import numpy as np
import random

from skimage.draw import polygon
from tqdm import tqdm
import imutil
from logutil import TimeSeries

latent_size = 20
gen_size = 20
mid_size = 128
batch_size = 32
iters = 50 * 300

env = None
prev_states = None


# First, construct a dataset
def random_example():
    x = np.random.randint(15, 25)
    y = np.random.randint(15, 25)
    radius = np.random.randint(6, 12)
    rotation = np.random.uniform(0, 2*np.pi)
    noisy_input = build_box(x, y, radius, rotation)
    target_output = build_box(x=20, y=20, radius=9, rotation=rotation)
    return noisy_input, target_output, rotation


def build_box(x, y, radius, rotation):
    state = np.zeros((1, 40, 40))
    def polar_to_euc(r, theta):
        return (y + r * np.sin(theta), x + r * np.cos(theta))
    points = [polar_to_euc(radius, rotation + t) for t in [
        np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]]
    r, c = zip(*points)
    rr, cc = polygon(r, c)
    state[:, rr, cc] = 1.
    return state


def build_dataset(size=50000):
    dataset = []
    for i in tqdm(range(size), desc='Building Dataset...'):
        x, y, r = random_example()
        dataset.append((x, y, r))
    return dataset  # list of examples





from torch.utils.data import Dataset, DataLoader


class BoxDataset(Dataset):
    def __init__(self, size=50000):
        super().__init__()
        self.size = size
        self.data = [[z.astype(dtype=np.float32) if not isinstance(z, float) else z for z in xyr ] for xyr in build_dataset(size)]

    def __len__(self):
        return  self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx]

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, latent_size, mid_size=512):
        super().__init__()
        self.latent_size = latent_size
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 3, kernel_size=4, stride=2, padding=1)

        self.fc1 = nn.Linear(20*20*3, mid_size)
        self.bn2 = nn.BatchNorm1d(mid_size)
        self.fc2 = nn.Linear(mid_size, mid_size)
        self.bn3 = nn.BatchNorm1d(mid_size)
        self.fc3 = nn.Linear(mid_size, latent_size)
        self.cuda()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = x.view(-1, 20*20*3)

        x = self.fc1(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc2(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2)
        z = self.fc3(x)
        # z = torch.sigmoid(z)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_size, mid_size=512):
        super().__init__()
        self.latent_size = latent_size
        self.fc1 = nn.Linear(latent_size, mid_size)
        self.bn0 = nn.BatchNorm1d(mid_size)
        self.fc2 = nn.Linear(mid_size, mid_size // 2)

        self.deconv1 = nn.ConvTranspose2d(mid_size // 2, mid_size // 4, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(mid_size // 4)
        # 128 x 5 x 5
        self.deconv2 = nn.ConvTranspose2d(mid_size // 4, mid_size // 8, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_size // 8)
        # 64 x 10 x 10
        self.deconv3 = nn.ConvTranspose2d(mid_size // 8, mid_size // 8, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(mid_size // 8)
        # 64 x 20 x 20
        self.deconv4 = nn.ConvTranspose2d(mid_size // 8, mid_size // 8, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(mid_size // 8)
        # 64 x 40 x 40
        self.deconv5 = nn.ConvTranspose2d(mid_size // 8, 1, kernel_size=3, stride=1, padding=1)
        self.cuda()

    def forward(self, z):
        x = self.fc1(z)
        x = F.leaky_relu(x, 0.2)
        x = self.bn0(x)

        x = self.fc2(x)
        x = F.leaky_relu(x, 0.2)

        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.deconv1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.bn1(x)

        x = self.deconv2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.bn2(x)

        x = self.deconv3(x)
        x = F.leaky_relu(x, 0.2)
        x = self.bn3(x)

        x = self.deconv4(x)
        x = F.leaky_relu(x, 0.2)
        x = self.bn4(x)

        x = self.deconv5(x)
        x = torch.sigmoid(x)
        return x


class AE(nn.Module):
    def __init__(self, latent_size, mid_size=512):
        super().__init__()
        self.latent_size = latent_size
        self.mid_size = mid_size
        self.encoder = Encoder(latent_size, mid_size)
        self.decoder = Decoder(latent_size, mid_size)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


class Generator(nn.Module):
    def __init__(self, latent_size, gen_size=None, pose_size=1):
        super().__init__()
        self.latent_size = latent_size
        self.pose_size = pose_size
        self.gen_size = gen_size or latent_size // 2
        self.fc1 = nn.Linear(pose_size, self.gen_size)
        self.fc2 = nn.Linear(self.gen_size, self.gen_size)
        self.fc3 = nn.Linear(self.gen_size, latent_size)
        self.cuda()

    def sample_around2d(self, p, n=1, eps=0.2):
        phi = torch.atan2(p[:, 1], p[:, 0])
        random_delta = (torch.rand((phi.shape[0], n)) - 0.5) * eps
        phi_around = phi + random_delta
        p_around = torch.stack((torch.cos(phi_around), torch.sin(phi_around)))
        signs = torch.sign(p_around - p)
        return p_around, signs

    def forward(self, p):
        p = p.reshape((32, 1))
        x = self.fc1(p)
        x = x * torch.sigmoid(x)

        x = self.fc2(x)
        x = x * torch.sigmoid(x)

        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

    def meta_loss2d(self, batch_z, batch_p):
        p_around, signs = self.sample_around2d(batch_p, 10)
        z_around = self.forward(p_around)


def train_ae(ae, dataset_size, iters=5000, batch_size=32):
    dataset = BoxDataset(dataset_size)
    ts = TimeSeries('Training ae', iters)
    opt_ae = optim.Adam(ae.parameters())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    i = 0
    while i < iters:
        for i_batch, batch in enumerate(dataloader):
            i += 1
            if i > iters:
                break
            ae.train()
            ae.zero_grad()
            x, y, p = [o.cuda() for o in batch]
            x_hat = ae.forward(x)
            loss_aae = F.binary_cross_entropy(x_hat, y)

            ts.collect("Reconstruction AE loss", loss_aae)

            loss_aae.backward()

            opt_ae.step()
            ae.eval()
            ts.print_every(2)



def main():
    # Create a 40x40 monochrome image autoencoder

    dataset = build_dataset(size=iters)

    def get_batch(size=32):
        idx = np.random.randint(len(dataset) - size)
        examples = dataset[idx:idx + size]
        return zip(*examples)

    encoder = Encoder(latent_size, mid_size)
    generator = Generator(latent_size)
    decoder = Decoder(latent_size, mid_size)
    opt_encoder = optim.Adam(encoder.parameters())
    opt_generator = optim.Adam(generator.parameters())
    opt_decoder = optim.Adam(decoder.parameters())

    ts = TimeSeries('Training', iters)

    # Train the network on the denoising autoencoder task
    for i in range(iters):
        encoder.train()
        generator.train()
        decoder.train()

        opt_encoder.zero_grad()
        opt_generator.zero_grad()
        opt_decoder.zero_grad()

        batch_input, batch_view_output, batch_target = get_batch()
        x = torch.Tensor(batch_input).cuda()
        y = torch.Tensor(batch_view_output).cuda()
        p = torch.Tensor(batch_target).cuda()

        z_enc = encoder(x)
        z_gen = generator(p)
        x_hat = decoder(z_enc)
        loss_aae = F.binary_cross_entropy(x_hat, y)
        z_dot_product = (z_enc * z_gen).sum(-1)
        z_enc_norm = torch.norm(z_enc, dim=1)
        z_gen_norm = torch.norm(z_gen, dim=1)
        loss_gen = z_dot_product / z_enc_norm / z_gen_norm
        loss_gen = z_enc.shape[0] - loss_gen.sum()

        if loss_aae < 0.01:
            k_gen = 1
            k_aae = 0.01
        else:
            k_gen = 0.1
            k_aae = 1

        loss = k_gen * loss_gen + k_aae * loss_aae
        ts.collect('Reconstruction loss', loss_aae)
        ts.collect('Generation loss', loss_gen)

        loss.backward()
        opt_encoder.step()
        opt_generator.step()
        opt_decoder.step()

        encoder.eval()
        generator.eval()
        decoder.eval()

        # if i % 25 == 0:
        #     filename = 'reconstructions/iter_{:06}_reconstruction.jpg'.format(i)
        #     x = torch.Tensor(demo_batch).cuda()
        #     z = encoder(x)
        #     x_hat = generator(z)
        #     img = torch.cat([x[:4], x_hat[:4]])
        #     caption = 'iter {}: orig. vs reconstruction'.format(i)
        #     imutil.show(img, filename=filename, resize_to=(256,512), img_padding=10, caption=caption, font_size=8)
        #     vid.write_frame(img, resize_to=(256,512), img_padding=10, caption=caption, font_size=12)
        ts.print_every(2)
    #
    # # Now examine the representation that the network has learned
    # EVAL_FRAMES = 360
    # z = torch.Tensor((1, latent_size)).cuda()
    # ts_eval = TimeSeries('Evaluation', EVAL_FRAMES)
    # vid_eval = imutil.VideoLoop('latent_space_traversal')
    # for i in range(EVAL_FRAMES):
    #     theta = 2*np.pi*(i / EVAL_FRAMES)
    #     box = build_box(20, 20, 10, theta)
    #     z = encoder(torch.Tensor(box).unsqueeze(0).cuda())[0]
    #     ts_eval.collect('Latent Dim 1', z[0])
    #     ts_eval.collect('Latent Dim 2', z[1])
    #     caption = "Theta={:.2f} Z_0={:.3f} Z_1={:.3f}".format(theta, z[0], z[1])
    #     pixels = imutil.show(box, resize_to=(512,512), caption=caption, font_size=12, return_pixels=True)
    #     vid_eval.write_frame(pixels)
    # print(ts)

def main2():
    ae = AE(10, 64)
    train_ae(ae, 50000, 5000)

if __name__ == '__main__':
    main2()