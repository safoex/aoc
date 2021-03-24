from mitmpose.model.pose.codebooks.codebook import *
import numpy as np
from mitmpose.model.pose.render import render_and_save
from tqdm import trange

def cartesian_product(*arrays):
    ndim = len(arrays)
    return np.stack(np.meshgrid(*arrays), axis=-1).reshape(-1, ndim)


class PoseGradient:
    def __init__(self, grider, model_path, ae, magic_radius=15.0 / 180 * 3.14, magic_steps=2, magic_spread=40,
                 magic_shrink=3,
                 magic_in_plane_spread=10, magic_in_plane_radius=20.0 / 180 * 3.14):
        self.model = model_path
        self.grider = grider
        self.ae = ae
        self.magic_steps = magic_steps
        self.magic_spread = magic_spread
        self.magic_shrink = magic_shrink
        self.magic_radius = magic_radius
        self.magic_in_plane_radius = magic_in_plane_radius
        self.magic_in_plane_spread = magic_in_plane_spread

    def sample_around(self, phi_0, phi_d, phi_n, r, z_0, z_d, z_n):
        phi = np.linspace(phi_0 - phi_d / 2, phi_0 + phi_d / 2, phi_n)
        z = np.linspace(z_0 - z_d / 2, z_0 + z_d / 2, z_n)
        phi_z = cartesian_product(phi, z)
        x_y_z = np.zeros((len(phi_z), 3))
        x_y_z[:, 0] = np.cos(phi_z[:, 0]) * r
        x_y_z[:, 1] = np.sin(phi_z[:, 0]) * r
        x_y_z[:, 2] = phi_z[:, 1]
        extra_rot = Rotation.from_euler('xyz', x_y_z).as_matrix()
        return phi_z, extra_rot

    def find_best_match_around(self, rot, codebook: Codebook):
        code = codebook.latent_exact(rot)
        extra_v = np.zeros((self.magic_spread * self.magic_in_plane_spread, 3))
        extra_phi_z = np.zeros((self.magic_spread * self.magic_in_plane_spread, 2))
        phi_best = 0
        z_best = 0
        d = np.pi / 10
        mips = self.magic_in_plane_spread
        best_rot = np.zeros_like(rot)
        for i_xy in range(self.magic_steps):
            if i_xy == 0:
                phi_d = 2 * np.pi
            else:
                phi_d = d

            r = self.magic_radius / self.magic_shrink ** i_xy
            r_z = self.magic_in_plane_radius / self.magic_shrink ** i_xy

            phi_z, extra_rots = self.sample_around(phi_best, phi_d, self.magic_spread, r, z_best, r_z, self.magic_in_plane_spread)
            rots = extra_rots.dot(rot)
            sims = codebook.cos_sim(code, codebook.latent_exact(rots)).cpu().numpy()
            i_min = np.argmax(sims)
            phi_best, z_best = extra_phi_z[i_min, :]
            best_rot = rots[i_min]
            print(sims[i_min])
        return best_rot

if __name__ == '__main__':
    model_path = '/home/safoex/Documents/data/aae/models/scans/cleaner.obj'
    ds = OnlineRenderDataset(Grid(100, 1), model_path)
    ae_path = '/home/safoex/Documents/data/aae/test_cleaner/single64.pth'

    ae = AAE(128, 64, (128, 256, 256, 512))
    ae.load_state_dict(torch.load(ae_path))

    ae.cuda()

    codebook = Codebook(ae, ds)

    pg = PoseGradient(Grid(100,1), model_path, ae)

    for i in trange(4):
        rot = special_ortho_group.rvs(3)
        best_rot = pg.find_best_match_around(rot, codebook)
        sim = codebook.cos_sim(codebook.latent_exact(best_rot), codebook.latent_exact(rot))
        render_and_save(ds.objren, rot, "/home/safoex/Documents/data/aae/test_gradient/rot%d_in.png"%i)
        render_and_save(ds.objren, best_rot, "/home/safoex/Documents/data/aae/test_gradient/rot%d_out_%f.png"%(i, float(sim)))
