from aoc.model.pose.aae.aae import AAE
from aoc.model.pose.datasets.dataset import RenderedDataset, IndexedDataset, OnlineRenderDataset
from aoc.model.pose.grids.grids import Grid, GradUniformGrid, AxisSwapGrid
from torch.utils.data import DataLoader
import torch

from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt

from aoc.model.pose.render import ObjectRenderer
from scipy.stats import special_ortho_group
import numpy as np
from tqdm import  tqdm


class Codebook:
    def __init__(self, model: AAE, dataset: [RenderedDataset, OnlineRenderDataset], batch_size=32, latent_size=None):
        self.model = model
        self.dataset = IndexedDataset(dataset)
        self._codebook = None
        self.batch_size = batch_size
        self.grider = dataset.grider
        self._ds = dataset
        self.device = next(self.model.parameters()).device

        self.encoder = self.model
        if isinstance(self.model, AAE):
            self.encoder = self.model.encoder

        self.latent_size = latent_size or self.model.latent_size

    @property
    def codebook(self):
        if self._codebook is None:
            dl = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
            self._codebook = torch.zeros((len(self.dataset), self.latent_size), dtype=torch.float32).to(device=self.device)
            self.model.eval()
            for data, idcs in tqdm(dl):
                imgs = data[0]
                with torch.no_grad():
                    self._codebook[idcs, :] = self.encoder.forward(imgs.to(device=self.device))

            self.normalize_(self._codebook)

        return self._codebook

    @staticmethod
    def normalize(code):
        if len(code.shape) > 1:
            norm = code.norm(p=2, dim=1, keepdim=True)
            return code.div(norm)
        else:
            return code / torch.norm(code)

    @staticmethod
    def normalize_(code):
        if len(code.shape) > 1:
            norm = code.norm(p=2, dim=1, keepdim=True)
            code.div_(norm)
        else:
            code.div_(torch.norm(code))

    def latent(self, img, norm=True):
        img = np.moveaxis(img, 2, 0) / 255.0
        img = img[np.newaxis, :, :, :]
        with torch.no_grad():
            code = self.encoder.forward(torch.Tensor(img).to(self.device))
        if norm:
            code /= torch.norm(code)
        return code

    def cos_sim(self, codes1, codes2=None, norm1=False, norm2=False):
        if codes2 is None:
            codes2 = self.codebook
            norm2 = True

        if not norm1:
            self.normalize_(codes1)

        if not norm2:
            self.normalize_(codes2)

        cross_prod = codes1 * codes2

        if len(cross_prod.shape) >= 2:
            return torch.sum(cross_prod, dim=1)
        else:
            return torch.sum(cross_prod)

    def best(self, code, norm=True):
        if not norm:
            self.normalize_(code)

        return torch.argmax(self.cos_sim(code)).item()
        # self.dataset.dataset.grider.grid[]

    def save(self, path):
        torch.save(self.codebook, path)

    def load(self, path):
        self._codebook = torch.load(path, map_location=self.device)

    def _cross_loss_idx(self, idcs1, idcs2):
        return self.cos_sim(self.codebook[idcs1], self.codebook[idcs2], True, True)

    def latent_approx(self, rots):
        return self.codebook[self.grider.nn_index(rots)]

    def cross_loss(self, rots1, rots2):
        return self.cos_sim(self.latent_approx(rots1), self.latent_approx(rots2), True, True)

    def cross_loss_exact(self, rots1, rots2):
        # do not forget to set model_path in dataset you provided!
        assert rots1.shape == rots2.shape
        with torch.no_grad():
            if len(rots1.shape) > 2:
                codes = [torch.FloatTensor(rots1.shape[0], self.latent_size).to(self.device) for _ in range(2)]
                for i, rot12 in enumerate(zip(rots1, rots2)):
                    imgs = [self._ds.objren.render_and_crop(rot)[0] for rot in rot12]
                    for code, img in zip(codes, imgs):
                        code[i, :] = self.latent(img)
            else:
                codes = [self.latent(self._ds.objren.render_and_crop(rot)[0]) for rot in (rots1, rots2)]

            return self.cos_sim(codes[0], codes[1])

    def latent_exact(self, rots):
        with torch.no_grad():
            if len(rots.shape) > 2:
                codes = torch.FloatTensor(rots.shape[0], self.latent_size).to(self.device)
                for i, rot in enumerate(rots):
                    img = self._ds.objren.render_and_crop(rot)[0]
                    codes[i, :] = self.latent(img)
            else:
                codes = self.latent(self._ds.objren.render_and_crop(rots)[0])

        return codes

    def verify(self, rots):
        cl = torch.zeros(rots.shape[0])
        for i, r in enumerate(rots):
            best_guess = self.best(self.latent_exact(r))
            cl[i] = torch.norm(torch.from_numpy(self.grider.grid[best_guess].T.dot(r) - np.eye(3)))
        return torch.min(cl), torch.max(cl), torch.median(cl), torch.mean(cl)


class CodebookGrad(Codebook):
    def __init__(self, model: AAE, dataset: RenderedDataset, batch_size=32):
        super().__init__(model, dataset, batch_size)
        self._interpolator = None

    @property
    def interpolator(self):
        if self._interpolator is None:
            self._interpolator = self.grider.make_interpolator(self.codebook)

        return self._interpolator

    def latent_approx(self, rots):
        return torch.FloatTensor(self.interpolator(rots)).to(self.device)


if __name__ == "__main__2":
    ae = AAE(128, 32, (16, 32, 32, 64)).cuda()
    ae.load_state_dict(torch.load('test_save4/ae32.pth'))

    fuze_path = '/home/safoex/Documents/libs/pyrender/examples/models/fuze.obj'
    grider = Grid(500, 20)
    ds = OnlineRenderDataset(grider, fuze_path)
    ds_path = 'test_save4'
    codebook = Codebook(ae, ds)
    codebook.save(ds_path + '/codebook.pt')
    # codebook.load(ds_path + '/codebook.pt')

    objren = ObjectRenderer(fuze_path)
    rot = special_ortho_group.rvs(3)
    img, depth = objren.render_and_crop(rot)
    lat = codebook.latent(img)
    print(ds.grider.grid[codebook.best(lat)])
    print(rot)

    r1 = special_ortho_group.rvs(3, 100)
    r2 = special_ortho_group.rvs(3, 100)
    print(r1.shape)
    import time
    cl_apprx = codebook.cross_loss(r1, r2)
    st1 = time.time()
    cl_apprx = codebook.cross_loss(r1, r2)
    t1 = time.time() - st1
    st2 = time.time()
    cl_exact = codebook.cross_loss_exact(r1, r2)
    t2 = time.time() - st2
    cl_diff = torch.abs(cl_apprx - cl_exact)
    print(torch.min(cl_diff), torch.max(cl_diff), torch.median(cl_diff))
    print(t1, t2)
    print(codebook.grider.index_grid.shape)

import time

def cross_loss_test(codebook, n=300):
    r1 = special_ortho_group.rvs(3, n)
    r2 = special_ortho_group.rvs(3, n)

    cl_apprx = codebook.cross_loss(r1, r2)
    cl_exact = codebook.cross_loss_exact(r1, r2)
    cl_diff = torch.abs(cl_apprx - cl_exact)
    return torch.min(cl_diff), torch.max(cl_diff), torch.median(cl_diff)


def cross_loss_vis(codebook, N=300, save_path=None):
    eulers = np.zeros((N, 3), dtype=np.float32)
    eulers[:, 1] = np.linspace(-np.pi / 4, +np.pi / 4, N)
    eulers[:, 0] = np.random.random(N) * np.pi / 10
    eulers[:, 2] = np.random.random(N) * np.pi / 10

    rots = Rotation.from_euler('xyz', eulers).as_matrix()

    test = codebook.cos_sim(codebook.latent_approx(rots), codebook.latent_exact(rots))
    plt.scatter(eulers[:, 1], np.array(test.cpu()))
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.clf()


if __name__ == "__main__4":
    ae = AAE(128, 32, (16, 32, 32, 64)).cuda()
    ae.load_state_dict(torch.load('test_save4/ae32.pth'))

    fuze_path = '/home/safoex/Documents/libs/pyrender/examples/models/fuze.obj'
    grider = GradUniformGrid(180, 90, 20)
    ds = OnlineRenderDataset(grider, fuze_path)
    ds_path = 'test_save4'
    codebook = CodebookGrad(ae, ds)
    # codebook.save(ds_path + '/codebook_grad2.pt')
    codebook.load(ds_path + '/codebook_grad2.pt')

    # r1 = special_ortho_group.rvs(3, 300)
    # r2 = special_ortho_group.rvs(3, 300)
    # print(r1.shape)
    # import time
    # from scipy.spatial.transform import Rotation
    # cl_apprx = codebook.cross_loss(r1, r2)
    # st1 = time.time()
    # cl_apprx = codebook.cross_loss(r1, r2)
    # t1 = time.time() - st1
    # st2 = time.time()
    # cl_exact = codebook.cross_loss_exact(r1, r2)
    # t2 = time.time() - st2
    # cl_diff = torch.abs(cl_apprx - cl_exact)
    # imax = torch.argmax(cl_diff)
    # bad1, bad2 = r1[imax], r2[imax]
    # print(Rotation.from_matrix([bad1, bad2]).as_euler('xyz'))
    # for bad in (bad1, bad2):
    #     print(codebook.cos_sim(codebook.latent_approx(bad), codebook.latent_exact(bad)))
    # print(torch.min(cl_diff), torch.max(cl_diff), torch.median(cl_diff))
    # print(t1, t2)


    n = 10
    r1 = special_ortho_group.rvs(3, n)
    r2 = special_ortho_group.rvs(3, n)

    cl_apprx = codebook.cross_loss(r1, r2)

    n = 1000000
    r1 = special_ortho_group.rvs(3, n)
    r2 = special_ortho_group.rvs(3, n)

    st1 = time.time()
    cl_apprx = codebook.cross_loss(r1, r2)
    t_cl = time.time() - st1
    print("time for %d cross_losses is %s" % (n, str(t_cl)))
    # cross_loss_vis(codebook, 3000)

if __name__ == "__main__2":
    ae = AAE(128, 32, (16, 32, 32, 64)).cuda()
    ae.load_state_dict(torch.load('test_save4/ae32.pth'))

    fuze_path = '/home/safoex/Documents/libs/pyrender/examples/models/fuze.obj'
    grider = GradUniformGrid(180, 90, 20)
    ds = OnlineRenderDataset(grider, fuze_path)
    ds_path = 'test_save4'
    codebook = CodebookGrad(ae, ds)
    # codebook.save(ds_path + '/codebook_grad2.pt')
    codebook.load(ds_path + '/codebook_grad2.pt')

    N = 300
    eulers = np.zeros((N, 3), dtype=np.float32)
    eulers[:, 1] = np.linspace(-np.pi/2, np.pi/2, N)
    eulers[:, 0] = np.random.random(N) * np.pi/20
    eulers[:, 2] = np.random.random(N) * np.pi/20


    rots = Rotation.from_euler('xyz', eulers).as_matrix()

    test = codebook.cos_sim(codebook.latent_approx(rots), codebook.latent_exact(rots))

    plt.scatter(eulers[:, 1], np.array(test.cpu()))
    plt.show()


if __name__ == "__main__":
    root_path = '/home/safoex/Documents/data/aae/cans_pth2/pollo'
    codebook_path = root_path + '/codebook_asp_mid.pt'
    aae_path = root_path + '/ae128.pth'
    model_path = '/home/safoex/Downloads/cat_food/models_fixed/pollo.obj'
    ae = AAE(128, 128, (128, 256, 256, 512)).cuda()
    ae.load_state_dict(torch.load(aae_path))

    grider = AxisSwapGrid(60, 60, 40)
    grids = {
        'mid': AxisSwapGrid(60, 60, 40, method='linear'),
        'small': AxisSwapGrid(20, 20, 40, method='linear'),
        'big': AxisSwapGrid(100, 100, 40, method='linear'),
        'gu': GradUniformGrid(120, 60, 40, method='linear'),
        'gu_small': GradUniformGrid(60, 30, 40, method='linear')
    }
    grids_nn = {
        'mid': AxisSwapGrid(60, 60, 40, method='nearest'),
        'small': AxisSwapGrid(20, 20, 40, method='nearest'),
        'big': AxisSwapGrid(100, 100, 40, method='nearest'),
        'gu': GradUniformGrid(120, 60, 40, method='nearest'),
        'gu_small': GradUniformGrid(60, 30, 40, method='nearest'),
    }
    ds = OnlineRenderDataset(grider, model_path, camera_dist=None)
    codebook = CodebookGrad(ae, ds)
    codebook.load(codebook_path)

    codebooks = {
        name: CodebookGrad(ae, OnlineRenderDataset(grid, model_path, camera_dist=None))
        for name, grid in grids.items()
    }
    codebooks_nn = {
        name: CodebookGrad(ae, OnlineRenderDataset(grid, model_path, camera_dist=None))
        for name, grid in grids_nn.items()
    }
    for name, cdbk in codebooks.items():
        cdbk.load(root_path + '/codebook_asp_%s.pt' % name)
    for name, cdbk in codebooks_nn.items():
        cdbk.load(root_path + '/codebook_asp_%s.pt' % name)

    workdir = '/home/safoex/Documents/data/aae/cans_pth2/pollo' + '/codebook_evaluation'
    import os
    if not os.path.exists(workdir):
        os.mkdir(workdir)
    for name, cdbk in codebooks.items():
        print(name)
        print(cross_loss_test(cdbk, 3000))
        cross_loss_vis(cdbk, 3000, workdir + '/%s.png' % name)
        cdbk_nn = codebooks_nn[name]
        print(cross_loss_test(cdbk_nn, 3000))
        cross_loss_vis(cdbk_nn, 3000, workdir + '/%s_nn.png' % name)

if __name__ == "__main__3":
    root_path = '/home/safoex/Documents/data/aae/cans_pth2/pollo'
    codebook_path = root_path + '/codebook_asp_big.pt'
    aae_path = root_path + '/ae128.pth'
    model_path = '/home/safoex/Downloads/cat_food/models_fixed/pollo.obj'
    ae = AAE(128, 128, (128, 256, 256, 512)).cuda()
    ae.load_state_dict(torch.load(aae_path))

    grid = Grid(100, 20)
    # ds_ren = RenderedDataset(grid, model_path)
    # ds_ren.create_dataset('test_saveXX')
    ds_onl = OnlineRenderDataset(grid, model_path)
    # ds_onl._objren = ds_ren._objren

    # codebook_ren = Codebook(ae, ds_ren)
    codebook_onl = Codebook(ae, ds_onl)

    # codebook_ren.save('test_ren.pth')
    codebook_onl.save('test_onl.pth')

    def print_top(crop, cdbk, n=5):
        scores = cdbk.cos_sim(cdbk.latent(crop))
        tops_orig, idcs = torch.topk(scores, n)
        print(tops_orig)
        return idcs[0]

    # codebooks = [codebook_ren, codebook_onl]
    codebooks = [codebook_onl]
    # for cdbk in codebooks:
    #     print(cdbk.verify(rots))

    rots = special_ortho_group.rvs(3, 100)
    for r in rots[:10]:
        # print(r)
        # render_and_save(r, 'best' + '/%d.png' % 0)

        for i, cdbk in enumerate(codebooks):
            with torch.no_grad():
                idx = print_top(ds_onl.objren.render_and_crop(r)[0], cdbk, 1)
                # print(cdbk.grider.grid[idx])
                # render_and_save(cdbk.grider.grid[idx], 'best' + '/%d.png' % (i+1))

        print('----')

