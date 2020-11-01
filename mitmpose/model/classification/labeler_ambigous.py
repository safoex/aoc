from mitmpose.model.classification.dataset import *
from mitmpose.model.pose.codebooks.codebook import Codebook
from mitmpose.model.pose.datasets.dataset import OnlineRenderDataset
import torch
import itertools
import os
from tqdm.auto import trange, tqdm
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import numpy as np
import shutil


class AmbigousObjectsLabeler:
    def __init__(self, models, grider_label, grider_codebook, ae, knn_median=5, borderline=0.25, width=0.1,
                 magic_grad_steps=5, magic_nn_points=6, magic_shrink_constant=0.4, magic_shrink_exp=1.5):
        self.models = models
        self.grider = grider_label
        self.grider_cdbk = grider_codebook
        self.ae = ae
        self._codebooks = None
        self._simililarities = None
        self._arg_simililarities = None
        self._searched = None
        self.model_list = [m for m in self.models]
        self.model_idx = {m: i for i,m in enumerate(self.model_list)}
        self._labels = None
        self._eulers = None

        self._smoothen = None
        self.knn_median = knn_median
        self._knn_index_grid = None

        self._sorted = None
        self._fin_labels = None
        self.borderline = borderline
        self.width = width

        self.magic_grad_steps = magic_grad_steps
        self.magic_nn_points = magic_nn_points
        self.magic_shrink_constant = magic_shrink_constant
        self.magic_shrink_exp = magic_shrink_exp


    @property
    def codebooks(self):
        if self._codebooks is None:
            self.ae.eval()
            self._codebooks = {}
            for mname, mprop in self.models.items():
                ds = OnlineRenderDataset(self.grider_cdbk, mprop['model_path'], camera_dist=None)
                self._codebooks[mname] = Codebook(self.ae, ds)

        return self._codebooks

    def find_best_match_around_gradient(self, code, cdbk: Codebook, starting_v=None, starting_i=None):
        # sounds good does not work
        def fun(v):
            return 1 - cdbk.cos_sim(code, cdbk.latent_exact(Rotation.from_euler('xyz', v).as_matrix())).item()

        if starting_v is None:
            if starting_i is None:
                raise RuntimeError('specify either starting_v or starting_i')
            starting_v = Rotation.from_matrix(cdbk.grider.grid[starting_i]).as_euler('xyz')

        res = minimize(fun, starting_v, tol=1e-2)
        result_v = res.x
        return Rotation.from_euler('xyz', result_v).as_matrix()

    def find_best_match_around(self, code, cdbk: Codebook, starting_v=None, starting_i=None):
        if starting_v is None:
            if starting_i is None:
                raise RuntimeError('specify either starting_v or starting_i')
            starting_v = Rotation.from_matrix(cdbk.grider.grid[starting_i]).as_euler('xyz')
        if torch.is_tensor(starting_v):
            sv = starting_v.cpu().numpy()
        else:
            sv = starting_v
        target = code
        top_score = 0
        for i in range(self.magic_grad_steps):
            dd = self.magic_shrink_constant / self.magic_shrink_exp ** i
            lin = np.linspace(-dd, dd, self.magic_nn_points)
            eulers = np.array([sv + np.array([dx, dy, dz]) for dx, dy, dz in itertools.product(lin, lin, lin)])
            rots = Rotation.from_euler('xyz', eulers).as_matrix()
            sims = cdbk.cos_sim(target, cdbk.latent_exact(rots))
            tops, idcs = torch.topk(sims, 10)
            idx = idcs.cpu().numpy().astype(np.int)
            # print(eulers[idx])
            sv = eulers[idx[0]]
            top_score = tops[0]
        # print('found ', top_score)
        return Rotation.from_euler('xyz', sv).as_matrix()

    @property
    def similarities(self):
        if self._simililarities is None:
            self.ae.eval()
            self._simililarities = torch.zeros((len(self.grider.grid), len(self.models), len(self.models)), device=self.ae.device)
            self._arg_simililarities = torch.zeros_like(self._simililarities)
            self._searched = torch.zeros_like(self._simililarities)
            self._eulers = torch.zeros((len(self.grider.grid), len(self.models), len(self.models), 3))

            for i in range(len(self.models)):
                for j in range(len(self.models)):
                    m_i = self.model_list[i]
                    m_j = self.model_list[j]
                    cdbk_i = self.codebooks[m_i]
                    cdbk_j = self.codebooks[m_j]

                    if i != j:
                        for k in trange(len(self.grider.grid)):
                            rot_k = self.grider.grid[k]
                            code = cdbk_i.latent_exact(rot_k)
                            scores = cdbk_j.cos_sim(code, norm1=True)
                            best = torch.argmax(scores)
                            self._arg_simililarities[k, i, j] = best
                            self._simililarities[k, i, j] = scores[best]
                            best_rot = self.find_best_match_around(code, cdbk_j, starting_i=best)
                            best_code = cdbk_j.latent_exact(best_rot)
                            self._searched[k, i, j] = cdbk_j.cos_sim(code, best_code)
                            # print('on grid is ', self._simililarities[k, i, j])
                            # print('searched is ', self._searched[k, i, j])
                            x = Rotation.from_matrix(best_rot).as_euler('xyz').flatten()
                            e = torch.from_numpy(x)
                            self._eulers[k, i, j, :] = e
                    else:
                        for k in trange(len(self.grider.grid)):
                            cos_sim = cdbk_j.cos_sim(cdbk_i.codebook[k], norm1=True)
                            self._simililarities[k, i, j] = torch.max(cos_sim[cos_sim < 1])

        return self._simililarities

    @property
    def labels(self):
        if self._labels is None:
            self._labels = torch.softmax(self.similarities, dim=2)

        return self._labels

    def save_codebooks(self, workdir):
        for mname, _ in self.models.items():
            self.codebooks[mname].save(workdir + '/' + 'codebook_%s.pt' % mname)

    def load_codebooks(self, workdir):
        for mname, _ in self.models.items():
            self.codebooks[mname].load(workdir + '/' + 'codebook_%s.pt' % mname)

    def save(self, workdir):
        self.save_codebooks(workdir)
        torch.save(self.labels, workdir + '/' + 'labels.pt')
        torch.save(self.similarities, workdir + '/' + 'similarities.pt')
        torch.save(self._eulers, workdir + '/' + 'eulers.pt')
        torch.save(self._searched, workdir + '/' + 'searched.pt')
        # torch.save(self._sorted, workdir + '/' + 'sorted.pt')
        torch.save(self.fin_labels, workdir + '/' + 'fin_labels.pt')

    def load(self, workdir):
        self.load_codebooks(workdir)
        self._labels = torch.load(workdir + '/' + 'labels.pt')
        self._simililarities = torch.load(workdir + '/' + 'similarities.pt')
        self._eulers = torch.load(workdir + '/' + 'eulers.pt')
        self._searched = torch.load(workdir + '/' + 'searched.pt')
        # self._sorted = torch.load(workdir + '/' + 'sorted.pt')
        self._fin_labels = torch.load(workdir + '/' + 'fin_labels.pt')

    @property
    def smooth_labels(self):
        if self._smoothen is None:
            self._smoothen = torch.zeros_like(self._searched)
            for i in range(len(self.models)):
                for j in range(len(self.models)):
                    if i != j:
                        for k in range(self._smoothen.shape[0]):
                            self._smoothen[k, i, j] = torch.median(self._searched[self.knn_index_grid[k], i, j])

        return self._smoothen

    @property
    def knn_index_grid(self):
        if self._knn_index_grid is None:
            self._knn_index_grid = torch.zeros((self.grider.grid.shape[0], self.knn_median), dtype=torch.long)
            grid_eulers = Rotation.from_matrix(self.grider.grid).as_euler('xyz')
            for i, eul in enumerate(grid_eulers):
                dif = np.abs(grid_eulers - eul)
                for j, maxdif in enumerate([2 * np.pi, np.pi, 2 * np.pi]):
                    dif[:, j] = np.min(np.array([dif[:, j], maxdif - dif[:, j]]), axis=0)
                torch_dif = torch.from_numpy(np.linalg.norm(dif, axis=1))
                self._knn_index_grid[i, :] = torch.topk(-torch_dif, self.knn_median)[1]

        return self._knn_index_grid

    def recalculate_fin_labels(self):
        self._sorted = torch.zeros_like(self.smooth_labels)
        self._fin_labels = torch.ones_like(self.labels)
        for i in range(len(self.models)):
            for j in range(len(self.models)):
                if i != j:
                    arg_sorted = torch.argsort(self.smooth_labels[:, i, j])
                    self._sorted[arg_sorted, i, j] = torch.linspace(0, 1, len(self._sorted[:, i, j]), device=self.ae.device)
                    self._fin_labels[:, i, j] = self.to_curve(self._sorted[:, i, j], self.borderline, self.width)
        self._fin_labels /= torch.sum(self._fin_labels, dim=2, keepdim=True)

    @property
    def fin_labels(self):
        if self._fin_labels is None:
            self.recalculate_fin_labels()
        return self._fin_labels

    @staticmethod
    def to_curve(ar, borderline, width):
        to_one = ar > borderline + width / 2
        to_zero = ar < borderline - width / 2
        to_interm = torch.logical_and(borderline - width / 2 < ar, ar < borderline + width / 2)
        res = torch.zeros_like(ar)
        res[to_one] = 1
        res[to_zero] = 0
        res[to_interm] = torch.sigmoid((ar[to_interm] - borderline) / (5 * width))
        return res

def render_and_save(ds, rot, path=None, rec_path=None, ae=None):
    img, _ = ds.objren.render_and_crop(rot)
    img = np.moveaxis(img, 2, 0) / 255.0
    t_img = torch.tensor(img)
    if rec_path:
        t_img_rec = ae.forward(t_img[None, :, :, :].cuda()).cpu()
        transforms.ToPILImage()(t_img_rec[0, :, :, :]).convert("RGB").save(rec_path)
    if path:
        img = transforms.ToPILImage()(t_img).convert("RGB")
        img.save(path)


def print_out_sim_views(labeler: AmbigousObjectsLabeler, models_names, i_from, i_to, top_n, wdir_save):
    cdbks = labeler.codebooks

    model_from = models_names[i_from]
    model_to = models_names[i_to]

    tops, idcs = torch.topk(-labeler.smooth_labels[:, i_from, i_to], top_n)

    print(tops)
    i_min = idcs[0].item()
    rot_min = labeler.grider.grid[i_min]


    if not os.path.exists(wdir_save):
        os.mkdir(wdir_save)

    for i, idx in enumerate(idcs):
        render_and_save(ods[model_from], rot=labeler.grider.grid[idx.item()], path=wdir_save + '/' + 'top%d.png' % i)
        euler_found = labeler._eulers[idx.item(), i_from, i_to, :]
        # print(euler_found)
        rot_found = Rotation.from_euler('xyz', euler_found).as_matrix()
        render_and_save(ods[model_to], rot=rot_found, path=wdir_save + '/' + 'top%d_f.png' % i)

if __name__ == "__main__":
    # models_dir = '/home/safoex/Downloads/cat_food/models_fixed/'
    # models_names = ['tonno_low', 'pollo', 'polpa']
    wdir_root = '/home/safoex/Documents/data/aae'
    models_dir = wdir_root + '/models/scans'
    models_names = ['fragola', 'pistacchi', 'tiramisu']
    models = {mname: {'model_path': models_dir + '/' + mname + '.obj'} for mname in models_names}
    workdir = wdir_root + '/test_labeler'
    if not os.path.exists(workdir):
        os.mkdir(workdir)
    grider = Grid(300, 1)

    ae = AAE(128, 256, (128, 256, 256, 512))
    ae_path = wdir_root + '/multi_boxes256.pth'
    ae.load_state_dict(torch.load(ae_path))

    ae.cuda()

    labeler = AmbigousObjectsLabeler(models, grider_label=grider, grider_codebook=Grid(1000, 10), ae=ae)
    # labeler.load_codebooks(workdir)
    labeler.load(workdir)
    # labeler.save(workdir)

    test_on = 'pistacchi'
    online_ds_fragola = OnlineRenderDataset(grider, models['fragola']['model_path'], camera_dist=None)
    online_ds = OnlineRenderDataset(grider, models[test_on]['model_path'], camera_dist=None)

    ods = {mname: OnlineRenderDataset(grider, models[mname]['model_path'], camera_dist=None) for mname in models_names}

    # euler = np.array([ 2.51397823, -1.1410451,  -0.72252894])

    # euler = np.array([ 0.53941419,  0.51667277, -2.09448489])
    # euler = np.array([2.194764,   0.23944807, 2.43638052])
    # euler = np.array([ 1.73685346,  0.39140481, -0.6275145 ])
    # euler = np.array([-0.52187396, -0.3788934,   2.31570534])

    for i_from, i_to in itertools.product(range(3), range(3)):
        if i_from != i_to:
            top_n = 300
            wdir_save = workdir + '/' + 'tops_%d_%d' % (i_from, i_to)
            labeler.knn_median = 10

            print_out_sim_views(labeler, models_names, i_from, i_to, top_n, wdir_save)

    # print(Rotation.from_matrix(rot_min).as_euler('xyz'))
    # color, depth = online_ds.objren.render(labeler.grider.grid[i_min])
    # online_ds.objren.test_show(color, depth)


    # ae_ideal = AAE(128, 128, (128, 256, 256, 512)).cuda()
    #
    # ae_ideal.load_state_dict(torch.load('/home/safoex/Documents/data/aae/cans_pth2/pollo/ae128.pth'))
    #
    # m_i = labeler.model_idx[test_on]
    # for i, x in enumerate(np.random.randint(0, len(grider.grid), 10)):
    #     rot = grider.grid[x]
    #     wdir = workdir + '/' + 'test_%d' % i
    #
    #     if os.path.exists(wdir):
    #         shutil.rmtree(wdir, ignore_errors=True)
    #
    #     os.mkdir(wdir)
    #
    #     label = str(labeler.fin_labels[x, m_i, :])
    #     render_and_save(online_ds, rot, wdir + '/' + 'rendered_p_%s.png' % label, wdir + '/' + 'rec.png', ae)
        # render_and_save(online_ds, rot, None, wdir + '/' + 'rec_ideal.png', ae_ideal)


