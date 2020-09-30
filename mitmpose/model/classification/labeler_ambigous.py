from mitmpose.model.classification.dataset import *
from mitmpose.model.pose.codebooks.codebook import Codebook
from mitmpose.model.pose.datasets.dataset import OnlineRenderDataset
import torch
import itertools
from tqdm.auto import trange, tqdm


class AmbigousObjectsLabeler:
    def __init__(self, models, grider, ae):
        self.models = models
        self.grider = grider
        self.ae = ae
        self._codebooks = None
        self._simililarities = None
        self.model_list = [m for m in self.models]
        self.model_idx = {m: i for i,m in enumerate(self.model_list)}
        self._labels = None


    @property
    def codebooks(self):
        if self._codebooks is None:
            self.ae.eval()
            self._codebooks = {}
            for mname, mprop in self.models.items():
                ds = OnlineRenderDataset(self.grider, mprop['model_path'], camera_dist=None)
                self._codebooks[mname] = Codebook(self.ae, ds)
                self._codebooks[mname].codebook

        return self._codebooks

    @property
    def similarities(self):
        if self._simililarities is None:
            self.ae.eval()
            self._simililarities = torch.ones((len(self.grider.grid), len(models), len(models)), device=self.ae.device)
            for i in range(len(models)):
                for j in range(len(models)):
                    if i != j:
                        m_i = self.model_list[i]
                        m_j = self.model_list[j]
                        cdbk_i = self.codebooks[m_i]
                        cdbk_j = self.codebooks[m_j]
                        for k in trange(len(self.grider.grid)):
                            self._simililarities[k, i, j] = torch.max(cdbk_j.cos_sim(cdbk_i.codebook[k], norm1=True))

        return self._simililarities

    @property
    def labels(self):
        if self._labels is None:
            self._labels = torch.softmax(self.similarities, dim=2)

        return self._labels

    def save(self, workdir):
        torch.save(self.labels, workdir + '/' + 'labels.pt')

    def load(self, workdir):
        self._labels = torch.load(workdir + '/' + 'labels.pt', map_location=self.ae.device)


if __name__ == "__main__":
    # models_dir = '/home/safoex/Downloads/cat_food/models_fixed/'
    # models_names = ['tonno_low', 'pollo', 'polpa']
    models_dir = '/home/safoex/Documents/data/aae/models/scans'
    models_names = ['fragola', 'pistacchi', 'tiramisu']
    models = {mname: {'model_path': models_dir + '/' + mname + '.obj'} for mname in models_names}
    workdir = '/home/safoex/Documents/data/aae/test_labeler_boxes'
    grider = Grid(100, 10)

    ae = AAE(128, 128, (128, 256, 256, 512))
    ae_path = '/home/safoex/Documents/data/aae/models/scans/multi_boxes_128.pth'
    ae.load_state_dict(torch.load(ae_path))

    ae.cuda()

    labeler = AmbigousObjectsLabeler(models, grider, ae)

    labeler.load(workdir)

    online_ds = OnlineRenderDataset(grider, models['fragola']['model_path'], camera_dist=None)
    import numpy as np
    import os
    import shutil


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

    ae_ideal = AAE(128, 128, (128, 256, 256, 512)).cuda()

    ae_ideal.load_state_dict(torch.load('/home/safoex/Documents/data/aae/cans_pth2/pollo/ae128.pth'))

    m_i = labeler.model_idx['fragola']
    for i, x in enumerate(np.random.randint(0, len(grider.grid), 10)):
        rot = grider.grid[x]
        wdir = workdir + '/' + 'test_%d' % i

        if os.path.exists(wdir):
            shutil.rmtree(wdir, ignore_errors=True)

        os.mkdir(wdir)

        label = str(labeler.labels[x, m_i, :])
        render_and_save(online_ds, rot, wdir + '/' + 'rendered_%s.png' % label, wdir + '/' + 'rec.png', ae)
        render_and_save(online_ds, rot, None, wdir + '/' + 'rec_ideal.png', ae_ideal)


