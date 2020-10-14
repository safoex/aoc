from mitmpose.model.classification.dataset import ManyObjectsRenderedDataset, Grid
from mitmpose.model.classification.labeler_ambigous import AmbigousObjectsLabeler
import torch
import numpy as np


class ManyAmbigousObjectsLabeledRenderedDataset(ManyObjectsRenderedDataset):
    def __init__(self, grider: Grid, grider_codebook: Grid, models: dict, joint_ae, aae_render_tranform, classification_transform=None, res=128, camera_dist=None,
                 render_res=640, intensity_render=10, intensity_augment=(2, 20), online=False, keep_top_threshold=0):
        super().__init__(grider, models, aae_render_tranform, classification_transform, res, camera_dist,
                         render_res, intensity_render, intensity_augment, online)
        grider_labeler = grider
        if grider.samples_in_plane > 1:
                grider_labeler = Grid(grider.samples_sphere, 1)
        self.labeler = AmbigousObjectsLabeler(models, grider_labeler, grider_codebook, joint_ae)
        self._labels = None

        self._len = None
        self._idcs = None
        self.top_threshold = keep_top_threshold

    def __len__(self):
        if self._len is not None:
            return self._len
        else:
            return super().__len__()

    def keep_top(self, threshold=1):
        top_labels, _ = torch.max(self.fin_labels, dim=2)
        self._idcs = torch.nonzero(top_labels >= threshold)
        self._len = len(self._idcs)

    @property
    def fin_labels(self):
        if self._labels is None:
            self._labels = self.labeler.fin_labels
            if self.grider.samples_in_plane > 1:
                # self._labels = self._labels.repeat(self.grider.samples_in_plane, 1, 1)
                self._labels = torch.repeat_interleave(self._labels, repeats=self.grider.samples_in_plane, dim=0)
            if self.top_threshold > 0:
                self.keep_top(self.top_threshold)
        return self._labels

    def create_dataset(self, folder):
        super(ManyAmbigousObjectsLabeledRenderedDataset, self).create_dataset(folder)
        self.labeler.save(folder)
        if self.top_threshold > 0:
            self.keep_top(self.top_threshold)

    def load_dataset(self, folder):
        super(ManyAmbigousObjectsLabeledRenderedDataset, self).load_dataset(folder)
        self.labeler.load(folder)
        if self.top_threshold > 0:
            self.keep_top(self.top_threshold)

    def __getitem__(self, idx):

        obj = None
        obj_id = 0
        if self.mode == 'class' and self.top_threshold > 0:
            idx, obj_id = self._idcs[idx]
            obj = self.labeler.model_list[obj_id]
        else:
            for obj, ds in self.datasets.items():
                if idx >= len(ds):
                    idx -= len(ds)
                else:
                    break
            assert obj is not None
            obj_id = self.labeler.model_idx[obj]

        if self.mode == 'class':
            ds_idx = -1
            if self.online:
                ds_idx = 0
            return self.transform(self.datasets[obj][idx][ds_idx]), self.fin_labels[idx, obj_id, :]
        else:
            return self.datasets[obj][idx]


if __name__ == '__main__':
    from mitmpose.model.pose.aae.aae import AAE, AAETransform
    import torch
    # models_dir = '/home/safoex/Downloads/cat_food/models_fixed/'
    # models_names = ['tonno_low', 'pollo', 'polpa']
    models_dir = '/home/safoex/Documents/data/aae/models/scans'
    models_names = ['fragola', 'pistacchi', 'tiramisu']

    models = {mname: {'model_path': models_dir + '/' + mname + '.obj'} for mname in models_names}
    workdir = '/home/safoex/Documents/data/aae/test_labeler'
    grider = Grid(300, 3)
    grider_codebook = Grid(1000, 10)

    ae = AAE(128, 256, (128, 256, 256, 512))
    # ae_path = '/home/safoex/Documents/data/aae/cans_pth2/multi128.pth'
    ae_path = '/home/safoex/Documents/data/aae/multi_boxes256.pth'
    ae.load_state_dict(torch.load(ae_path))

    ae.cuda()

    aae_transform = AAETransform(0.5, '/home/safoex/Documents/data/VOCtrainval_11-May-2012', add_aug=False)
    ds = ManyAmbigousObjectsLabeledRenderedDataset(grider, grider_codebook, models, ae,
                                                   aae_render_tranform=aae_transform, keep_top_threshold=1)
    ds.labeler.load(workdir)
    ds.load_dataset(workdir)

    for idx in np.random.randint(0, len(ds), 10):
        print(ds[idx][1])





