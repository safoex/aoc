from mitmpose.model.classification.dataset import ManyObjectsRenderedDataset, Grid
from mitmpose.model.classification.labeler_ambigous import AmbigousObjectsLabeler
import torch
import numpy as np
import itertools
from torchvision import transforms


class ManyAmbigousObjectsLabeledRenderedDataset(ManyObjectsRenderedDataset):
    def __init__(self, grider: Grid, grider_codebook: Grid, models: dict, joint_ae, aae_render_tranform, classification_transform=None, res=128, camera_dist=None,
                 render_res=640, intensity_render=10, intensity_augment=(2, 20), online=False, keep_top_threshold=None,
                 keep_fraction=1):
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
        self.fraction = keep_fraction

    def __len__(self):
        if self._len is not None:
            return self._len
        else:
            return super().__len__()

    def keep_top(self, threshold=1):
        top_labels, _ = torch.max(self.fin_labels, dim=2)
        self._idcs = torch.nonzero(top_labels >= threshold)
        self._len = len(self._idcs)

    def get_top_ndim(self, a, top_n):
        _, i = torch.topk(a.flatten(), top_n)
        return (np.array(np.unravel_index(i.numpy(), a.shape)).T)

    def keep_fraction(self, fraction=1):
        sorted, _ = torch.max(self.labeler._sorted, dim=2)
        if isinstance(fraction, tuple):
            indices = torch.nonzero(torch.logical_and(sorted >= fraction[0] , sorted <= fraction[1]))
        else:
            indices = torch.nonzero(sorted <= fraction)

        self._idcs = torch.repeat_interleave(indices, self.grider.samples_in_plane, dim=0)
        self._idcs[:, 0] *= self.grider.samples_in_plane
        for x in range(self.grider.samples_in_plane):
            for z in range(indices.shape[0]):
                self._idcs[x+z*self.grider.samples_in_plane, 0] += x

        self._len = len(self._idcs)

    def recalculate_fin_labels(self):
        self._labels = self.labeler.fin_labels
        if self.grider.samples_in_plane > 1:
            # self._labels = self._labels.repeat(self.grider.samples_in_plane, 1, 1)
            self._labels = torch.repeat_interleave(self._labels, repeats=self.grider.samples_in_plane, dim=0)

        if self.top_threshold is not None and self.top_threshold > 0:
            self.keep_top(self.top_threshold)

        if self.top_threshold is None:
            if isinstance(self.fraction, tuple) or self.fraction < 1:
                self.keep_fraction(self.fraction)
            n = len(self.labeler.model_list)

            for i, j in itertools.product(range(n), range(n)):
                self._labels[:, i, j] = float(i == j)

    @property
    def fin_labels(self):
        if self._labels is None:
            self.recalculate_fin_labels()

        return self._labels

    def create_dataset(self, folder):
        super(ManyAmbigousObjectsLabeledRenderedDataset, self).create_dataset(folder)
        self.labeler.save(folder)
        self.recalculate_fin_labels()

    def load_dataset(self, folder):
        super(ManyAmbigousObjectsLabeledRenderedDataset, self).load_dataset(folder)
        self.labeler.load(folder)
        self.recalculate_fin_labels()

    def __getitem__(self, idx):
        obj = None
        obj_id = 0
        if self.mode == 'class' and (self.top_threshold is not None and self.top_threshold > 0 or isinstance(self.fraction, tuple) or self.fraction < 1):
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
    cltrans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(236),
        transforms.RandomCrop(224),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    aae_transform = AAETransform(0.5, '/home/safoex/Documents/data/VOCtrainval_11-May-2012', add_aug=False, add_patches=False)
    ds = ManyAmbigousObjectsLabeledRenderedDataset(grider, grider_codebook, models, ae,
                                                   aae_render_tranform=aae_transform,
                                                   classification_transform=cltrans,
                                                   keep_fraction=0.7)
    ds.labeler.load(workdir)
    ds.labeler.recalculate_fin_labels()
    ds.create_dataset(workdir)

    for i, idx in enumerate(np.random.randint(0, len(ds), 10)):
        img = ds[idx][0].cpu()
        # img = np.moveaxis(img, 0, 2) * 255.0
        img = transforms.ToPILImage()(img).convert("RGB")
        img.save(workdir + '/test%d.png' % i)

    from mitmpose.model.classification.labeler_ambigous import render_and_save

    # for i, idx in enumerate(np.random.randint(0, ds._idcs.shape[0], 20)):
    #     ii = ds._idcs[idx][0]
    #     obj = ds.labeler.model_list[0]
    #     render_and_save(ds.datasets[obj].reconstruction_dataset, rot=ds.grider.grid[ii], path=workdir + '/TEST%d.png'%i)





