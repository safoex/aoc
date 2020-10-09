from mitmpose.model.classification.dataset import ManyObjectsRenderedDataset, Grid
from mitmpose.model.classification.labeler_ambigous import AmbigousObjectsLabeler


class ManyAmbigousObjectsLabeledRenderedDataset(ManyObjectsRenderedDataset):
    def __init__(self, grider: Grid, grider_codebook: Grid, models: dict, joint_ae, aae_render_tranform, classification_transform=None, res=128, camera_dist=None,
                 render_res=640, intensity_render=10, intensity_augment=(2, 20), n_aug_workers=8, online=False):
        super().__init__(grider, models, aae_render_tranform, classification_transform, res, camera_dist,
                         render_res, intensity_render, intensity_augment, n_aug_workers, online)
        grider_labeler = grider
        if grider.samples_in_plane > 1:
                grider_labeler = Grid(grider.samples_sphere, 1)
        self.labeler = AmbigousObjectsLabeler(models, grider_labeler, grider_codebook, joint_ae)
        self._labels = None

    @property
    def fin_labels(self):
        if self._labels is None:
            self._labels = self.labeler.fin_labels[:, None, :, :]
            if self.grider.samples_in_plane > 1:
                self._labels = self._labels.repeat(1, self.grider.samples_in_plane, 1, 1)

        return self._labels

    def create_dataset(self, folder):
        super(ManyAmbigousObjectsLabeledRenderedDataset, self).create_dataset(folder)
        self.labeler.save(folder)

    def load_dataset(self, folder):
        super(ManyAmbigousObjectsLabeledRenderedDataset, self).load_dataset(folder)
        self.labeler.load(folder)

    def __getitem__(self, idx):
        obj = None
        for obj, ds in self.datasets.items():
            if idx >= len(ds):
                idx -= len(ds)
            else:
                break

        assert obj is not None

        if self.mode == 'class':
            obj_id = self.labeler.model_idx[obj]
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
    grider = Grid(300, 1)
    grider_codebook = Grid(1000, 10)

    ae = AAE(128, 256, (128, 256, 256, 512))
    # ae_path = '/home/safoex/Documents/data/aae/cans_pth2/multi128.pth'
    ae_path = '/home/safoex/Documents/data/aae/multi_boxes256.pth'
    ae.load_state_dict(torch.load(ae_path))

    ae.cuda()

    ds = ManyAmbigousObjectsLabeledRenderedDataset(grider, grider_codebook, models, ae,
                                                   aae_render_tranform=AAETransform(0.5, '/home/safoex/Documents/data/VOCtrainval_11-May-2012', add_aug=False))
    ds.labeler.load(workdir)
    ds.create_dataset(workdir)

