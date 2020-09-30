from mitmpose.model.classification.dataset import ManyObjectsRenderedDataset, Grid
from mitmpose.model.classification.labeler_ambigous import AmbigousObjectsLabeler


class ManyAmbigousObjectsLabeledRenderedDataset(ManyObjectsRenderedDataset):
    def __init__(self, grider: Grid, models: dict, joint_ae, aae_render_tranform, classification_transform=None, res=128, camera_dist=None,
                 render_res=640, intensity_render=10, intensity_augment=(2, 20), n_aug_workers=8):
        super().__init__(grider, models, aae_render_tranform, classification_transform, res, camera_dist,
                         render_res, intensity_render, intensity_augment, n_aug_workers)

        self.labeler = AmbigousObjectsLabeler(models, grider, joint_ae)

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
            return self.transform(self.datasets[obj][idx][-1]), self.labeler.labels[idx, obj_id, :]
        else:
            return self.datasets[obj][idx]


if __name__ == '__main__':
    from mitmpose.model.pose.aae.aae import AAE, AAETransform
    import torch
    models_dir = '/home/safoex/Downloads/cat_food/models_fixed/'
    models_names = ['tonno_low', 'pollo', 'polpa']
    models = {mname: {'model_path': models_dir + '/' + mname + '.obj'} for mname in models_names}
    workdir = '/home/safoex/Documents/data/aae/test_labeled_dataset'
    grider = Grid(1000, 10)

    ae = AAE(128, 128, (128, 256, 256, 512))
    ae_path = '/home/safoex/Documents/data/aae/cans_pth2/multi128.pth'
    ae.load_state_dict(torch.load(ae_path))

    ae.cuda()

    ds = ManyAmbigousObjectsLabeledRenderedDataset(grider, models, ae,
                                                   aae_render_tranform=AAETransform(0.5, '/home/safoex/Documents/data/VOCtrainval_11-May-2012', add_aug=False))

    ds.create_dataset(workdir)

