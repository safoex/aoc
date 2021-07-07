from torch.utils.data import Dataset
from aoc.model.pose.grids.grids import Grid
from aoc.model.pose.datasets.augment import AAETransform
from aoc.model.pose.datasets.dataset import OnlineRenderDataset, RenderedDataset, AugmentedAndRenderedDataset
from aoc.model.pose.aae.aae import AEDataModule, AAE
from torchvision import transforms
import pytorch_lightning as pl
from aoc.model.classification.dataset import ManyObjectsRenderedDataset
from aoc.model.classification.dataset_with_subset import ManyObjectsDatasetWithSubset
import copy
import os
import numpy as np


class HierarchicalManyObjectsDataset(ManyObjectsRenderedDataset):
    def __init__(self, grider: Grid, models: dict, aae_render_transform, classification_transform=None, res=128,
                 camera_dist=None,
                 render_res=640, intensity_render=10, intensity_augment=(2, 20), online=False, aae_scale_factor=1.2):
        self.params = {
            'grider': grider,
            'models': models,
            'aae_render_tranform': aae_render_transform,
            'classification_transform': classification_transform,
            'res': res,
            'camera_dist': camera_dist,
            'render_res': render_res,
            'intensity_render': intensity_render,
            'intensity_augment': intensity_augment,
            'online': online,
            'aae_scale_factor': aae_scale_factor
        }
        super().__init__(**self.params)
        self.classes = None
        self.mode = 'class'
        self.online = True # trick for get item

    def get_dataset(self, models_subset, grid_subsets=None):
        new_models_dict = {m: self.models[m] for m in models_subset}
        new_params = copy.deepcopy(self.params)
        new_params['models'] = new_models_dict
        new_params['grid_subsets'] = grid_subsets
        return ManyObjectsDatasetWithSubset(**new_params)

    def make_hierarchical_dataset(self, classes):
        self._datasets = {}
        self.classes = classes
        for i, (cl, msubset) in enumerate(classes.items()):
            new_params = copy.deepcopy(self.params)
            new_params['models'] = {m: self.models[m] for m in classes[cl]}
            self._datasets[cl] = ManyObjectsRenderedDataset(**new_params)
            self._datasets[cl].set_mode('class')
            self.labels[cl] = i

    def create_dataset(self, folder):
        self.get_dataset(self.models).create_dataset(folder)

    def load_dataset(self, folder):
        if self.classes is not None:
            for cl, ds in self._datasets.items():
                ds.load_dataset(folder)



if __name__ == "__main__":
    models_dir = '/home/safoex/Documents/data/aae/models/scans/'
    models_names = ['cow', 'melpollo', 'fragola', 'pistacchi']
    models = {mname: {'model_path': models_dir + '/' + mname + '.obj', 'camera_dist': None} for mname in models_names}
    workdir = '/home/safoex/Documents/data/aae/hierarchical'
    grider = Grid(300, 5)
    ds = HierarchicalManyObjectsDataset(grider, models, res=236,
                                    aae_render_transform=AAETransform(0.5,
                                                                     '/home/safoex/Documents/data/VOCtrainval_11-May-2012',
                                                                     add_patches=False, size=(236, 236)))
    # ds.set_mode('aae')
    ds.create_dataset(workdir)

    classes ={'boxes': ['meltacchin', 'melpollo'],
         'bottles': ['humana1', 'humana2']}
    ds.make_hierarchical_dataset(
        classes
    )

    ds.load_dataset(workdir)
    print(ds.mode)



    # for i, cl in enumerate(classes):
    #     if not os.path.exists(workdir + '/%d' % i):
    #         os.mkdir(workdir + '/%d' % i)
    #
    # for x in np.random.randint(0, len(ds), 50):
    #     img, label = ds[x]
    #     transforms.ToPILImage()(img).convert("RGB").save(workdir + '/%d/%d.png' % (label, x))

    # trainer = pl.Trainer(gpus=1, max_epochs=100)
    #
    # dm = AEDataModule(ds, workdir)
    # ae = AAE(128, 64, (32, 64, 64, 128))
    #
    # trainer.fit(ae, dm)
