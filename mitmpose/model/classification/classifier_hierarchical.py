import torchvision
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from mitmpose.model.classification.dataset import ManyObjectsRenderedDataset
from mitmpose.model.classification.dataset_ambigous import ManyAmbigousObjectsLabeledRenderedDataset
from mitmpose.model.classification.classifier import ObjectClassifierDataModule, ObjectClassifier
from mitmpose.model.pose.grids.grids import Grid
from mitmpose.model.pose.datasets.augment import AAETransform
from mitmpose.model.classification.dataset_hierarchical import HierarchicalManyObjectsDataset, \
    ManyObjectsDatasetWithSubset
from mitmpose.model.pose.aae.aae import AAE, AEDataModule
from mitmpose.model.classification.classifier import ObjectClassifier, ObjectClassifierDataModule
from mitmpose.model.classification.labeler_ambigous import AmbigousObjectsLabeler
import pytorch_lightning as pl
import os
import numpy as np
from torchvision import transforms as T
from scipy.spatial.transform import Rotation


class HierarchicalClassifier:
    def __init__(self, workdir, dataset: HierarchicalManyObjectsDataset,
                 global_aae_params=(), ambiguous_aae_params=(),
                 get_classifier=lambda: torchvision.models.resnet18(pretrained=True), device=None):
        self.dataset = dataset
        self.global_aae_params = global_aae_params
        self.ambiguous_aae_params = ambiguous_aae_params
        self.get_classifier = get_classifier
        self.global_aae = AAE(*global_aae_params)
        self.aaes = {}
        self.workdir = workdir
        self.classes = {}
        self.global_classes = []
        self.global_classifier = None
        self.in_class_classifiers = {}
        self.labelers = {}
        self.device = device

    def train_aae(self, ae: AAE, ds, save_path: str, trainer_params, batch_size=64, num_workers=8):
        dm = AEDataModule(ds, self.workdir, batch_size=batch_size, num_workers=num_workers)
        trainer = pl.Trainer(**trainer_params)
        trainer.fit(ae, dm)
        torch.save(ae.state_dict(), save_path)

    def save_global_aae(self, trainer_params, batch_size=64, num_workers=8):
        self.train_aae(self.global_aae, self.dataset, self.workdir + '/' + 'global%d.pth' % self.global_aae.latent_size,
                       trainer_params, batch_size, num_workers)

    def load_aae(self, ae: AAE, save_path: str, device=None):
        ae.load_state_dict(torch.load(save_path))
        if device or self.device:
            ae.to(device= (device or self.device) )

    def load_global_aae(self, device=None):
        self.load_aae(self.global_aae, self.workdir + '/' + 'global%d.pth' % self.global_aae.latent_size, device)

    def manual_set_classes(self, classes: dict):
        self.classes = classes
        self.global_classes = [key for key in self.classes]

    def class_workdir(self, class_name):
        wdir = self.workdir + '/' + class_name
        if not os.path.exists(wdir):
            os.mkdir(wdir)
        return wdir

    def save_local_aae(self, global_class, trainer_params, batch_size=64, num_workers=8):
        ds = self.dataset.get_dataset(self.classes[global_class])
        class_workdir = self.class_workdir(global_class)
        aae = self.aaes[global_class] = AAE(*self.ambiguous_aae_params)
        self.train_aae(aae, ds, class_workdir + '/' + 'local%d.pth' % aae.latent_size, trainer_params, batch_size, num_workers)

    def load_local_aae(self, global_class, path=None, device=None):
        aae = self.aaes[global_class] = AAE(*self.ambiguous_aae_params)
        if path is None:
            path = self.class_workdir(global_class) + '/' + 'local%d.pth' % aae.latent_size
        self.load_aae(aae, path, device)

    def save_global_classifier(self):
        self.global_classifier = ObjectClassifier(len(self.classes), freeze_conv=True)
        trainer = pl.Trainer(gpus=1, max_epochs=1)

        ocdm = ObjectClassifierDataModule(self.dataset)

        trainer.fit(self.global_classifier, ocdm)
        torch.save(self.global_classifier.state_dict(), self.workdir + '/' + 'global.pth')

    def load_global_classifier(self):
        self.global_classifier = ObjectClassifier(len(self.classes)).to(self.device)
        self.global_classifier.load_state_dict(torch.load(self.workdir + '/' + 'global.pth'))

    def debug_print_dataset(self, ds, testdir, n):
        if not os.path.exists(testdir):
            os.mkdir(testdir)

        for idx in np.random.randint(0, len(ds), n):
        # for idx in range(n):
            img, lbl = ds[idx]
            (T.ToPILImage()(img.cpu())).save(testdir + 'img_%d_%d.png' % (idx, lbl))


    def save_local_classifiers(self, threshold=0.2):
        self.in_class_classifiers = {
            cl: ObjectClassifier(len(subclasses), freeze_conv=False) for cl, subclasses in self.classes.items()
        }
        for cl, subclasses in self.classes.items():
            trainer = pl.Trainer(gpus=1, max_epochs=4)
            # TODO: remove assumptions that there are two objects

            def is_valid_rotation(rot, lbler):
                rot_idx = self.get_closest_without_inplane(rot, self.labelers[cl].grider.grid).item()
                return self.labelers[cl]._sorted[rot_idx, lbler, 1 - lbler] < threshold

            grid_subsets = torch.stack([torch.tensor([is_valid_rotation(rot, ilbler) for rot in self.dataset.grider.grid]) for ilbler in range(2)])

            # grid_subsets = torch.stack((self.labelers[cl]._sorted[:, 0, 1] > 1 - threshold, self.labelers[cl]._sorted[:, 1, 0] > 1 - threshold))

            ds = self.dataset.get_dataset(subclasses, grid_subsets)
            ds.load_dataset(self.workdir)
            self.debug_print_dataset(ds, '/home/safoex/Documents/data/aae/panda_data/test/' + cl + '/', 100)
            ocdm = ObjectClassifierDataModule(ds)

            trainer.fit(self.in_class_classifiers[cl], ocdm)
            torch.save(self.in_class_classifiers[cl].state_dict(), self.workdir + '/' + '%s.pth' % cl)

    def load_local_classifiers(self):
        self.in_class_classifiers = {
            cl: ObjectClassifier(len(subclasses)).to(self.device) for cl, subclasses in self.classes.items()
        }
        for cl in self.classes:
            self.in_class_classifiers[cl].load_state_dict(torch.load(self.workdir + '/' + '%s.pth' % cl))

    def get_xyz(self, grid):
        a = Rotation.from_matrix(grid).as_euler('xyz')
        if len(a.shape) < 2:
            a = a.reshape((1, 3))
        x = np.zeros_like(a)
        r = 1
        x[:, 0] = r * np.sin(a[:, 0]) * np.cos(a[:, 1])
        x[:, 1] = r * np.sin(a[:, 0]) * np.sin(a[:, 1])
        x[:, 2] = r * np.cos(a[:, 0])
        return x

    def get_closest_without_inplane(self, rot, grid, top_k=1):
        gxyz = torch.from_numpy(self.get_xyz(grid))

        rxyz = torch.from_numpy(self.get_xyz(rot)).view(1, 3)
        return torch.topk(-torch.norm(gxyz - rxyz, dim=1), top_k)[1]

    def load_labelers(self):
        models = self.dataset.models
        for gcl in self.classes:
            models_subset = {m: models[m] for m in models if m in self.classes[gcl]}
            workdir_gcl = self.workdir + '/' + gcl
            self.load_local_aae(gcl, workdir_gcl + '/' + 'multi256.pth', device=self.device)
            self.labelers[gcl] = AmbigousObjectsLabeler(models_subset, grider_label=Grid(300, 1),
                                                        grider_codebook=Grid(4000, 40),
                                                        ae=self.aaes[gcl])
            self.labelers[gcl].load(workdir_gcl)
            self.labelers[gcl].recalculate_fin_labels()

    def load_everything(self):
        self.load_global_classifier()
        self.load_local_classifiers()

        self.load_labelers()





if __name__ == '__main__':
    workdir = '/home/safoex/Documents/data/aae/release2/release'
    models_dir = '/home/safoex/Documents/data/aae/models/scans/'
    models_names = ['meltacchin', 'melpollo', 'humana1', 'humana2']
    models = {mname: {'model_path': models_dir + '/' + mname + '.obj', 'camera_dist': None} for mname in models_names}
    grider = Grid(300, 20)
    ds = HierarchicalManyObjectsDataset(grider, models, res=236, classification_transform=HierarchicalManyObjectsDataset.transform_normalize,
                                        aae_render_transform=AAETransform(0.5,
                                                                          '/home/safoex/Documents/data/VOCtrainval_11-May-2012',
                                                                          add_patches=False, add_aug=False, size=(236, 236)),
                                        aae_scale_factor=1.5)
    ds = HierarchicalManyObjectsDataset(grider, models, aae_render_transform=AAETransform(0.5,
                                                                          '/home/safoex/Documents/data/VOCtrainval_11-May-2012',
                                                                          add_patches=True))

    # ds.set_mode('aae')
    # ds.create_dataset(workdir)

    classes = {'babyfood': ['meltacchin', 'melpollo'],
               'babymilk': ['humana1', 'humana2']}
    ds.make_hierarchical_dataset(
        classes
    )


    device = torch.device('cuda:0')
    ds.load_dataset(workdir)
    aae_params = (128, 256, (128, 256, 256, 512))
    hcl = HierarchicalClassifier(workdir, ds, ambiguous_aae_params=aae_params, global_aae_params=aae_params, device=device)

    hcl.manual_set_classes(classes)


    hcl.load_labelers()

    # hcl.save_global_classifier()
    hcl.save_local_classifiers()

    print(len(hcl.dataset))
    testdir = '/home/safoex/Documents/data/aae/panda_data/test/global_ds/'
    if not os.path.exists(testdir):
        os.mkdir(testdir)

    for idx in np.random.randint(0, len(hcl.dataset), 40):
        img, lbl = hcl.dataset[idx]
        (T.ToPILImage()(img.cpu())).save(testdir + 'img_%d_%d.png' % (idx, lbl))

