from mitmpose.model.classification.dataset_hierarchical import HierarchicalManyObjectsDataset, AAETransform, Grid
from mitmpose.model.classification.classifier_hierarchical import HierarchicalClassifier
import os
import numpy as np
from torchvision import transforms
import torch


class Experiment:
    def __init__(self, workdir, models_dir, models_names, grider, aae_params=(128, 256, (128, 256, 256, 512)),
                 bg_path='/home/safoex/Documents/data/VOCtrainval_11-May-2012', device=None, classes=None):
        self.workdir = workdir
        self.models_dir = models_dir
        self.models_names = models_names
        self.grider = grider
        self.aae_params = aae_params
        self.bg_path = bg_path
        self.device = device
        self.models = {mname: {'model_path': models_dir + '/' + mname + '.obj', 'camera_dist': None} for mname in
                       models_names}
        self.classes = classes

    def global_class_of(self, local_class):
        for gcl, gcl_list in self.classes.items():
            if local_class in gcl_list:
                return gcl
        return None


class TrainClassifiers(Experiment):
    def __init__(self, workdir, models_dir, models_names, grider, aae_params=(128, 256, (128, 256, 256, 512)),
                 bg_path='/home/safoex/Documents/data/VOCtrainval_11-May-2012', device=None,
                 classes=None, grider_labeled_size=300):
        super().__init__(workdir, models_dir, models_names, grider, aae_params, bg_path, device, classes)
        self.ds = HierarchicalManyObjectsDataset(self.grider, self.models, res=236,
                                                 classification_transform=HierarchicalManyObjectsDataset.transform_normalize,
                                                 aae_render_transform=AAETransform(0.5,
                                                                                   self.bg_path,
                                                                                   add_patches=False, add_aug=False,
                                                                                   size=(236, 236)),
                                                 aae_scale_factor=1.5)
        # self.ds = HierarchicalManyObjectsDataset(self.grider, self.models, aae_render_transform=AAETransform(0.5,
        #                                                                       self.bg_path,
        #                                                                       add_patches=True))

        self.hcl = HierarchicalClassifier(self.workdir, self.ds, ambiguous_aae_params=self.aae_params,
                                          global_aae_params=self.aae_params,
                                          device=device, grider_labeled_size=grider_labeled_size)

    def create_dataset(self):
        self.ds.create_dataset(self.workdir)

    def load_dataset(self):
        self.ds.make_hierarchical_dataset(
            self.classes
        )
        self.ds.load_dataset(self.workdir)
        self.hcl.manual_set_classes(self.classes)
        self.hcl.load_labelers()

    def train_all_classifiers(self, threshold=0.6):
        # you can also use
        # self.hcl.save_local_classifier(...)
        # and
        # self.hcl.save_global_classifier(...)
        # individually after calling self.load_dataset()
        self.hcl.save_global_classifier()
        self.hcl.save_local_classifiers(threshold=threshold)

    def printout_classes(self, samples_each_class=50):
        for i, cl in enumerate(self.classes):
            if not os.path.exists(self.workdir + '/%d' % i):
                os.mkdir(self.workdir + '/%d' % i)

        for x in np.random.randint(0, len(self.ds), samples_each_class):
            img, label = self.ds[x]
            transforms.ToPILImage()(img).convert("RGB").save(self.workdir + '/%d/%d.png' % (label, x))

if __name__ == '__main__':
    workdir = '/home/safoex/Documents/data/aae/release2/release2'
    models_dir = '/home/safoex/Documents/data/aae/models/scans/'
    models_names = ['meltacchin', 'melpollo', 'humana1', 'humana2']
    models = {mname: {'model_path': models_dir + '/' + mname + '.obj', 'camera_dist': None} for mname in models_names}
    grider = Grid(300, 20)
    classes = {'babyfood': ['meltacchin', 'melpollo'],
               'babymilk': ['humana1', 'humana2']}

    device = torch.device('cuda:0')

    tc = TrainClassifiers(workdir, models_dir, models_names, grider, device=device, classes=classes)

    # tc.create_dataset()

    tc.load_dataset()

    # tc.printout_classes()

    # tc.hcl.save_global_classifier()
    tc.hcl.save_local_classifiers(threshold=0.3, max_epochs=5)

    print(len(tc.hcl.dataset))
