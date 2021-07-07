from .classify import InferenceClassifier, Codebook, OnlineRenderDataset, Grid
import torch
import os
import pickle
from aoc.routines.train.classifiers import Experiment, TrainClassifiers
from tqdm import tqdm

class InferenceOfflineExperiment(Experiment):
    def __init__(self, workdir, models_dir, models_names, grider, aae_params=(128, 256, (128, 256, 256, 512)),
                 bg_path='/home/safoex/Documents/data/VOCtrainval_11-May-2012', device=None,
                 classes=None, grider_labeled_size=300):
        super().__init__(workdir, models_dir, models_names, grider, aae_params, bg_path, device, classes)
        self.tc = TrainClassifiers(workdir, models_dir, models_names, grider, aae_params, bg_path, device, classes, grider_labeled_size=grider_labeled_size)
        self.inference = None
        self.path_to_exp_data_folder = None
        self.radiuses = None
        self.cache = {}

    def load_dataset(self):
        self.tc.load_dataset()

    def load_hcl_and_inference(self, extra_folder_global="", extra_folder_local="", extra_cache=None):
        self.tc.hcl.load_everything(extra_folder_global, extra_folder_local)
        cache = extra_cache or self.cache 
        self.inference = InferenceClassifier(self.tc.hcl, self.device, cache=cache)

    def load_or_render_codebooks(self, cdbk_grider=Grid(4000, 40)):
        for gcl, subclasses in self.classes.items():
            for lcl in subclasses:
                self.inference.cdbks[lcl] = Codebook(self.tc.hcl.aaes[gcl],
                                                     OnlineRenderDataset(cdbk_grider, self.models[lcl]['model_path']))
                path_to_cdbk = self.workdir + '/' + gcl + '/' + 'codebook_%s.pt' % lcl
                if os.path.exists(path_to_cdbk):
                    self.inference.cdbks[lcl].load(path_to_cdbk)
                else:
                    self.inference.cdbks[lcl].save(path_to_cdbk)

    def set_experimental_data(self, path_to_exp_data_folder, radiuses=None):
        if radiuses is None:
            radiuses = [0.3] * len(self.models_names)

        self.radiuses = radiuses
        self.path_to_exp_data_folder = path_to_exp_data_folder


    def test_and_save_results(self, extra_folder_results="", max_imgs_each_class=None, print_intermediate_results=True):
        results = {

        }
        for model_name, radius in tqdm(list(zip(self.models_names, self.radiuses))[0:]):
            if print_intermediate_results:
                print('----------%s--------' % model_name)
            if max_imgs_each_class is None:
                n = len(os.listdir(self.path_to_exp_data_folder + '%s/rad_%.2f' % (model_name, radius)))
            else:
                n = max_imgs_each_class
            test_imgs = [
                self.path_to_exp_data_folder + '%s/rad_%.2f/image_%d.png' % (model_name, radius, i) for i in range(1, n)
            ]
            results[model_name] = []
            for i, img_path in tqdm(enumerate(test_imgs)):
                with torch.no_grad():
                    # threshold is not really used when "assume_global_class" parameter is set
                    result = self.inference.classify(img_path, threshold=0.4,
                                                     assume_global_class=self.global_class_of(model_name), cache=self.cache)
                    results[model_name].append(result)
                    if print_intermediate_results:
                        print(result, i)

        with open(self.workdir + '/' + extra_folder_results + '/' + 'results.pickle', 'wb') as f:
            pickle.dump(results, f)
