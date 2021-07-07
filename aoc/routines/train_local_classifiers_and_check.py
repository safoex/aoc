print('i can print')

from aoc.routines.test.inference.offline_experiment import InferenceOfflineExperiment, Grid
import torch
import os


prefix = '/home/IIT.LOCAL/esafronov/aaedata/'
workdir = prefix + 'release9/'
models_dir = prefix + '/models_fixed/'
exp_data_path = '/home/IIT.LOCAL/esafronov/aaedata/panda_10_02_2021/'

models_names = ['pistacchi', 'fragola', 'cioccolato', 'vaniglia']
models = {mname: {'model_path': models_dir + '/' + mname + '.obj', 'camera_dist': None} for mname in models_names}
grider = Grid(300, 20)
# grider_tc = Grid(100, 5)
classes = {'redboxes': [ 'pistacchi', 'fragola'],
           'yellowboxes': ['cioccolato', 'vaniglia']}


gpu_n = 2
device = torch.device('cuda:%d' % gpu_n)
print('ioe')
bg_path = '~/VOC'
ioe = InferenceOfflineExperiment(workdir, models_dir, models_names, grider, device=device, classes=classes, bg_path=bg_path, grider_labeled_size=100)
print('ioe.load_dataset()')

ioe.tc.create_dataset()
ioe.load_dataset()

#ioe.tc.hcl.save_global_classifier(gpu_n=gpu_n)
print('ioe.tc.hcl.load_global_classifier()')
ioe.tc.hcl.load_global_classifier()
ioe.set_experimental_data(exp_data_path, [0.35, 0.35, 0.35, 0.35])

#for max_epochs in range(1, 4):
#    for subset_fraction_int in range(2, 6):
#        subset_fraction = subset_fraction_int / 10.0
#        ioe.tc.hcl.save_global_classifier(extra_folder="global_%d_%.1f" % (max_epochs, subset_fraction),
#                                          max_epochs=max_epochs, subset_fraction=subset_fraction, gpu_n=gpu_n)

for t in range(3, 20):
    threshold = t / 20.0
    print('%d' % t)
    # max_epochs = int(1.5 / threshold + 0.01)
    # max_epochs = min((3, max_epochs))
    for max_epochs in range(4, 5):
        folder = "6_03_local_%d_%.2f" % (max_epochs, threshold)
        # folder_save = "local_%d_%.2f" % (max_epochs, threshold)
        # if not os.path.exists(folder_save):
            # os.mkdir(folder_save)
        ioe.tc.hcl.save_local_classifiers(extra_folder=folder, threshold=threshold, max_epochs=max_epochs, gpu_n=gpu_n)
        ioe.tc.hcl.load_local_classifiers(extra_folder=folder)
        ioe.load_hcl_and_inference(extra_folder_local=folder)
        ioe.load_or_render_codebooks()
        ioe.test_and_save_results(extra_folder_results=folder, max_imgs_each_class=None, print_intermediate_results=False)
