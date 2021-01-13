print('i can print')

from mitmpose.routines.test.inference.offline_experiment import InferenceOfflineExperiment, Grid
import torch



workdir = '/home/safoex/Documents/data/aae/release2/release2'
models_dir = '/home/safoex/Documents/data/aae/models/scans/'
models_names = ['meltacchin', 'melpollo', 'humana1', 'humana2']
models = {mname: {'model_path': models_dir + '/' + mname + '.obj', 'camera_dist': None} for mname in models_names}
grider = Grid(300, 20)
# grider_tc = Grid(100, 5)
classes = {'babyfood': ['meltacchin', 'melpollo'],
           'babymilk': ['humana1', 'humana2']}


device = torch.device('cuda:0')
print('ioe')
ioe = InferenceOfflineExperiment(workdir, models_dir, models_names, grider, device=device, classes=classes)
print('ioe.load_dataset()')
ioe.load_dataset()
# ioe.tc.hcl.save_global_classifier()
print('ioe.tc.hcl.load_global_classifier()')
ioe.tc.hcl.load_global_classifier()
ioe.set_experimental_data('/home/safoex/Documents/data/aae/10_12_2020/', [0.3, 0.3, 0.3, 0.3])

for i in range(6, 11):
    threshold = i / 10.0
    print('%d' % i)
    max_epochs = int(1.5 / threshold + 0.01)
    max_epochs = min((3, max_epochs))
    ioe.tc.hcl.save_local_classifiers(extra_folder="local_%.2f" % threshold, threshold=threshold, max_epochs=max_epochs)
    ioe.tc.hcl.load_local_classifiers(extra_folder="local_%.2f" % threshold)
    ioe.load_hcl_and_inference(extra_folder_local="local_%.2f" % threshold)
    ioe.load_or_render_codebooks()
    ioe.test_and_save_results(extra_folder_results="local_%.2f" % threshold, max_imgs_each_class=None)
