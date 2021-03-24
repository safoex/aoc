from mitmpose.model.pose.grids.grids import Grid
models_dir = '/home/safoex/Documents/data/aae/models/scans'
models_names = ['fragola', 'pistacchi', 'tiramisu']
models = {mname: {'model_path': models_dir + '/' + mname + '.obj'} for mname in models_names}
example_model_path =  models[models_names[0]]['model_path']
workdir = '/home/safoex/Documents/data/aae/tests'
voc_path = '/home/safoex/Documents/data/VOCtrainval_11-May-2012'

default_train_grider = Grid(3000, 20)
default_inference_grider = Grid(3000, 40)
default_small_grider = Grid(100, 10)
default_super_small_grider = Grid(10, 3)

# print('Warning! Did you change paths in /routines/test/settings.py according to your setup?!')

