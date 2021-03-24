from mitmpose.routines.test.inference.next_pose import *
from mitmpose.model.classification.dataset_hierarchical import *
import shutil
from torchvision import transforms as T
import pickle
from tqdm import trange

if __name__ == "__main__":
    workdir = "/home/safoex/Documents/data/aae/release3"

    assumed_class = 'babyfood'
    testclass = 'melpollo'

    # panda_dir = "/home/safoex/Documents/data/aae/panda_data_10_02_2021/%s/%s_0/rad_0.35" % (testclass, testclass)
    classes = {'babyfood': ['meltacchin', 'melpollo'],
               'babymilk': ['humana1', 'humana2']}

    models_dir = '/home/safoex/Documents/data/aae/models/scans/'
    models_names = ['meltacchin', 'melpollo', 'humana1', 'humana2']
    models = {mname: {'model_path': models_dir + '/' + mname + '.obj', 'camera_dist': None} for mname in models_names}

    device = torch.device('cuda:0')
    # grider = Grid(100, 5)
    grider = Grid(300, 20)
    ds = HierarchicalManyObjectsDataset(grider, models, res=236,
                                        classification_transform=HierarchicalManyObjectsDataset.transform_normalize,
                                        aae_render_transform=AAETransform(0.5,
                                                                          '/home/safoex/Documents/data/VOCtrainval_11-May-2012',
                                                                          add_patches=False, size=(236, 236)))
    hcl = HierarchicalClassifier(workdir, ds, ambiguous_aae_params=(128, 256, (128, 256, 256, 512)),
                                 global_aae_params=(128, 256, (128, 256, 256, 512)),
                                 device=device, grider_labeled_size=300)

    hcl.manual_set_classes(classes)
    hcl.load_everything("", "")

    icl = InferenceClassifier(hcl, device)

    cdbk_grider = Grid(4000, 40)

    for gcl, subclasses in classes.items():
        for lcl in subclasses:
            icl.cdbks[lcl] = Codebook(hcl.aaes[gcl],
                                      OnlineRenderDataset(cdbk_grider, ds.models[lcl]['model_path']))
            path_to_cdbk = workdir + '/' + gcl + '/' + 'codebook_%s.pt' % lcl
            if os.path.exists(path_to_cdbk):
                icl.cdbks[lcl].load(path_to_cdbk)
            else:
                icl.cdbks[lcl].save(path_to_cdbk)

    bag_prefix = '/home/safoex/Desktop/esafronov/bags_data/%d/'
    bag_img_pattern = bag_prefix + '/img%05d.jpg'
    bag_pickle_format = bag_prefix + '/results.pickle'

    global_classes = ['babyfood', 'babymilk', 'babyfood']
    t_start = time.time()
    for bag_n in range(3):
        filenames = os.listdir(bag_prefix % bag_n)
        images_n = sum(fn[-3:] == 'jpg' for fn in filenames)
        print(images_n)
        results = []
        for i in trange(images_n):
            try:
                bbox = icl.closest_by_size(bag_img_pattern % (bag_n, i))
                res = icl.classify2(bag_img_pattern % (bag_n, i), threshold=0.0, assume_global_class=None, cache={})
                results.append((bbox, res))
            except:
                results.append((None, None))
        with open(bag_pickle_format % bag_n, 'wb') as pf:
            pickle.dump(results, pf)
    t_end = time.time()
