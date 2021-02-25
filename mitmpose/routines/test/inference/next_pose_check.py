from mitmpose.routines.test.inference.next_pose import *
from mitmpose.model.classification.dataset_hierarchical import *

class FakeResponse:
    def __init__(self, recorded_data_dir):
        self.angular_limits = np.load(recorded_data_dir + '/angles.npy')
        self.feasibles = np.load(recorded_data_dir + '/feasibles.npy')
        self.traj = TrajectoryGenerator(self.angular_limits.tolist(), self.feasibles)
        self.image_paths = [recorded_data_dir + '/image_%d.png' % (i + 1) for i in range(len(self.traj.rots))]
        print("Total feasibles are: %d" % np.sum(self.feasibles))

    def get_i(self, idx):
        return self.image_paths[idx], self.traj.rots[idx]

    def gen_random_idx(self):
        return np.random.randint(0, len(self.traj.rots), 1)[0]

    def get_random(self):
        idx = self.gen_random_idx()
        print(idx)
        return self.get_i(idx)

    def get_idx_of(self, rot):
        return np.argmin(np.linalg.norm(np.array(self.traj.rots) - rot, axis=(1,2)))

    def get_by_rot(self, rot):
        return self.get_i(self.get_idx_of(rot))


workdir = "/home/safoex/Documents/data/aae/release3"


assumed_class = 'babyfood'
testclass = 'melpollo'


panda_dir = "/home/safoex/Documents/data/aae/panda_data_10_02_2021/%s/%s_2/rad_0.35" % (testclass, testclass)

fr = FakeResponse(panda_dir)

device = torch.device('cuda:0')
classes = {'babyfood': ['meltacchin', 'melpollo'],
           'babymilk': ['humana1', 'humana2']}

models_dir = '/home/safoex/Documents/data/aae/models/scans/'
models_names = ['meltacchin', 'melpollo', 'humana1', 'humana2']
models = {mname: {'model_path': models_dir + '/' + mname + '.obj', 'camera_dist': None} for mname in models_names}
# grider = Grid(100, 5)
grider = Grid(300, 20)
ds = HierarchicalManyObjectsDataset(grider, models, res=236,
                                    classification_transform=HierarchicalManyObjectsDataset.transform_normalize,
                                    aae_render_transform=AAETransform(0.5,
                                                                      '/home/safoex/Documents/data/VOCtrainval_11-May-2012',
                                                                      add_patches=False, size=(236, 236)))

nipple = NextPoseProvider(workdir, ds, fr.traj.rots, device, classes)
nipple.load_everything("", "")
nipple.load_or_render_codebooks()



checkdir = panda_dir + '/check/'
if not os.path.exists(checkdir):
    os.mkdir(checkdir)

# for i_, img_path in enumerate(fr.image_paths):
#     i = i_ + 1
#     nipple.icl.log_steps(img_path, checkdir + 'img_%d_a.png' % i, checkdir + 'img_%d_crop.png' % i,
#                         checkdir + 'img_%d_rec.png' % i, checkdir + 'img_%d' % i + '_z_%d.png', assume_global_class=assumed_class)



succeded = 0
improved = 0
worsen = 0
total = 0
incorrect = 0

ambiguity_threshold = 0.3
jump_limit = 6
for _ in range(100):
    ambiguity = 1
    tabs = ''
    next_rot = fr.traj.rots[fr.gen_random_idx()]
    initial_class = None
    final_class = None
    try:
        j = 0
        while ambiguity > ambiguity_threshold and j < jump_limit:
            image_path, rot = fr.get_by_rot(next_rot)
            result = nipple.classify(image_path, rot, assumed_class, ambiguity_threshold, next_random=False)
            print(tabs, result[0], result[1][1] if result[1] is not None else -1)
            if initial_class is None:
                initial_class = result[0][4]
            if result[1] is not None:
                (expected_ambiguity, next_rot), next_i = result[1]
                print(tabs, expected_ambiguity)
                tabs += '\t'
                j += 1
            else:
                ambiguity = 0
                final_class = result[0][4]
            if j == jump_limit:
                final_class = result[0][4]
    except:
        pass

    total += 1

    if final_class == testclass:
        succeded += 1

    if final_class is not None:
        if initial_class != testclass and final_class == testclass:
            improved += 1

        if initial_class == testclass and final_class != testclass:
            worsen += 1

        if final_class != testclass:
            incorrect += 1

print('total : %d' % total)
print('succeded : %d' % succeded)
print('incorrect : %d' % incorrect)
print('improved : %d' % improved)
print('worsen : %d' % worsen)
