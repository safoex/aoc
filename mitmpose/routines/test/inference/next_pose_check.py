from mitmpose.routines.test.inference.next_pose import *
from mitmpose.model.classification.dataset_hierarchical import *
import shutil
from torchvision import transforms as T


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
        return np.argmin(np.linalg.norm(np.array(self.traj.rots) - rot, axis=(1, 2)))

    def get_by_rot(self, rot):
        return self.get_i(self.get_idx_of(rot))


workdir = "/home/safoex/Documents/data/aae/release3"

assumed_class = 'babyfood'
testclass = 'melpollo'

panda_dir = "/home/safoex/Documents/data/aae/panda_data_10_02_2021/%s/%s_0/rad_0.35" % (testclass, testclass)

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

nipple = NextPoseProvider(workdir, ds, fr.traj.rots, device, classes, fr=fr)
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
jump_limit = 1
test_folder = panda_dir + '/next_corrected_camera_frame/'
if not os.path.exists(test_folder):
    os.mkdir(test_folder)

axis = ['x', 'y', 'z']

# pose = special_ortho_group.rvs(3)
# match_img, mask = nipple.icl.cdbks[classes[assumed_class][0]]._ds.objren.render_and_crop(pose)
# T.ToPILImage()(torch.from_numpy(np.moveaxis(match_img, 2, 0) / 255.0)).save(test_folder + "_ident.png")
#
# for i_axis, ax in enumerate(axis):
#     eulers = [0, 0, 0]
#     eulers[i_axis] = np.pi/2
#     delta_pose = Rotation.from_euler('xyz', eulers).as_matrix()
#     pose_end = pose.T.dot(delta_pose).T
#     match_img, mask = nipple.icl.cdbks[classes[assumed_class][0]]._ds.objren.render_and_crop(pose_end)
#     T.ToPILImage()(torch.from_numpy(np.moveaxis(match_img, 2, 0) / 255.0)).save(test_folder + "_ident_%s.png" % ax)

all_best_poses = []
for idx in range(100):
    ambiguity = 1
    tabs = ''
    next_rot = fr.traj.rots[idx]
    initial_class = None
    final_class = None
    j = 0
    try:
        while ambiguity > ambiguity_threshold and j < jump_limit:
            image_path, rot = fr.get_by_rot(next_rot)
            result = nipple.classify(image_path, rot, assumed_class, ambiguity_threshold, next_random=False, first_i=idx)
            print(tabs, result[0], result[1][1] if result[1] is not None else -1)
            if initial_class is None:
                initial_class = result[0][4]
            if result[1] is not None:
                (expected_ambiguity, next_rot), next_i, best_poses = result[1]
                print(tabs, expected_ambiguity)
                tabs += '\t'
                shutil.copy(fr.image_paths[idx], test_folder + 'img_%d_in.png' % (idx + 1))
                shutil.copy(fr.image_paths[next_i], test_folder + 'img_%d_ne.png' % (idx + 1))
                all_best_poses.append((idx, best_poses))
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
#
#

for idx, best_poses in all_best_poses:
    for i_, pose in enumerate(best_poses):
        match_img, mask = nipple.icl.cdbks[classes[assumed_class][i_]]._ds.objren.render_and_crop(pose)
        T.ToPILImage()(torch.from_numpy(np.moveaxis(match_img, 2, 0) / 255.0)).save(test_folder + 'img_%d_re_%d.png' % (idx + 1, i_))
