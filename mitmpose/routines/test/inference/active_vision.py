from mitmpose.routines.test.inference.next_pose_check import *
import sys, pickle

#testclass = 'melpollo'

prefix = '/home/icub/esafronov/'
workdir = prefix + '/release'
#panda_dir = "/home/safoex/Documents/data/aae/panda_data_10_02_2021/%s/%s_0/rad_0.35" % (testclass, testclass)



device = torch.device('cuda:0')
classes = {'babyfood': ['meltacchin', 'melpollo'],
           'babymilk': ['humana1', 'humana2']}

models_dir = prefix + '/models'
models_names = ['meltacchin', 'melpollo', 'humana1', 'humana2']
models = {mname: {'model_path': models_dir + '/' + mname + '.obj', 'camera_dist': None} for mname in models_names}


if __name__ == "__main__":
    
    workdir_exp = prefix + '/data/' + sys.argv[1] + '/'
    recorded_data_dir = workdir_exp + 'rad_0.25'

    assumed_class = ''
    for gcl, lcls in classes.items():
        if sys.argv[1] in lcls:
            assumed_class = gcl


    
    fr = FakeResponse(recorded_data_dir)
    #print(fr.feasibles)
    
    
    n_exps = int(sys.argv[2])

    # grider = Grid(100, 5)
    grider = Grid(300, 20)
    ds = HierarchicalManyObjectsDataset(grider, models, res=236,
                                        classification_transform=HierarchicalManyObjectsDataset.transform_normalize,
                                        aae_render_transform=AAETransform(0.5,
                                                                          prefix + '/VOC',
                                                                          add_patches=False, size=(236, 236)))

    nipple = NextPoseProvider(workdir, ds, fr.traj.rots, device, classes, fr=fr)
    nipple.load_everything("", "")
    nipple.load_or_render_codebooks()

    av_test_dir = workdir_exp + '/av_test_%d/' % n_exps

    exp_dir_pattern = av_test_dir + 'exp_%d/'
    base_dir_pattern = av_test_dir + 'base_%d/'
    jump_limit = 1e5
    sleep_between = 1
    tests = 10
    ambiguity = 1
    tabs = ''
    ambiguity_threshold = 0.01
    ambiguity_threshold_av = 0.25
    
    with_baseline = False

    for t_ in range(2 * tests):
        if sleep_between > 0:
            time.sleep(sleep_between)
        initial_class = None
        final_class = None
        j = 0
        
        t = t_ // 2

        if t_ % 2 == 0:
            next_random = False
            dir_pattern = exp_dir_pattern
        else:
            if with_baseline:
                next_random = True
                dir_pattern = base_dir_pattern
            else:
                continue

        image_pattern = dir_pattern % t + '/image_%d.png'
        input_idx_pattern = dir_pattern % t + '/input_idx_%d.npy'
        response_idx_pattern = dir_pattern % t + '/output_idx_%d.npy'

            
        ambiguity = 1
        tabs = ""
        #try:
        results = []
        while ambiguity > ambiguity_threshold and j < jump_limit:
            image_path = image_pattern % j
            idx_path = input_idx_pattern % j
            print('wait for:')
            print(image_path)
            print(idx_path)
            idx = -1
            while not os.path.exists(image_path) or not os.path.exists(idx_path):
                if os.path.exists(idx_path):
                    idx = np.load(idx_path).tolist()
                    print(idx)
                    if idx < 0:
                        break
                time.sleep(0.1)
            time.sleep(0.1)
            idx = np.load(idx_path).tolist()
            if idx < 0:
                break

            print("idx is %d" % idx)
            rot = fr.traj.rots[fr.traj.glob_to_feas[idx]]

            result = nipple.classify(image_path, rot, assumed_class, ambiguity_threshold, next_random=next_random,
                                     first_i=idx)
            print(tabs, result[0], result[1][1] if result[1] is not None else -1)
            
            results.append(result)
            
            if result[0] is not None:
                ambiguity = 0.5*np.sum([result[0][2], result[0][3]])

            if result[1] is not None:
                (expected_ambiguity, next_rot), next_i, best_poses, all_scores = result[1]
                print(tabs, expected_ambiguity)
                all_scores = sorted(all_scores, key=lambda x: x[0])
                print(all_scores[:5])
                sorted_idcs = [fr.traj.feas_to_glob[kdx] for a, kdx in all_scores]
                if ambiguity > ambiguity_threshold_av:
                    np.save(response_idx_pattern % j, np.array(sorted_idcs))
                else:
                    np.save(response_idx_pattern % j, np.array(idx * np.ones_like(sorted_idcs, dtype=np.int)))
                tabs += '\t'
                # all_best_poses.append((idx, best_poses))
                j += 1
            else:
                ambiguity = 0
                np.save(response_idx_pattern % j, -1)
            
                        
        with open(exp_dir_pattern % t + '/results.pickle', 'wb') as f:
            pickle.dump(results, f)
        #except:
        #    pass
