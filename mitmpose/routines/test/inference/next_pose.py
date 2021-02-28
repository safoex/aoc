from mitmpose.model.classification.classifier_hierarchical import HierarchicalClassifier
from mitmpose.routines.test.inference.classify import InferenceClassifier
from mitmpose.model.pose.codebooks.codebook import *

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import os
import shutil

def correct_camera_rot(camera_rot):
    camera_euler = Rotation.from_matrix(camera_rot).as_euler('xyz')
    camera_euler[0] -= np.pi
    return Rotation.from_euler('xyz', camera_euler).as_matrix()


def camera_frame(camera_rot, object_rots):
    return np.matmul(np.expand_dims(correct_camera_rot(camera_rot).T, 0), object_rots)

def object_frame(camera_rot, object_rots):
    return np.matmul(np.expand_dims(correct_camera_rot(camera_rot), 0), object_rots)

def normalize_rows(mat_to_normalize):

    return mat_to_normalize / np.linalg.norm(mat_to_normalize, axis=1, keepdims=True)


class TrajectoryGenerator:
    def __init__(self, angular_limits, feasibles):
        min_theta, max_theta, theta_steps, min_phi, max_phi, phi_steps = angular_limits
        theta = np.linspace(min_theta, max_theta, int(theta_steps))
        phi = np.linspace(min_phi, max_phi, int(phi_steps))

        theta_space = np.empty((0,))
        phi_space = np.empty((0,))
        for idx in range(phi.size):
            theta_space = np.append(theta_space, theta)
            phi_space = np.append(phi_space, phi[idx] * np.ones_like(theta))

        theta_space = np.reshape(theta_space, (1, theta_space.size))
        phi_space = np.reshape(phi_space, (1, phi_space.size))

        # Define sphere center
        center = [0.45, 0.0, 0.07]
        rad = 1
        # Obtain pose center points
        x = rad * np.cos(theta_space) * np.cos(phi_space) + center[0]
        y = rad * np.sin(theta_space) * np.cos(phi_space) + center[1]
        z = rad * np.sin(phi_space) + center[2]

        # find the x axis by looking at center of sphere
        # find the y axis by finding the derivative of x and y wrt theta
        # find the z axis by cross product

        points = np.hstack((np.transpose(x), np.transpose(y), np.transpose(z)))

        z_ax = np.reshape(center, (1, 3)) - points
        z_ax = normalize_rows(z_ax)

        y_ax_x = -np.sin(theta_space)
        y_ax_y = np.cos(theta_space)
        y_ax_z = np.zeros_like(y_ax_x)

        y_ax = np.hstack((np.transpose(y_ax_x), np.transpose(y_ax_y), np.transpose(y_ax_z)))
        y_ax = normalize_rows(y_ax)

        x_ax = np.cross(y_ax, z_ax)
        x_ax = normalize_rows(x_ax)

        self.rots = []
        self.orig_rots = []

        self.glob_to_feas = {}
        self.feas_to_glob = {}
        fidx = 0
        for idx in range(x_ax.shape[0]):
            rot_matrix = np.identity(3)
            rot_matrix = np.transpose(np.stack((
                x_ax[idx, :],
                y_ax[idx, :],
                z_ax[idx, :]
            )))
            self.orig_rots.append(rot_matrix)
            if feasibles[idx]:
                self.rots.append(rot_matrix)
                self.glob_to_feas[idx] = fidx
                self.feas_to_glob[fidx]= idx
                fidx += 1


# test_folder + 'img_%d_re_%d.png' % (idx + 1, i_)
from torchvision import  transforms as T


class NextPoseProvider:
    def __init__(self, workdir, ds, possible_orientations, device, classes, aae_params=(128, 256, (128, 256, 256, 512)), grider_labeled_size=300, fr=None):
        self.workdir = workdir
        self.ds = ds
        self.aae_params = aae_params
        self.hcl = HierarchicalClassifier(self.workdir, self.ds, ambiguous_aae_params=self.aae_params,
                                          global_aae_params=self.aae_params,
                                          device=device, grider_labeled_size=grider_labeled_size)
        self.cache = {}
        self.icl = InferenceClassifier(self.hcl, device, self.cache)
        self.possible_orientations = possible_orientations
        self.classes = classes
        self.fr = fr

    def load_or_render_codebooks(self, cdbk_grider=Grid(4000, 40)):
        for gcl, subclasses in self.classes.items():
            for lcl in subclasses:
                self.icl.cdbks[lcl] = Codebook(self.hcl.aaes[gcl],
                                                     OnlineRenderDataset(cdbk_grider, self.ds.models[lcl]['model_path']))
                path_to_cdbk = self.workdir + '/' + gcl + '/' + 'codebook_%s.pt' % lcl
                if os.path.exists(path_to_cdbk):
                    self.icl.cdbks[lcl].load(path_to_cdbk)
                else:
                    self.icl.cdbks[lcl].save(path_to_cdbk)

    def print_rendered_pair(self, rots, save_path, assumed_class):
        for i_, rot in enumerate(rots):
            match_img, mask = self.icl.cdbks[self.classes[assumed_class][i_]]._ds.objren.render_and_crop(rot)
            T.ToPILImage()(torch.from_numpy(np.moveaxis(match_img, 2, 0) / 255.0)).save(save_path % i_)

    def print_rendered(self, rot, save_path, assumed_class, lcln):
        match_img, mask = self.icl.cdbks[self.classes[assumed_class][lcln]]._ds.objren.render_and_crop(rot)
        T.ToPILImage()(torch.from_numpy(np.moveaxis(match_img, 2, 0) / 255.0)).save(save_path)

    def load_everything(self, extra_folder_local, extra_folder_global=""):
        self.hcl.manual_set_classes(self.classes)
        self.hcl.load_everything(extra_folder_global, extra_folder_local)

    def classify(self, image_path, robot_orientation, assume_global_class=None, ambiguity_threshold=0.4, next_random=True, first_i=None):
        # orientation as matrix
        res = self.icl.classify2(image_path, assume_global_class=assume_global_class, cache=self.cache)
        if res is None:
            return None, None
        result, orientation_hypothesis = res

        if sum((result[2], result[3]))/2 > ambiguity_threshold:
            return result, self.next_pose_(orientation_hypothesis, robot_orientation, assume_global_class, next_random, first_i=first_i)
        else:
            return result, None

    def next_pose_random_(self):
        idcs = list(range(0, len(self.possible_orientations)))
        idcs_return = np.random.permutation(idcs)
        #idx = np.random.randint(0, len(self.possible_orientations), 1)[0]
        idx = idcs_return[0]
        all_scores = [(1, jdx) for jdx in idcs_return]
        best_poses = None
        rot = self.possible_orientations[idx]

        #(expected_ambiguity, next_rot), next_i, best_poses, all_scores
        return (1, rot), idx, best_poses, all_scores

    def next_pose_(self, orientation_hypothesis, robot_orientation, global_class, next_random, first_i):
        if next_random:
            return self.next_pose_random_()
        n_subclasses = self.hcl.labelers[global_class]._sorted.shape[2]
        min_ambig_pose = (1, None)
        min_i = None

        lcls = self.classes[global_class]
        best_expected_poses = [None, None]

        testclass = 'melpollo'
        panda_dir = "/home/safoex/Documents/data/aae/panda_data_10_02_2021/%s/%s_0/rad_0.35" % (testclass, testclass)
        nmoves_dir = panda_dir + '/next_moves/'
        #if not os.path.exists(nmoves_dir):
        #    os.mkdir(nmoves_dir)
        all_scores = []

        for k, new_camera_pose in enumerate(self.possible_orientations):
            if k % 5 != 0:
                continue
            max_score = 0
            possible_object_poses = [None, None]
            for identity, object_orientation in orientation_hypothesis:
                # corrected_object_pose = camera_frame(robot_orientation, object_orientation)
                # possible_object_pose = object_frame(new_camera_pose, corrected_object_pose)

                # camera_pose_ = correct_camera_rot(new_camera_pose)
                # robot_orientation_ = correct_camera_rot(robot_orientation)
                # possible_object_pose = camera_pose_.T.dot(robot_orientation_.dot(object_orientation))
                # possible_object_pose = new_camera_pose.dot(robot_orientation.T.dot(object_orientation))
                delta_robot = robot_orientation.T.dot(new_camera_pose)
                # print('delta_robot:')
                # print(delta_robot)
                # print()
                # print('-----------')
                # print()
                Rx = Rotation.from_euler('xyz', [np.pi, 0, 0]).as_matrix()
                Rz = Rotation.from_euler('xyz', [0, 0, -np.pi/2]).as_matrix()
                Rextra = Rx.dot(Rz)
                possible_object_pose = object_orientation.T.dot(Rextra).dot(delta_robot).dot(Rextra.T).T
                # possible_object_pose = object_orientation.T.dot(delta_robot).T
                # possible_object_pose = np.eye(3).dot(object_orientation.T).T
                possible_object_poses[identity] = possible_object_pose
                # print(identity)

                # self.print_rendered(possible_object_pose, nmoves_dir + '/im_' + str(k + 1) + '_x%d.png' % identity,
                #                          global_class, identity)

                # if identity == 0:
                #     self.print_rendered(possible_object_pose, nmoves_dir + '/im_' + str(k + 1) + '_nex.png',
                #                         global_class, identity)
                #     self.print_rendered(object_orientation, nmoves_dir + '/im_' + str(k + 1) + '_ini.png' ,
                #                         global_class, identity)

                close_idcs = self.icl.get_closest_without_inplane(possible_object_pose,
                                                                  self.hcl.labelers[global_class].grider.grid, 1)

                scores = torch.cat(
                    tuple(self.hcl.labelers[global_class]._sorted[close_idcs, identity, j] for j in range(n_subclasses) if identity != j))
                # print(scores)
                if torch.max(scores) > max_score:
                    max_score = torch.max(scores)

            # shutil.copy(self.fr.image_paths[first_i], nmoves_dir + 'im_%d_in.png' % (k + 1))
            # shutil.copy(self.fr.image_paths[k], nmoves_dir + 'im_%d_ne.png' % (k + 1))
            #
            # for _id in range(2):
            #     self.print_rendered(self.fr.traj.rots[k], nmoves_dir + '/im_' + str(k + 1) + '_x%d.png' % (2 + _id),
            #                     global_class, _id)

            # res2 = self.icl.classify2(self.fr.image_paths[k], assume_global_class=global_class, cache=self.cache)
            # if res2 is not None:
            #     result2, orientation_hypothesis2 = res2
            #     # print(orientation_hypothesis2)
            #     obj_r_list = [None, None]
            #     for _i, _or in orientation_hypothesis:
            #         obj_r_list[_i] = _or
            #
            #     obj_r_list2 = [None, None]
            #     for _i, _or in orientation_hypothesis2:
            #         obj_r_list2[_i] = _or
            #
            #     obj_delta = [or2.T.dot(or1) for or1, or2 in zip(obj_r_list, obj_r_list2)]
            #     camera_delta = new_camera_pose.T.dot(robot_orientation)
            #     print('*****************************************')
            #     print()
            #     def print_rot(rot):
            #         print(["%.3f"%e for e in Rotation.from_matrix(rot).as_euler('xyz')])
            #
            #     for obj_i, obj_d in enumerate(obj_delta):
            #         print("obj_delta%d: " % obj_i)
            #         print(print_rot(obj_d))
            #         print()
            #     print('camera_delta: ')
            #     print(print_rot(camera_delta))
            #     print()
            #     print()

            # self.print_rendered_pair(possible_object_poses, nmoves_dir + '/im_' + str(k+1) + '_x%d.png', global_class)
            all_scores.append((max_score, k))
            if max_score < min_ambig_pose[0]:
                min_ambig_pose = (max_score, new_camera_pose)
                min_i = k
                best_expected_poses = possible_object_poses

        # print("H: ")
        # print(robot_orientation)
        #
        # print("H_next: ")
        # print(min_ambig_pose[1])
        #
        # print("H0: ")
        # for _, object_pose in orientation_hypothesis:
        #     print(object_pose)
        #
        # print()
        # print('--------------------------')
        # print()


        # print("best next image is ", min_i, self.possible_orientations[min_i])

        return min_ambig_pose, min_i, best_expected_poses, all_scores
