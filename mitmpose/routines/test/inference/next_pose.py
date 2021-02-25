from mitmpose.model.classification.classifier_hierarchical import HierarchicalClassifier
from mitmpose.routines.test.inference.classify import InferenceClassifier
from mitmpose.model.pose.codebooks.codebook import *

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import os


def correct_camera_rot(camera_rot):
    camera_euler = Rotation.from_matrix(camera_rot).as_euler('xyz')
    # camera_euler[0] -= np.pi
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
        for idx in range(x_ax.shape[0]):
            rot_matrix = np.identity(3)
            rot_matrix = np.transpose(np.stack((
                x_ax[idx, :],
                y_ax[idx, :],
                z_ax[idx, :]
            )))
            if feasibles[idx]:
                self.rots.append(rot_matrix)


class NextPoseProvider:
    def __init__(self, workdir, ds, possible_orientations, device, classes, aae_params=(128, 256, (128, 256, 256, 512)), grider_labeled_size=300):
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

    def load_everything(self, extra_folder_local, extra_folder_global=""):
        self.hcl.manual_set_classes(self.classes)
        self.hcl.load_everything(extra_folder_global, extra_folder_local)

    def classify(self, image_path, robot_orientation, assume_global_class=None, ambiguity_threshold=0.4, next_random=True):
        # orientation as matrix
        result, orientation_hypothesis = self.icl.classify2(image_path, assume_global_class=assume_global_class, cache=self.cache)
        if sum((result[2], result[3]))/2 > ambiguity_threshold:
            return result, self.next_pose_(orientation_hypothesis, robot_orientation, assume_global_class, next_random)
        else:
            return result, None

    def next_pose_random_(self):
        idx = np.random.randint(0, len(self.possible_orientations), 1)[0]
        rot = self.possible_orientations[idx]
        return (1, rot), idx

    def next_pose_(self, orientation_hypothesis, robot_orientation, global_class, next_random):
        if next_random:
            return self.next_pose_random_()
        n_subclasses = self.hcl.labelers[global_class]._sorted.shape[2]
        min_ambig_pose = (1, None)
        min_i = None
        for k, camera_pose in enumerate(self.possible_orientations):
            if k % 5 != 0:
                continue
            max_score = 0
            for identity, object_orientation in orientation_hypothesis:
                # corrected_object_pose = object_frame(robot_orientation, object_orientation)
                # possible_object_pose = object_frame(camera_pose, corrected_object_pose)
                possible_object_pose = camera_frame(camera_pose, object_orientation)
                close_idcs = self.icl.get_closest_without_inplane(possible_object_pose,
                                                                  self.hcl.labelers[global_class].grider.grid, 1)
                scores = torch.cat(
                    tuple(self.hcl.labelers[global_class]._sorted[close_idcs, identity, j] for j in range(n_subclasses) if identity != j))
                # print(scores)
                if torch.max(scores) > max_score:
                    max_score = torch.max(scores)

            if max_score < min_ambig_pose[0]:
                min_ambig_pose = (max_score, camera_pose)
                min_i = k
        # print("best next image is ", min_i, self.possible_orientations[min_i])
        return min_ambig_pose, min_i
