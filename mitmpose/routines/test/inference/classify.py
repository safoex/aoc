import torchvision
from PIL import Image
from torchvision import transforms as T
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from mitmpose.model.pose.aae.aae import AAETransform, AAE
from mitmpose.model.classification.classifier_hierarchical import HierarchicalClassifier, \
    HierarchicalManyObjectsDataset, Grid
from scipy.spatial.transform import Rotation
from mitmpose.model.pose.codebooks.codebook import Codebook, OnlineRenderDataset
import os, sys
import pickle

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
import torchvision


class InferenceClassifier:
    def __init__(self, hcl: HierarchicalClassifier, device, cache=None):
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.detector.to(device)
        self.detector.eval()
        self.hcl = hcl
        self.device = device
        self.cdbks = {}

        self.cache = cache or {}

    def get_img(self, img_or_path):
        if isinstance(img_or_path, str):
            img = Image.open(img_or_path)  # Load the image
        else:
            img = img_or_path
        return img

    def get_prediction(self, img_or_path, threshold, filter_classes=['book', 'bottle']):
        img = self.get_img(img_or_path)
        transform = T.Compose([T.ToTensor()])  # Define PyTorch Transform
        img = transform(img)  # Apply the transform to the image
        with torch.no_grad():
            pred = self.detector([img.to(self.device)])  # Pass the image to the model
        #   print(pred)
        filter = [COCO_INSTANCE_CATEGORY_NAMES[cl] in filter_classes for cl in pred[0]['labels']]
        pred = [{k: t[filter] for k, t in pred[0].items()}]
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in
                      list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in
                      list(pred[0]['boxes'].detach().cpu().numpy())]  # Bounding boxes
        pred_score = list(pred[0]['scores'].detach().cpu().numpy())
        if len(pred_score) == 0:
            return None, None
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][
            -1]  # Get list of index with score greater than threshold.
        pred_boxes = pred_boxes[:pred_t + 1]
        pred_class = pred_class[:pred_t + 1]
        return pred_boxes, pred_class

    def closest_to_the_center(self, img_path, filter_classes=['book', 'bottle']):
        boxes, pred_cls = self.get_prediction(img_path, 0, filter_classes)  # Get predictions
        img = cv2.imread(img_path)  # Read image with cv2
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        center = np.array([img.shape[0] / 2, img.shape[1] / 2])
        print(boxes)
        bbox_centers = np.array(
            [np.array([(bbox[1][0] + bbox[0][0]) / 2, (bbox[1][1] + bbox[0][1]) / 2]) for bbox in boxes])
        print(bbox_centers)
        print(np.linalg.norm(bbox_centers - center))
        closest = np.argmin(np.linalg.norm(bbox_centers - center, axis=1))
        return boxes[closest]

    def closest_by_size(self, img_path, filter_classes=['book', 'bottle']):
        boxes, pred_cls = self.get_prediction(img_path, 0, filter_classes)  # Get predictions
        if boxes is None:
            return None
        img = cv2.imread(img_path)  # Read image with cv2
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        sz = (img.shape[0] + img.shape[1]) / 2
        bbox_sizes = np.max([np.array([bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1]]) for bbox in boxes])
        closest = np.argmin(np.abs(bbox_sizes - sz))
        return boxes[closest]

    def crop_and_resize(self, img, bbox, target_res=128):
        # target_res = target_res or self.target_res
        row_center = int(np.mean(bbox[2:]))
        col_center = int(np.mean(bbox[:2]))
        widest = max(bbox[1] - bbox[0], bbox[3] - bbox[2])
        half_side = int((widest * 1.2) / 2)
        left = row_center - half_side
        right = row_center + half_side
        top = col_center - half_side
        bottom = col_center + half_side

        final_box = (top, left, bottom, right)

        return np.array(Image.fromarray(img).crop(final_box).resize((target_res, target_res)), dtype=np.float32)

    def detect_and_crop(self, img_path, filter_classes=['book', 'bottle'], target_res=128, cache=None):
        cache_key = img_path + ' ' + str(filter_classes) + ' ' + str(target_res)

        # print(len(cache), cache_key)
        if cache_key in cache:
            return cache[cache_key]
        else:
            # print('uncached')
            img = cv2.imread(img_path)  # Read image with cv2
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            # bbox = closest_to_the_center(img_path, filter_classes)
            bbox = self.closest_by_size(img_path, filter_classes)
            if bbox is None:
                cache[cache_key] = None
            # print(bbox)
            else:
                cache[cache_key] = self.crop_and_resize(img, [bbox[0][0], bbox[1][0], bbox[0][1], bbox[1][1]], target_res=target_res)

        return cache[cache_key]

    def get_xyz(self, grid):
        a = Rotation.from_matrix(grid).as_euler('xyz')
        if len(a.shape) < 2:
            a = a.reshape((1, 3))
        x = np.zeros_like(a)
        r = 1
        x[:, 0] = r * np.sin(a[:, 0]) * np.cos(a[:, 1])
        x[:, 1] = r * np.sin(a[:, 0]) * np.sin(a[:, 1])
        x[:, 2] = r * np.cos(a[:, 0])
        return x

    def get_closest_without_inplane(self, rot, grid, top_k=1):
        gxyz = torch.from_numpy(self.get_xyz(grid))

        rxyz = torch.from_numpy(self.get_xyz(rot)).view(1, 3)
        # print(gxyz[::10000,:])
        # print(torch.norm(gxyz-rxyz,dim=1)[::10000])
        return torch.topk(-torch.norm(gxyz - rxyz, dim=1), top_k)[1]

    def global_classify_with_aaes(self, crop):
        t_input = crop
        cur_max = 0
        gcl = 0
        with torch.no_grad():
            for igcl in self.hcl.classes:
                aae = self.hcl.aaes[igcl]
                for i, lcl in enumerate(self.hcl.classes[igcl]):
                    tops, idcs = torch.topk(self.cdbks[lcl].cos_sim(aae.encoder.forward(t_input)), 2)
                    res = tops[0].item()
                    if res > cur_max:
                        gcl = igcl
                        cur_max = res
        return gcl

    def log_steps(self, img_path, copy_path, crop_path, rec_path, cdbk_path_pattern, assume_global_class=None):
        crop = self.detect_and_crop(img_path, cache=dict())
        if crop is None:
            return
        Image.open(img_path).save(copy_path)

        crop = T.ToTensor()(crop / 255.).to(self.device).view(1, 3, 128, 128)
        (T.ToPILImage()(crop[0,:,:,:].cpu())).save(crop_path)
        # with torch.no_grad():
        #     global_class = torch.argmax(self.hcl.global_classifier(crop)).item()
            # print(global_class)
        t_input = crop

        if assume_global_class:
            gcl = assume_global_class
        else:
            gcl = self.global_classify_with_aaes(crop)

        # gcl = self.hcl.global_classes[global_class]
        aae = self.hcl.aaes[gcl]
        with torch.no_grad():
            t_rec = aae(t_input.view(1, 3, 128, 128))[0].cpu()
            T.ToPILImage()(t_rec).save(rec_path)
            for i, lcl in enumerate(self.hcl.classes[gcl]):
                i_best = self.cdbks[lcl].best(aae.encoder.forward(t_input))
                match_img = self.cdbks[lcl]._ds[i_best][0]
                T.ToPILImage()(torch.from_numpy(match_img)).save(cdbk_path_pattern % i)

    def classify(self, img_path, robot_orientation=None, threshold=0.4, assume_global_class=None, cache=None):
        crop = self.detect_and_crop(img_path, cache=cache)
        if crop is None:
            return

        original_crop = self.detect_and_crop(img_path, target_res=224, cache=cache)
        crop = T.ToTensor()(crop / 255.).to(self.device).view(1, 3, 128, 128)
        with torch.no_grad():
            pil_crop = Image.fromarray(original_crop.astype(np.uint8))
            normalized_crop = HierarchicalManyObjectsDataset.transform_inference(pil_crop).view(1, 3, 224, 224).to(self.device)
            global_class = torch.argmax(self.hcl.global_classifier(normalized_crop)).item()
            # print(global_class)
        gcl_cl = self.hcl.global_classes[global_class]
        gcl = self.global_classify_with_aaes(crop)
        aae = self.hcl.aaes[gcl]
        t_input = crop
        result = [gcl, None]
        result = [gcl, gcl_cl, None, None, None, None]
        # [
        # global class according to AAE,
        # global class according to global classifier,
        # score assuming local class 0,
        # score assuming local class 1,
        # local class according to local classifier,
        # local class according to AAE
        # ]
        if assume_global_class is not None:
            gcl = assume_global_class
        with torch.no_grad():
            bad_pose = False

            max_score = 0
            max_cl = None
            for i, lcl in enumerate(self.hcl.classes[gcl]):
                top = torch.topk(self.cdbks[lcl].cos_sim(aae.encoder.forward(t_input)), 1)[0]
                if top > max_score:
                    max_score = top
                    max_cl = lcl

            n_subclasses = len(self.hcl.classes[gcl])
            for i, lcl in enumerate(self.hcl.classes[gcl]):
                tops, idcs = torch.topk(self.cdbks[lcl].cos_sim(aae.encoder.forward(t_input)), 2)
                max_score = 0
                for idx in idcs:
                    rot = self.cdbks[lcl].grider.grid[idx]
                    close_idcs = self.get_closest_without_inplane(rot, self.hcl.labelers[gcl].grider.grid, 1)
                    # TODO: remove this hack only for two
                    scores = torch.cat(tuple(self.hcl.labelers[gcl]._sorted[close_idcs, i, j] for j in range(n_subclasses) if i != j))
                    # print(scores)
                    if  torch.max(scores) > max_score:
                        max_score = torch.max(scores)

                    if torch.any(scores > threshold):
                        bad_pose = True

                result[2 + i] = max_score.cpu().item()

            # if not bad_pose:
            pil_crop = Image.fromarray(original_crop.astype(np.uint8))
            normalized_crop = HierarchicalManyObjectsDataset.transform_inference(pil_crop).view(1, 3, 224, 224).to(self.device)

            lcl = torch.argmax(self.hcl.in_class_classifiers[gcl](normalized_crop)).item()
            # result = [gcl, gcl_cl, self.hcl.classes[gcl][lcl], max_cl]
            result[4] = self.hcl.classes[gcl][lcl]
            result[5] = max_cl
            return result
            # else:
            #     return [gcl, gcl_cl, None]

    def classify2(self, img_path, robot_orientation=None, threshold=0.4, assume_global_class=None, cache=None):
        crop = self.detect_and_crop(img_path, cache=cache)
        if crop is None:
            return

        original_crop = self.detect_and_crop(img_path, target_res=224, cache=cache)
        crop = T.ToTensor()(crop / 255.).to(self.device).view(1, 3, 128, 128)
        with torch.no_grad():
            pil_crop = Image.fromarray(original_crop.astype(np.uint8))
            normalized_crop = HierarchicalManyObjectsDataset.transform_inference(pil_crop).view(1, 3, 224, 224).to(self.device)
            global_class = torch.argmax(self.hcl.global_classifier(normalized_crop)).item()
            # print(global_class)
        gcl_cl = self.hcl.global_classes[global_class]
        gcl = self.global_classify_with_aaes(crop)
        aae = self.hcl.aaes[gcl]
        t_input = crop
        result = [gcl, None]
        result = [gcl, gcl_cl, None, None, None, None]
        # [
        # global class according to AAE,
        # global class according to global classifier,
        # score assuming local class 0,
        # score assuming local class 1,
        # local class according to local classifier,
        # local class according to AAE
        # ]
        if assume_global_class is not None:
            gcl = assume_global_class

        with torch.no_grad():
            bad_pose = False

            max_score = 0
            max_cl = None
            for i, lcl in enumerate(self.hcl.classes[gcl]):
                top = torch.topk(self.cdbks[lcl].cos_sim(aae.encoder.forward(t_input)), 1)[0]
                if top > max_score:
                    max_score = top
                    max_cl = lcl

            n_subclasses = len(self.hcl.classes[gcl])
            orientation_hypothesis = list()
            for i, lcl in enumerate(self.hcl.classes[gcl]):
                tops, idcs = torch.topk(self.cdbks[lcl].cos_sim(aae.encoder.forward(t_input)), 1)
                max_score = 0
                for idx in idcs:
                    rot = self.cdbks[lcl].grider.grid[idx]
                    close_idcs = self.get_closest_without_inplane(rot, self.hcl.labelers[gcl].grider.grid, 1)
                    # TODO: remove this hack only for two
                    scores = torch.cat(tuple(self.hcl.labelers[gcl]._sorted[close_idcs, i, j] for j in range(n_subclasses) if i != j))
                    # print(scores)
                    if  torch.max(scores) > max_score:
                        max_score = torch.max(scores)
                    orientation_hypothesis.append((i, rot))
                    if torch.any(scores > threshold):
                        bad_pose = True

                result[2 + i] = max_score.cpu().item()

            # if not bad_pose:
            pil_crop = Image.fromarray(original_crop.astype(np.uint8))
            normalized_crop = HierarchicalManyObjectsDataset.transform_inference(pil_crop).view(1, 3, 224, 224).to(self.device)

            lcl = torch.argmax(self.hcl.in_class_classifiers[gcl](normalized_crop)).item()
            # result = [gcl, gcl_cl, self.hcl.classes[gcl][lcl], max_cl]
            result[4] = self.hcl.classes[gcl][lcl]
            result[5] = max_cl
            return result, orientation_hypothesis
            # else:
            #     return [gcl, gcl_cl, None]


if __name__ == '__main__':
    # workdir = '/home/safoex/Documents/data/aae/release2/release'
    workdir = '/home/safoex/Documents/data/aae/release2/release2'
    models_dir = '/home/safoex/Documents/data/aae/models/scans/'
    models_names = ['meltacchin', 'melpollo', 'humana1', 'humana2']
    models = {mname: {'model_path': models_dir + '/' + mname + '.obj', 'camera_dist': None} for mname in models_names}
    # grider = Grid(100, 5)
    grider = Grid(300, 20)
    ds = HierarchicalManyObjectsDataset(grider, models, res=236, classification_transform=HierarchicalManyObjectsDataset.transform_normalize,
                                        aae_render_transform=AAETransform(0.5,
                                                                          '/home/safoex/Documents/data/VOCtrainval_11-May-2012',
                                                                          add_patches=False, size=(236, 236)))
    # ds = HierarchicalManyObjectsDataset(grider, models, aae_render_transform=AAETransform(0.5,
    #                                                                       '/home/safoex/Documents/data/VOCtrainval_11-May-2012',
    #                                                                       add_patches=True))

    # ds.set_mode('aae')
    # ds.create_dataset(workdir)

    classes = {'babyfood': ['meltacchin', 'melpollo'],
               'babymilk': ['humana1', 'humana2']}
    def global_class_of(local_class, classes):
        for gcl, gcl_list in classes.items():
            if local_class in gcl_list:
                return gcl
        return None
    ds.make_hierarchical_dataset(
        classes
    )

    device = torch.device('cuda:0')
    ds.load_dataset(workdir)
    aae_params = (128, 256, (128, 256, 256, 512))
    hcl = HierarchicalClassifier(workdir, ds, ambiguous_aae_params=aae_params, global_aae_params=aae_params,
                                 device=device)

    hcl.manual_set_classes(classes)

    hcl.load_everything()

    inference = InferenceClassifier(hcl, device)


    for gcl, subclasses in classes.items():
        for lcl in subclasses:
            inference.cdbks[lcl] = Codebook(hcl.aaes[gcl], OnlineRenderDataset(Grid(4000, 40), models[lcl]['model_path']))
            # inference.cdbks[lcl].save(workdir + '/' + gcl + '/' + 'codebook_%s.pt' % lcl)
            inference.cdbks[lcl].load(workdir + '/' + gcl + '/' + 'codebook_%s.pt' % lcl)

    # model_names = ['melpollo', 'meltacchin', 'humana1', 'humana2']
    # radiuses = [0.35, 0.27, 0.35, 0.32]
    #
    # path_to_panda_data = '/home/safoex/Documents/data/aae/panda_data/data/'
    # N = 30

    model_names = ['melpollo', 'meltacchin', 'humana1', 'humana2']
    radiuses = [0.30, 0.30, 0.30, 0.30]

    path_to_panda_data = '/home/safoex/Documents/data/aae/10_12_2020/'
    N = None

    results = {

    }
    for model_name, radius in list(zip(model_names, radiuses))[0:]:
        print('----------%s--------' % model_name)
        if N is None:
            n = len(os.listdir(path_to_panda_data + '%s/rad_%.2f' % (model_name, radius)))
        else:
            n = N
        test_imgs = [
            path_to_panda_data + '%s/rad_%.2f/image_%d.png' % (model_name, radius, i) for i in range(1, n)
        ]
        results[model_name] = []
        for i, img_path in enumerate(test_imgs):
            with torch.no_grad():
                result = inference.classify(img_path, threshold=0.4, assume_global_class=global_class_of(model_name, classes))
                results[model_name].append(result)
                print(result, i)


    with open(workdir + '/results_0.6a.pickle', 'wb') as f:
        pickle.dump(results, f)
        # checkdir = '/home/safoex/Documents/data/aae/panda_data/test/' + model_name + '/'
        # if not os.path.exists(checkdir):
        #     os.mkdir(checkdir)

        # for i, img_path in enumerate(test_imgs):
        #     inference.log_steps(img_path, checkdir + 'img_%d_a.png' % i, checkdir + 'img_%d_crop.png' % i,
        #                         checkdir + 'img_%d_rec.png' % i, checkdir + 'img_%d' % i + '_z_%d.png')
