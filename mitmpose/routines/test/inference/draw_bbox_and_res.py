from mitmpose.routines.test.inference.next_pose import *
from mitmpose.model.classification.dataset_hierarchical import *
import shutil
from torchvision import transforms as T
import pickle
from tqdm import trange
import cv2

if __name__ == "__main__":
    bag_prefix = '/home/safoex/Desktop/esafronov/bags_data/%d/'
    img_suffix = '/img%05d.jpg'
    bag_img_pattern = bag_prefix + img_suffix
    bag_save_folder = bag_prefix + '/processed/'
    bag_img_save_pattern = bag_save_folder + img_suffix
    bag_pickle_format = bag_prefix + '/results.pickle'

    global_classes = ['babyfood', 'babymilk', 'babyfood']
    remap_subclasses = {
        'humana2': 'humana1',
        'humana1': 'humana2',
        'meltacchin': 'turkey',
        'melpollo': 'chicken'
    }
    for bag_n in range(3):
        with open(bag_pickle_format % bag_n, 'rb') as pf:
            results = pickle.load(pf)

        if not os.path.exists(bag_save_folder % bag_n):
            os.mkdir(bag_save_folder % bag_n)

        filenames = os.listdir(bag_prefix % bag_n)
        images_n = sum(fn[-3:] == 'jpg' for fn in filenames)
        print(images_n)
        # images_n = 500

        bot = None
        bl = None
        changed = 0
        for i in trange(images_n):
            bbox, res = results[i]
            img_path = bag_img_pattern % (bag_n, i)
            img_save_path = bag_img_save_pattern % (bag_n, i)
            if bbox is None:
                shutil.copy(img_path, img_save_path)
                continue
            img = cv2.imread(img_path)
            # [bbox[0][0], bbox[1][0], bbox[0][1], bbox[1][1]]
            ambiguity = (res[0][2] + res[0][3]) / 2
            objclass = res[0][4]
            if ambiguity < 0.4:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.rectangle(img, bbox[0], bbox[1], color, 2)
            bottom_left = (bbox[0][0], int(bbox[1][1] + 25))
            orig_bl = bottom_left
            fontColor = (255, 255, 255)
            text11 = "class:"
            text12 = remap_subclasses[objclass]
            text21 = "ambiguity:"
            text22 = "%.2f" % ambiguity
            bot_y = bottom_left[1] + 60
            right_x = bottom_left[0] + 340

            if bot_y > img.shape[0]:
                # if bot is None:
                #     bot = False
                # else:
                #     if bot is False:
                #         changed = i
                # bot = True
                bottom_left = (bottom_left[0], int(bbox[0][1] - 50))

            if right_x > img.shape[1]:
                bottom_left = (int(bottom_left[0] - 70), bottom_left[1])

            # else:
            #     if bot is None:
            #         bot = True
            #     else:
            #         if bot is True:
            #             changed = i
            #     bot = False

            # alpha = 0.2
            # print(i, changed)
            # if i - changed > 5:
            #     bottom_left = (int(bl[0] * (1-alpha) + alpha * bottom_left[0]),
            #                    int(bl[1] * (1-alpha) + alpha * bottom_left[1]))
            # else:
            #     bl = bottom_left

            # no filtering..

            # bottom_left = orig_bl
            cv2.putText(img, text11, bottom_left, cv2.FONT_HERSHEY_SIMPLEX, 1, fontColor, 2)
            cv2.putText(img, text21, (bottom_left[0], bottom_left[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, fontColor, 2)
            cv2.putText(img, text12, (int(bottom_left[0] + 180), bottom_left[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, fontColor, 2)
            cv2.putText(img, text22, (int(bottom_left[0] + 180), bottom_left[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imwrite(img_save_path, img)
