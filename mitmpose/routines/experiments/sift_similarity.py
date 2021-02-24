import cv2
import pickle
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_backend
import pysift


class SiftExperiment:
    def __init__(self, imageList, imagePrefix, saveKeypointsPrefix, saveDescriptorsPrefix):
        self.imageList = imageList
        self.imagesPrefix = imagePrefix
        self.descriptorsPrefix = saveDescriptorsPrefix
        self.keypointsPrefix = saveKeypointsPrefix

        # We use grayscale images for generating keypoints
        self.imagesBW = []
        for imageName in self.imageList:
            imagePath = imagePrefix + str(imageName)
            self.imagesBW.append(self.imageResizeTrain(cv2.imread(imagePath, 0)))

        self.bf = cv2.BFMatcher()

    # Resize images to a similar dimension
    # This helps improve accuracy and decreases unnecessarily high number of keypoints

    @staticmethod
    def imageResizeTrain(image):
        maxD = 1024
        height, width = image.shape
        aspectRatio = width / height
        if aspectRatio < 1:
            newSize = (int(maxD * aspectRatio), maxD)
        else:
            newSize = (maxD, int(maxD / aspectRatio))
        image = cv2.resize(image, newSize)
        return image

    @staticmethod
    def imageResizeTest(image):
        maxD = 1024
        height, width, channel = image.shape
        aspectRatio = width / height
        if aspectRatio < 1:
            newSize = (int(maxD * aspectRatio), maxD)
        else:
            newSize = (maxD, int(maxD / aspectRatio))
        image = cv2.resize(image, newSize)
        return image

    @staticmethod
    def computeSIFT(image):
        return pysift.computeKeypointsAndDescriptors(image)

    def compute_descriptors_and_save(self, n_procs=30):

        def sift_it_up(i):
            print("Starting for image: " + self.imageList[i])
            image = self.imagesBW[i]
            keypoint, descriptor = self.computeSIFT(image)
            # keypoints.append((keypointTemp, i))
            # descriptors.append((descriptorTemp, i))
            deserializedKeypoints = []
            filepath = self.keypointsPrefix + str(self.imageList[i].split('.')[0]) + ".txt"
            print(filepath)
            for point in keypoint:
                temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
                deserializedKeypoints.append(temp)
            with open(filepath, 'wb') as fp:
                pickle.dump(deserializedKeypoints, fp)
            filepath = self.descriptorsPrefix + str(self.imageList[i].split('.')[0]) + ".txt"
            with open(filepath, 'wb') as fp:
                pickle.dump(descriptor, fp)
            print("Ending for image: " + self.imageList[i])

        N = len(self.imagesBW)
        # N = 60
        Parallel(n_jobs=n_procs)(delayed(sift_it_up)(i) for i in range(N))

    def fetchKeypointFromFile(self, i):
        filepath = self.keypointsPrefix + str(self.imageList[i].split('.')[0]) + ".txt"
        keypoint = []
        file = open(filepath, 'rb')
        deserializedKeypoints = pickle.load(file)
        file.close()
        for point in deserializedKeypoints:
            temp = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2], _response=point[3],
                                _octave=point[4], _class_id=point[5])
            keypoint.append(temp)
        return keypoint

    def fetchDescriptorFromFile(self, i):
        filepath = self.descriptorsPrefix + str(self.imageList[i].split('.')[0]) + ".txt"
        file = open(filepath, 'rb')
        descriptor = pickle.load(file)
        file.close()
        return descriptor

    def calculateMatches(self, des1, des2):
        matches = self.bf.knnMatch(des1, des2, k=2)
        topResults1 = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                topResults1.append([m])

        matches = bf.knnMatch(des2, des1, k=2)
        topResults2 = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                topResults2.append([m])

        topResults = []
        for match1 in topResults1:
            match1QueryIndex = match1[0].queryIdx
            match1TrainIndex = match1[0].trainIdx

            for match2 in topResults2:
                match2QueryIndex = match2[0].queryIdx
                match2TrainIndex = match2[0].trainIdx

                if (match1QueryIndex == match2TrainIndex) and (match1TrainIndex == match2QueryIndex):
                    topResults.append(match1)
        return topResults

    @staticmethod
    def calculateScore(matches, keypoint1, keypoint2):
        return 100 * (matches / min(keypoint1, keypoint2))

    @staticmethod
    def getPlot(image1, image2, keypoint1, keypoint2, matches):
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        matchPlot = cv2.drawMatchesKnn(image1, keypoint1, image2, keypoint2, matches, None, [255, 255, 255], flags=2)
        return matchPlot

    def getPlotFor(self, i, j, keypoint1, keypoint2, matches):
        image1 = self.imageResizeTest(cv2.imread(self.imagesPrefix + self.imageList[i]))
        image2 = self.imageResizeTest(cv2.imread(self.imagesPrefix + self.imageList[j]))
        return self.getPlot(image1, image2, keypoint1, keypoint2, matches)

    def calculateResultsFor(self, i, j, need_plot=True, need_print=False):
        keypoint1 = self.fetchKeypointFromFile(i)
        descriptor1 = self.fetchDescriptorFromFile(i)
        keypoint2 = self.fetchKeypointFromFile(j)
        descriptor2 = self.fetchDescriptorFromFile(j)
        matches = self.calculateMatches(descriptor1, descriptor2)
        score = self.calculateScore(len(matches), len(keypoint1), len(keypoint2))
        plot = self.getPlotFor(i, j, keypoint1, keypoint2, matches)
        if need_print:
            print(len(matches), len(keypoint1), len(keypoint2), len(descriptor1), len(descriptor2))
            print(score)
        if need_plot:
            plt.imshow(plot), plt.show()
        return score

    def compute_similarities(self):
        return [self.calculateResultsFor(x, x + 300, need_plot=False) for x in range(300)]


if __name__ == "__main__":
    imageList = ["top%d.png" % i for i in range(300)] + ["top%d_f.png" % i for i in range(300)]
    imagePrefix = "data/images/tops_0_1/"
    saveKeypointsPrefix = "data/keypoints/"
    saveDescriptorsPrefix = "data/descriptors/"

    sexp = SiftExperiment(imageList, imagePrefix, saveKeypointsPrefix, saveDescriptorsPrefix)
    sexp.compute_descriptors_and_save(n_procs=30)
    sift_sim = sexp.compute_similarities()


