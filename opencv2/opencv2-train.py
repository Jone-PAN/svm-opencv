#!/usr/bin/python
# Test dataset's pictures' name formate is like left_1234.jpg.
# Getting test dataset's pictures'label is to evaluate model's test accuracy.
# SVM's input is hog_descriptors,and hog_descriptors is compute from pictures.
# c and gamma value can be see in svm.xml.
import os
import sys
import cv2
import numpy as np

WIDTH = 16
HEIGHT = 32
IMAGE_SIZE = (WIDTH, HEIGHT)

name_and_label = {'lightOn': 0, 'lightOff': 1}


def load_data():
    wd = os.getcwd()
    data_path = os.path.join(wd, 'data')
    assert(os.path.exists(data_path))
    datas = []
    labels = []

    for _, dirs, _ in os.walk(data_path):
        for d in dirs:      # d represents class folder
            if d not in name_and_label:
                continue
            files = os.listdir(os.path.join(data_path, d))      # list of all files in d folder
            files = [f for f in files if f.endswith('.jpg')]    # list of all files endswith('.jpg') in d folder
            for f in files:
                image = cv2.imread(os.path.join(
                    data_path, d, f), cv2.IMREAD_GRAYSCALE)     # imread one train picture
                image = cv2.resize(image, IMAGE_SIZE)           # resize img to IMAGE_SIZE
                datas.append(image)
                labels.append(name_and_label[d])                # append corresponding img's label for the img,labels is like [2,1,3,...]

    datas = np.array(datas)
    labels = np.array(labels)
    print('All data is loaded')
    print("category name and it's label:", name_and_label)  # {'circle': 0, 'forward': 1, 'left': 2, 'right': 3}
    return datas, labels    # return formate is np.array


def load_test_data():
    wd = os.getcwd()
    data_path = os.path.join(wd, 'test')    # get the test data path
    assert(os.path.exists(data_path))
    datas = []
    labels = []

    for _, _, files in os.walk(data_path):
        for f in files:
            if not f.endswith('.jpg'):
                continue

            image = cv2.imread(os.path.join(
                data_path, f), cv2.IMREAD_GRAYSCALE)        # imread one test picture
            image = cv2.resize(image, IMAGE_SIZE)
            datas.append(image)
            label = f.split('_')[0]     # test-picture's formate is like "left_1234.jpg". Get label from pictures' name.
            labels.append(name_and_label[label])    # append corresponding img's label for the img,labels is like [2,1,3,...]

    print('All data is loaded')
    print("category name and it's label:", name_and_label)  # {'circle': 0, 'forward': 1, 'left': 2, 'right': 3}
    return datas, labels         # return formate is  list


class StatModel(object):
    def load(self, fn):
        self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)

 
class SVM(StatModel):  
    def __init__(self, C=1, gamma=0.5):
        self.params = dict(kernel_type=cv2.SVM_RBF,
                           svm_type=cv2.SVM_C_SVC,
                           C=C,
                           gamma=gamma)
        self.model = cv2.SVM()

    def train(self, samples, responses):
        self.model = cv2.SVM()
        self.model.train(samples, responses, params=self.params)

    def predict(self, samples):
        return self.model.predict_all(samples).ravel()

    def get_var_count(self):
        return self.model.get_var_count()


def evaluate_model(model, samples, labels):
    resp = model.predict(samples)
    accuracy = (labels == resp).mean()
    print('accuracy: %.2f %%' % (accuracy*100))

    confusion = np.zeros((4, 4), np.int32)
    for i, j in zip(labels, resp):
        i = int(i)
        j = int(j)
        confusion[i, j] += 1
    print('confusion matrix:')
    print(confusion)


def get_hog():
    hog = cv2.HOGDescriptor()
    hog.load('hog_descriptor.xml')
    return hog


def train():

    print('Loading data ...')
    datas, labels = load_data()

    print('Shuffle data ...')
    rand = np.random.RandomState(42)
    shuffle = rand.permutation(len(datas))
    datas, labels = datas[shuffle], labels[shuffle]

    print('Extract features using HOG descriptor ...')
    hog = get_hog()
    print(hog.winSize)

    hog_descriptors = []
    for img in datas:
        hog_descriptors.append(hog.compute(img, (4, 4)))
    hog_descriptors = np.squeeze(hog_descriptors)
    print('Length of hog descriptors: %d' % (len(hog_descriptors)))

    print('Spliting data into training (90%) and test set (10%)... ')
    train_n = int(0.9*len(hog_descriptors))
    hog_descriptors_train, hog_descriptors_test = np.split(
        hog_descriptors, [train_n])
    labels_train, labels_test = np.split(labels, [train_n])

    print('Training SVM model ...')
    model = SVM(C=1.25, gamma=0.03375)
    model.train(hog_descriptors_train, labels_train)

    print('Evaluating model ... ')
    evaluate_model(model, hog_descriptors_test, labels_test)

    print('Save the trained SVM model ...')
    model.save('svm-opencv2.xml')


def test():
    svm = SVM()
    svm.load('svm-opencv2.xml')
    hog = get_hog()
    datas, labels = load_test_data()
    hog_descriptors = []
    for img in datas:
        hog_descriptors.append(hog.compute(img, (4, 4)))
    hog_descriptors = np.squeeze(hog_descriptors)
    print('Length of hog descriptors: %d' % (len(hog_descriptors)))

    print('Evaluating model ... ')
    evaluate_model(svm, hog_descriptors, labels)


if __name__ == '__main__':
    #train()
    test()
