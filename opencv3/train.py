#!/usr/bin/python3

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
        for d in dirs:
            if d not in name_and_label:
                continue
            files = os.listdir(os.path.join(data_path, d))
            files = [f for f in files if f.endswith('.jpg')]
            for f in files:
                image = cv2.imread(os.path.join(
                    data_path, d, f), cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, IMAGE_SIZE)
                datas.append(image)
                labels.append(name_and_label[d])

    datas = np.array(datas)
    labels = np.array(labels)
    print('All data is loaded')
    print("category name and it's label:", name_and_label)
    return datas, labels


def load_test_data():
    wd = os.getcwd()
    data_path = os.path.join(wd, 'test')
    assert(os.path.exists(data_path))
    datas = []
    labels = []

    for _, _, files in os.walk(data_path):
        for f in files:
            if not f.endswith('.jpg'):
                continue

            image = cv2.imread(os.path.join(
                data_path, f), cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, IMAGE_SIZE)
            datas.append(image)
            label = f.split('_')[0]
            print(label)
            # print(name_and_label[label])
            labels.append(name_and_label[label])

    print('All data is loaded')
    print("category name and it's label:", name_and_label)
    return datas, labels


def get_hog():
    winSize = IMAGE_SIZE
    blockSize = (8, 8)
    blockStride = (4, 4)
    cellSize = (4, 4)
    # blockSize = (4, 4)
    # blockStride = (2, 2)
    # cellSize = (2, 2)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    # signedGradient = False
    signedGradient = True

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture,
                            winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)
    return hog


def svmInit(C=12.5, gamma=0.50625):
    model = cv2.ml.SVM_create()
    model.setGamma(gamma)
    model.setC(C)
    model.setKernel(cv2.ml.SVM_RBF)
    model.setType(cv2.ml.SVM_C_SVC)

    return model


def svmTrain(model, samples, responses):
    # model.train(samples, cv2.ml.ROW_SAMPLE, responses)
    model.trainAuto(samples, cv2.ml.ROW_SAMPLE, responses)
    return model


def svmPredict(model, samples):
    return model.predict(samples)[1].ravel()


def svmEvaluate(model, samples, labels):
    predictions = svmPredict(model, samples)
    accuracy = (labels == predictions).mean()
    print('Percentage Accuracy: %.2f %%' % (accuracy*100))

    confusion = np.zeros((len(name_and_label), len(name_and_label)), np.int32)
    for i, j in zip(labels, predictions):
        confusion[int(i), int(j)] += 1
    print('confusion matrix:')
    print(confusion)


def main():

    print('Loading data ...')
    datas, labels = load_data()
    # print(labels)

    print('Shuffle data ...')
    rand = np.random.RandomState(42)
    shuffle = rand.permutation(len(datas))
    datas, labels = datas[shuffle], labels[shuffle]
    # datas = datas[shuffle]
    # labels = labels[shuffle]

    print('Extract features using HOG descriptor ...')
    hog = get_hog()
    hog_descriptors = []
    for img in datas:
        hog_descriptors.append(hog.compute(img))
    hog_descriptors = np.squeeze(hog_descriptors)
    print('Length of hog descriptors: %d' % (len(hog_descriptors)))

    print('Spliting data into training (90%) and test set (10%)... ')
    train_n = int(0.9*len(hog_descriptors))

    hog_descriptors_train, hog_descriptors_test = np.split(
        hog_descriptors, [train_n])
    labels_train, labels_test = np.split(labels, [train_n])

    print('Training SVM model ...')
    model = svmInit()
    svmTrain(model, hog_descriptors_train, labels_train)

    print('Evaluating model ... ')
    svmEvaluate(model, hog_descriptors_test, labels_test)

    print('Save the trained SVM model ...')
    model.save('svm.xml')
    hog.save('hog_descriptor.xml')


def test():
    svm = cv2.ml.SVM_load('svm.xml')
    hog = get_hog()
    datas, labels = load_test_data()
    hog_descriptors = []
    for img in datas:
        hog_descriptors.append(hog.compute(img))
    hog_descriptors = np.squeeze(hog_descriptors)
    print('Length of hog descriptors: %d' % (len(hog_descriptors)))

    print('Evaluating model ... ')
    svmEvaluate(svm, hog_descriptors, labels)


if __name__ == '__main__':
    # main()
    test()
