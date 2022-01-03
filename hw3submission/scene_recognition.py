import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from scipy import stats
from pathlib import Path, PureWindowsPath
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


def extract_dataset_info(data_path):
    # extract information from train.txt
    f = open(os.path.join(data_path, "train.txt"), "r")
    contents_train = f.readlines()
    label_classes, label_train_list, img_train_list = [], [], []
    for sample in contents_train:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        if label not in label_classes:
            label_classes.append(label)
        label_train_list.append(sample[0])
        img_train_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))
    #print('Classes: {}'.format(label_classes))

    # extract information from test.txt
    f = open(os.path.join(data_path, "test.txt"), "r")
    contents_test = f.readlines()
    label_test_list, img_test_list = [], []
    for sample in contents_test:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        label_test_list.append(label)
        img_test_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))  # you can directly use img_path if you run in Windows

    return label_classes, label_train_list, img_train_list, label_test_list, img_test_list


def get_tiny_image(img, output_size):
    # To do
    outputW = output_size[0]
    outputH = output_size[1]
    imgMean = np.mean(img)
    imgStd = np.std(img)
    normalizedImg = (img - imgMean) / imgStd
    tinyImg = cv2.resize(normalizedImg, (outputW, outputH))    
    feature = tinyImg.flatten()
    return feature


def predict_knn(feature_train, label_train, feature_test, k):
    # To do
    '''
    feature_train = [# of training data, dim of image feature (256 for 16x16 tiny img)]
    label_train = [1,15] specifies labels of training data
    feature_test = [# testing data samples, dim of img features]
    k = num of neighbors for label prediction
    '''
    
    neighbor = KNeighborsClassifier(n_neighbors = k)
    neighbor.fit(feature_train, label_train)
    label_test_pred = neighbor.predict(feature_test)

    return label_test_pred


def classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    
    label_train = []
    label_test = []
    feature_train = []
    feature_test = []
    
    output_size = (16,16)
    k = 15
    
    for a in label_train_list:
        label_train.append(label_classes.index(a))
    
    for b in label_test_list:
        label_test.append(label_classes.index(b))
    
    for c in img_train_list:
        img = cv2.imread(c,0)
        tinyImg = get_tiny_image(img, output_size)
        feature_train.append(tinyImg)
    
    for d in img_test_list:
        img = cv2.imread(d,0)
        tinyImg = get_tiny_image(img, output_size)
        feature_test.append(tinyImg)

    
    labelPredicted = predict_knn(feature_train, label_train, feature_test, k)
    confusion = confusion_matrix(label_test, labelPredicted)
    traceSum = np.trace(confusion)
    totalHorizontal = np.sum(confusion, axis=1)
    total = np.sum(totalHorizontal, axis = 0)
    accuracy = traceSum / total
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    
    return confusion, accuracy


def compute_dsift(img, stride, size):
    # To do
    sift = cv2.xfeatures2d.SIFT_create()

    keypoints = [cv2.KeyPoint(x, y, size) for y in range(0, img.shape[0], stride) 
                                         for x in range(0, img.shape[1], stride)]

    kps, dense_feature = sift.compute(img , keypoints)
    return dense_feature


def build_visual_dictionary(dense_feature_list, dic_size):
    # To do
    
    dense_features = np.vstack(dense_feature_list)   
    kmeans = KMeans(n_clusters=dic_size, n_init=15, max_iter=200, random_state=0).fit(dense_features)
    vocab = kmeans.cluster_centers_
    return vocab


def compute_bow(feature, vocab):
    # To do
    #size of bow_feature = (1 x dict_size)
    #shape of vocab = (90 x 128)
    
    neighbor = NearestNeighbors()
    neighbor.fit(vocab)
    ind = neighbor.kneighbors(feature, n_neighbors = 1, return_distance=False)

    bow = [0] * np.shape(vocab)[0]
    bow = np.asarray(bow)
    for i in ind:
        bow[i] += 1
    
    bow_feature = bow / np.linalg.norm(bow)

    return bow_feature


def classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    stride = 10
    size = 10
    dic_size = 90
    k = 10
    
    label_train = []
    label_test = []
    
    imgTrainDenseFeat = []
    
    for c in img_train_list:
        img = cv2.imread(c,0)
        SIFTfeat = compute_dsift(img, stride, size)
        imgTrainDenseFeat.append(SIFTfeat)

    vocab= build_visual_dictionary(imgTrainDenseFeat, dic_size)
    np.savetxt('Vocab', vocab, delimiter=', ')
    
    '''
    vocab = np.loadtxt('experimVocab6',  delimiter = ', ')    #accuracy = 0.468
    #vocab = np.loadtxt("experimentVocab7", delimiter = ", ")   #acc = 0.473
    #vocab = np.loadtxt("experimentVocab8", delimiter = ", ")   #acc = 0.498
    '''
    
    #vocab = np.loadtxt("experimentVocab10", delimiter = ", ")
    
    for a in label_train_list:
        label_train.append(label_classes.index(a))
    
    for b in label_test_list:
        label_test.append(label_classes.index(b))
    
    imgTrainBow = []
    for c in imgTrainDenseFeat:
        imgTrainBowFeat = compute_bow(c, vocab)
        imgTrainBow.append(imgTrainBowFeat) 
    
    imgTestBow = []
    for d in img_test_list:
        img = cv2.imread(d,0)
        SIFTFeat = compute_dsift(img,stride,size)
        imgTestBowFeat = compute_bow(SIFTFeat,vocab)
        imgTestBow.append(imgTestBowFeat) 
    
    predictedLabels = predict_knn(imgTrainBow, label_train, imgTestBow, k)
    confusion = confusion_matrix(label_test, predictedLabels) 
    traceSum = np.trace(confusion)
    totalHorizontal = np.sum(confusion, axis=1)
    total = np.sum(totalHorizontal,axis = 0)
    accuracy = traceSum / total
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def predict_svm(feature_train, label_train, feature_test, n_classes):
    # To do
    
    '''    
    C = 0.5 w experimVocab9 -> 0.565
    C = 5 w experimVocab9 -> 0.558
    
    C = 5 w experimVocab10 -> 0.586
    C = 0.5 w experimVocab10 -> 0.602
    '''
    lsvc = LinearSVC(tol = 1e-6, C=0.5)
    lsvc.fit(feature_train, label_train)
    label_test_pred = lsvc.predict(feature_test)
    
    return label_test_pred


def classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    stride = 20
    size = 20
    dic_size = 90
    
    label_train = []
    label_test = []
    imgTrainDenseFeat = []
    
    for c in img_train_list:
        img = cv2.imread(c,0)
        SIFTfeat = compute_dsift(img, stride, size)
        imgTrainDenseFeat.append(SIFTfeat)
    
    vocab= build_visual_dictionary(imgTrainDenseFeat, dic_size)
    np.savetxt('Vocab', vocab, delimiter = ', ')
    
    #vocab = np.loadtxt("experimentVocab10", delimiter = ", ")
    
    for a in label_train_list:
        label_train.append(label_classes.index(a))
    
    for b in label_test_list:
        label_test.append(label_classes.index(b))
        
    imgTrainBow = []
    for c in imgTrainDenseFeat:
        imgTrainBowFeat = compute_bow(c, vocab)
        imgTrainBow.append(imgTrainBowFeat)
    
    imgTestBow = []
    for d in img_test_list:
        img = cv2.imread(d,0)
        SIFTFeat = compute_dsift(img,stride,size)
        imgTestBowFeat = compute_bow(SIFTFeat,vocab)
        imgTestBow.append(imgTestBowFeat) 

    predictedLabels = predict_svm(imgTrainBow, label_train, imgTestBow, len(label_classes))
    confusion = confusion_matrix(label_test, predictedLabels) 
    traceSum = np.trace(confusion)
    totalHorizontal = np.sum(confusion, axis=1)
    total = np.sum(totalHorizontal, axis = 0)
    accuracy = traceSum / total
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # avoid top and bottom part of heatmap been cut
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # To do: replace with your dataset path
    label_classes, label_train_list, img_train_list, label_test_list, img_test_list = extract_dataset_info("./scene_classification_data")
    
    #classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    #classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
    
    #classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)




