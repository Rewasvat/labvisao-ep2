#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import random
import cv2
from time import time
import numpy as np

###
# 1) Detection and description of image patches
# 2) Assigning patch descriptors to a set of predetermined clusters (a vocabulary) with a vector quantization algorithm
# 3) Constructing a Bag Of Keypoints, which counts the number of patches assigned to each cluster
# 4) Applying a multi-class classifier, treating the bag of keypoints as the feature vector, and thus determine which
#    category or categories to assign to the image.
###

###
#  1? Detection and description of image patches for a set of labeled training images
#  2? Constructing a set of vocabularies: each is a set of cluster centres, with respect to which descriptors are vector quantized.
#  3? Extracting bags of keypoints for these vocabularies
# 4A? Training multi-class classifiers using the bags of keypoints as feature vectors
# 4B? Selecting the vocabulary and classifier giving the best overall classification accuracy
###

# Feature Extraction: SIFT
# Visual Vocabulary Construction: k-means
# Categorization: naiveBayes ou SVM ou KNN

def t():
    import os
    l = os.listdir("../dataset")
    l.sort()
    names = []
    for s in l:
        n = "_".join( s.split(".")[0].split("_")[:-1] )
        if not n in names:
            names.append(n)
    return l, names

def tt():
    l,names = t()
    print len(names)
    for s in names: print s
    return l,names
    
###################################################
def extractFeatures(img):
    sift = cv2.SIFT()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    

###################################################
    
###################################################

###################################################
class DataSet:
    def __init__(self, path, trainingPercentage, featureExtractor):
        self.path = path
        self.featureExtractor = featureExtractor
        self.tp = trainingPercentage
        self.classes = {}
        self.classNames = []
        
    def load(self):
        files = os.listdir(self.path)
        for fileName in files:
            className = "_".join( fileName.split(".")[0].split("_")[:-1] )
            classImgIndex = fileName.split(".")[0].split("_")[-1]
            if not className in self.classes:
                self.classNames.append(className)
                self.classes[className] = DataClass(className, len(self.classNames)-1)
            self.classes[className].indexes.append( int(classImgIndex) )
        for c in self.classes.values():
            c.indexes.sort()
            
    def getTrainingData(self):
        values = None
        labels = []
        for c in self.classes.values():
            mark = time()
            # separate training set from class image set
            c.generateTrainingSet(self.tp)
            # generate vocabulary
            c.calculateClusters(self)
            
            if values == None:
                values = c.clusters
            else:
                values = np.append(values, c.clusters, 0)
            labels += [c.label]*c.clusters.shape[0]
            stepTime = time()-mark
            print "Calculated clusters for %s (%.2f secs)" % (c.name, stepTime)
        return values, np.asarray( labels, dtype='float32')
            

        
class DataClass:
    def __init__(self, name, label):
        self.name = name
        self.label = label
        self.indexes = []
        self.trainIndexes = []
        self.clusters = None
        
    def size(self):
        return len(self.indexes)
        
    def generateTrainingSet(self, percentage):
        self.trainIndexes = self.indexes[:int(len(self.indexes)*percentage)]
        del self.indexes[:int(len(self.indexes)*percentage)]
        return self.trainIndexes
        
    def popRandomSample(self):
        sample = random.choice(self.indexes)
        self.indexes.remove(sample)
        return sample
        
    def calculateClusters(self, dataset):
        classDescs = None
        # Get the descriptors for each image in training set
        for index in self.trainIndexes:
            descs = self.getImageDescriptor(index, dataset)
            if classDescs == None:
                classDescs = descs
            else:
                classDescs = np.append(classDescs, descs, 0)
        # Clusterize (generate vocabulary)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        compactness, bestLabels, self.clusters = cv2.kmeans(classDescs, 1000, criteria, 10, cv2.KMEANS_PP_CENTERS)   #k = 1000

    def getImageDescriptor(self, index, dataset):
        fileName = dataset.path + self.name + "_" + str(index) + ".jpg"  # hardcoded extension
        img = cv2.imread(fileName)
        keypoints, descriptors = dataset.featureExtractor.detectAndCompute(img, None)
        return descriptors
        

class Classifier:
    def __init__(self, algorithm):
        self.cla = algorithm
        
    def train(self, data):
        values, labels = data.getTrainingData()
        print "VALUES", values.shape, values.dtype
        print "LABELS", labels.shape, labels.dtype
        self.cla.train( values, labels )
        
    def test(self, data):
        #TODO: improve
        for c in data.classes.values():
            print "Checking class %s (%s)" % (c.name, c.label)
            oks = 0
            for bla in xrange(10):
                sample = c.popRandomSample()
                value = c.getImageDescriptor( sample, data)
                #find_nearest(samples, k[, results[, neighborResponses[, dists]]]) -> retval, results, neighborResponses, dists
                response, results, neighborResponsers, dists = self.cla.find_nearest( value, 5 )
                
                #print "\tretval =", retval
                #print "\tresults =", results
                #print "\tneighborResponsers =", neighborResponsers
                #print "\tdists =", dists
                print "[%s] Testing file %s -> %s" % (response==c.label, c.name+"_"+str(sample), data.classNames[int(response)])
                oks += int(response==c.label)
            print "TEST FOR CLASS %s RESULTS: %s/%s" % (c.name, oks, 10)
                  
###################################################

######################################################################################################
# Main
######################################################################################################
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Train and test a bag of visual words classifier on given image dataset.")
    parser.add_argument("datapath", help="Path to the dataset folder.")
    def tpType(strValue):
        tp = float(strValue)
        if tp < 0:  return 0.0
        elif tp > 1.0:  return 1.0
        return tp
    parser.add_argument("--trainingPercentage", "-tp", type=tpType, default=0.1, help="Threshold percentage of samples in each class to use for training.")
    parser.add_argument("--load", "-l", action="store_true", default=False, help="BLABLABLA")

    args = parser.parse_args()
        
    data = DataSet(args.datapath, args.trainingPercentage, cv2.SIFT())
    classifier = Classifier( cv2.KNearest() )
    
    total = 0.0
    mark = time()
    data.load()
    stepTime = time()-mark
    total += stepTime
    print "Dataset carregado em %.2f secs." % (stepTime)
    
    mark = time()
    if args.load:
        classifier.cla.load("./classifiersave")
    else:
        classifier.train(data)
        try:
            classifier.cla.save("./classifiersave")
        except:
            print "ERROR: Could not save classifier to file..."
    stepTime = time()-mark
    total += stepTime
    print "Classificador treinado em %.2f secs." % (stepTime)
    
    mark = time()
    classifier.test(data)
    stepTime = time()-mark
    total += stepTime
    print "Testes rodados em %.2f secs." % (stepTime)
    print "Execução total levou %.2f secs" % (total) 
