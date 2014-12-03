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
    def __init__(self, path, trainingPercentage, randomTrainingSet, featureExtractor):
        self.path = path
        self.featureExtractor = featureExtractor
        self.tp = trainingPercentage
        self.rts = randomTrainingSet
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
            c.generateTrainingSet(self.tp, self.rts)
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
        
    def generateTrainingSet(self, percentage, isRandom):
        if isRandom:
            random.shuffle(self.indexes)
        self.trainIndexes = self.indexes[:int(len(self.indexes)*percentage)]
        del self.indexes[:int(len(self.indexes)*percentage)]
        self.indexes.sort()
        self.trainIndexes.sort()
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
    def __init__(self, algorithm, stateFile):
        self.cla = algorithm
        self.stateFile = stateFile
        
    def train(self, data):
        runTrain = True
        if self.stateFile != "":
            try:
                classifier.cla.load(self.stateFile)
                runTrain = False
            except:
                print "ERROR: Could not save classifier to file..."
        if runTrain:
            values, labels = data.getTrainingData()
            self.cla.train( values, labels )
        
    def runTests(self, data, testsPerClass):
        results = {}
        for c in data.classes.values():
            print "Checking class %s (label %s)" % (c.name, c.label)
            oks = 0
            for bla in xrange(testsPerClass):
                sample = c.popRandomSample()
                value = c.getImageDescriptor( sample, data)

                response = self.predict(value)
                check = response == c.label
                oks += int(check)
                print "\t[%s] Testing file %s -> %s" % ("OK" if check else "--", c.name+"_"+str(sample), data.classNames[int(response)])
            results[c.name] = oks   
            print "\tFinal Results: %s/%s" % (oks, testsPerClass)
        return results
            
    def save(self, filename):
        try:
            self.cla.save(filename)
        except:
            print "ERROR: Could not save classifier to file..."
            
    def predict(self, sample):
        print "WARNING: this should be overwritten in subclasses..."
        return self.cla.predict(sample)
            
                  
class KNNClassifier(Classifier):
    def __init__(self, stateFile):
        Classifier.__init__(self, cv2.KNearest(), stateFile)
        
    def predict(self, sample):
        response, results, neighborResponsers, dists = self.cla.find_nearest(sample, 5 )
        return response
        
class SVMClassifier(Classifier):
    def __init__(self, stateFile):
        Classifier.__init__(self, cv2.SVM(), stateFile)
        
    def predict(self, sample):
        return self.cla.predict(sample)
    
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
    parser.add_argument("--trainingPercentage", "-tp", type=tpType, default=0.1, help="Threshold percentage of samples in each class to use for training. (default 0.1)")
    parser.add_argument("--testsPerClass", "-tpc", type=int, default=10, help="Number of tests per class to run. (default 10)")
    parser.add_argument("--load", "-l", action="store_true", default=False, help="If we should try to load classifier state from state file. (default no)")
    parser.add_argument("--save", "-s", action="store_true", default=False, help="If we should try to save classifier state to state file. (default no)")
    parser.add_argument("--stateFile", "-sf", default="", help="Name of the state file to save/load. (default classfierState)")
    parser.add_argument("--randomTrainingSet", "-rts", action="store_true", default=False, help="If the training set should be defined from random samples from each class from dataset. If not, " +
        "training set will be the first X samples from each class. (default not)")
    claOptions = ["KNN", "SVM"]
    parser.add_argument("--classifier", "-c", choices=claOptions, default=claOptions[0], help="Which classifier to use. (default KNN)")
    extractOptions = ["SIFT", "SURF"]
    parser.add_argument("--extractor", "-e", choices=extractOptions, default=extractOptions[0], help="Which feature extractor to use. (default SIFT)")
    
    #qual classifier

    args = parser.parse_args()
        
    if args.extractor == "SIFT":
        extractor = cv2.SIFT()
    elif args.extractor == "SURF":
        extractor = cv2.SURF(400)
    data = DataSet(args.datapath, args.trainingPercentage, args.randomTrainingSet, extractor)
    
    stateFile = (args.stateFile if args.stateFile != "" else "classifierState") if args.load else ""
    classifier = KNNClassifier( stateFile )
    
    total = 0.0
    mark = time()
    data.load()
    stepTime = time()-mark
    total += stepTime
    print "Dataset carregado em %.2f secs." % (stepTime)
    
    mark = time()
    classifier.train(data)
    if args.save:
        classifier.save(args.stateFile if args.stateFile != "" else "classifierState")
    stepTime = time()-mark
    total += stepTime
    print "Classificador treinado em %.2f secs." % (stepTime)
    
    mark = time()
    results = classifier.runTests(data, args.testsPerClass)
    stepTime = time()-mark
    total += stepTime
    print "Testes rodados em %.2f secs." % (stepTime)
    print "Execução total levou %.2f secs" % (total) 
    
    print "\nResultados dos testes:"
    for className, hits in results.items():
        print "\t%s: %s of %s tests passed (%.1f%%)" % (className, hits, args.testsPerClass, 100*hits/args.testsPerClass)
