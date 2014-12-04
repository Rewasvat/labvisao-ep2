#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, sys
import random
import cv2
from time import time
import numpy as np

    
###################################################
def getConsoleWidth():
    #only works on unix
    try:
        rows, columns = os.popen('stty size', 'r').read().split()
        return int(columns)
    except:
        return 80
    
class ProgressBar:
    def __init__(self, text, totalWork):
        self.text = text
        self.totalWork = float(totalWork)
        self.currentWork = 0
        self.mark = time()
        
    def update(self, add=1):
        self.currentWork += add
        sys.stdout.write("\r%s" % self.__str__())
        sys.stdout.flush()
        
    def finish(self):
        sys.stdout.write("\r%s\n" % self.__str__())
        sys.stdout.flush()
        
    def clear(self):
        sys.stdout.write("\r" + (" "*(getConsoleWidth())) + "\r")
        sys.stdout.flush()
        
    def __str__(self):
        s = self.text + ": ["
        p = self.currentWork/self.totalWork
        elapsed = time()-self.mark
        estimated = elapsed * ( (self.totalWork/self.currentWork) - 1 )
        timeText = "] (%.2f secs | %.2f secs)" % (elapsed, estimated)
        barSize = getConsoleWidth() - len(s) - len(timeText)
        barDone = int(barSize*p)
        s += "=" * barDone
        s += " " * (barSize - barDone)
        s += timeText
        s += " " * (getConsoleWidth() - len(s))
        return s
        
        

###################################################
class DataSet:
    def __init__(self, path, trainingPercentage, k, randomTrainingSet, featureExtractor):
        self.path = path
        self.featureExtractor = featureExtractor
        self.tp = trainingPercentage
        self.k = k
        self.rts = randomTrainingSet
        self.classes = {}
        self.classNames = []
        self.clusters = None
        self.bag = None
        
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
        trainingDescriptors = None

        for c in self.classes.values():
            mark = time()
            # separate training set from class image set
            c.generateTrainingSet(self.tp, self.rts)
            # generate descriptors
            descs = c.getTrainingDescriptors(self)
            if trainingDescriptors == None:
                trainingDescriptors = descs
            else:
                trainingDescriptors = np.append(trainingDescriptors, descs, 0)
            stepTime = time()-mark
            print "Calculated descriptors for %s (%.2f secs)" % (c.name, stepTime)
            
        # Clusterize (generate vocabulary)
        mark = time()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        compactness, bestLabels, self.clusters = cv2.kmeans(trainingDescriptors, self.k, criteria, 10, cv2.KMEANS_PP_CENTERS)   #k = 1000
        stepTime = time()-mark
        print "Generated clusters (vocabulary) in %.2f secs" % (stepTime)
        
        mark = time()
        self.bag = cv2.KNearest()
        self.bag.train(self.clusters, np.arange(self.k))
        stepTime = time()-mark
        print "Created bag of visual words from clusters in %.2f secs" % (stepTime)
        
        for c in self.classes.values():
            hists = c.getTrainingHistograms(self)
            if values == None:
                values = hists
            else:
                values = np.append(values, hists, 0)
            labels += [c.label]*hists.shape[0]
        return values, np.asarray( labels, dtype='float32')
            
###################################################  
class DataClass:
    def __init__(self, name, label):
        self.name = name
        self.label = label
        self.indexes = []
        self.trainIndexes = []
        
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
        if len(self.indexes) == 0:  return None
        sample = random.choice(self.indexes)
        self.indexes.remove(sample)
        return sample

    def getImageDescriptor(self, index, dataset):
        fileName = dataset.path + self.name + "_" + str(index) + ".jpg"  # hardcoded extension
        img = cv2.imread(fileName)
        keypoints, descriptors = dataset.featureExtractor.detectAndCompute(img, None)
        return descriptors
        
    def getTrainingDescriptors(self, dataset):
        classDescs = None
        # Get the descriptors for each image in training set
        for index in self.trainIndexes:
            descs = self.getImageDescriptor(index, dataset)
            classDescs = descs if classDescs==None else np.append(classDescs, descs, 0)
        return classDescs
        
    def getImageHistogram(self, index, dataset):
        descs = self.getImageDescriptor(index, dataset)
        response, results, neighborResponsers, dists = dataset.bag.find_nearest(descs, 5 )
        hist = np.zeros( (1, dataset.k), dtype='float32' )
        for i in results:
            hist[0][int(i[0])] += 1
        return hist
        
    def getTrainingHistograms(self, dataset):
        classHists = None
        # Get the histogram for each image in training set
        #print self.name, "TRAINING HISTOGRAMS"
        bar = ProgressBar("Pegando Histogramas de "+self.name, len(self.trainIndexes))
        for index in self.trainIndexes:
            hist = self.getImageHistogram(index, dataset)
            classHists = hist if classHists==None else np.append(classHists, hist, 0)
            bar.update()
        bar.finish()
        return classHists 

###################################################
class Classifier:
    def __init__(self, algorithm, stateFile):
        self.cla = algorithm
        self.stateFile = stateFile
        
    def train(self, data):
        runTrain = True
        values, labels = data.getTrainingData()
        if self.stateFile != "":
            try:
                classifier.cla.load(self.stateFile)
                runTrain = False
            except:
                print "ERROR: Could not load classifier from file..."
        if runTrain:
            self._doTrain( values, labels )
        
    def runTests(self, data, testsPerClass):
        results = {}
        for c in data.classes.values():
            print "Checking class %s (label %s)" % (c.name, c.label)
            oks = 0
            testsRun = 0
            for bla in xrange(testsPerClass):
                sample = c.popRandomSample()
                if sample == None:
                    break
                value = c.getImageHistogram(sample, data)
                testsRun += 1
                response = self.predict(value)
                check = response == c.label
                oks += int(check)
                print "\t[%s] Testing file %s -> %s" % ("OK" if check else "--", c.name+"_"+str(sample), data.classNames[int(response)])
            results[c.name] = (oks, testsRun)
            print "\tFinal Results: %s/%s" % (oks, testsRun)
        return results
            
    def save(self, filename):
        try:
            self.cla.save(filename)
        except:
            print "ERROR: Could not save classifier to file..."
            
    def predict(self, sample):
        print "WARNING: this should be overwritten in subclasses..."
        return self.cla.predict(sample)
            
###################################################  
class KNNClassifier(Classifier):
    def __init__(self, stateFile):
        Classifier.__init__(self, cv2.KNearest(), stateFile)
        
    def predict(self, sample):
        response, results, neighborResponsers, dists = self.cla.find_nearest(sample, 5 )
        return response
    def _doTrain(self, values, labels):
        self.cla.train(values, labels)

###################################################
class SVMClassifier(Classifier):
    def __init__(self, stateFile):
        Classifier.__init__(self, cv2.SVM(), stateFile)
        
    def predict(self, sample):
        return self.cla.predict(sample)
    def _doTrain(self, values, labels):
        svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                           svm_type = cv2.SVM_C_SVC,
                           C=2.67, gamma=5.383 )
        self.cla.train(values, labels, params=svm_params)
    
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
    parser.add_argument("--K", "-k", type=int, default=1000, help="Number of clusters in bag of visual words vocabulary. (default 1000)")
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
    
    args = parser.parse_args()
        
    if args.extractor == "SIFT":
        extractor = cv2.SIFT()
    elif args.extractor == "SURF":
        extractor = cv2.SURF(400)
    data = DataSet(args.datapath, args.trainingPercentage, args.K, args.randomTrainingSet, extractor)
    
    stateFile = (args.stateFile if args.stateFile != "" else "classifierState") if args.load else ""
    if args.classifier == "KNN":
        classifier = KNNClassifier( stateFile )
    elif args.classifier == "SVM":
        classifier = SVMClassifier( stateFile )
    
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
    for className, res in results.items():
        print "\t%s: %s of %s tests passed (%.1f%%)" % (className, res[0], res[1], 100*res[0]/res[1])
