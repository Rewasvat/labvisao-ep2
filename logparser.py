#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, sys
from collections import namedtuple
from visualbags import ProgressBar, getConsoleWidth

LogData = namedtuple("LogData", "numClasses descs kmeans hists train testes total numTestes numAcertos classHits imgHits imgMisses")
      
def parse(filename):
    f = open(filename, "r")
    lines = f.readlines()
    f.close()
    
    numClasses = 0 #numero de classes
    descs = 0  # tempo de calculo descritores
    kmeans = 0 #tempo que levou o kmeans
    hists = 0 #tempo de calculo de histogramas
    train = 0 #tempo total treinamento
    testes = 0 #tempo total testes
    total = 0 #tempo total
    numTestes = 0  #numero total de testes
    numAcertos = 0 #numero total de acertos
    inTestes = False
    classHits = {}
    imgHits = {}
    imgMisses = {}
    bar = ProgressBar("Parseando "+filename, len(lines))
    for line in lines:
        if inTestes:
            parts = line.split()
            num = int( parts[3] )
            oks = int( parts[1] )
            className = parts[0][:-1]
            classHits[className] = 100.0 * float(oks) / num
            numTestes += num
            numAcertos += oks
        elif line.startswith("Calculated descriptors"):
            descs += float( line.split("(")[1].split(")")[0].split()[0] )
            numClasses += 1
        elif line.startswith("Generated clusters"):
            kmeans = float( line.split()[4] )
        elif line.startswith("Pegando Histogramas"):
            hists += float( line.split("(")[1].split()[0] )
        elif line.startswith("Classificador treinado"):
            train = float( line.split()[3] )
        elif line.startswith("Testes rodados"):
            testes = float( line.split()[3] )
        elif line.startswith("Execução total"):
            total = float( line.split()[3] )
        elif line.startswith("Resultados dos testes"):
            inTestes = True
        elif line.startswith("	["):
            ok = (line.split("[")[1].split("]")[0] ) == "OK"
            index = int( line.split()[3].split("_")[-1] )
            className = "_".join( line.split()[3].split("_")[:-1] )
            if ok:
                if not className in imgHits:
                    imgHits[className] = []
                imgHits[className].append(index)
            else:
                if not className in imgMisses:
                    imgMisses[className] = []
                imgMisses[className].append(index)
        bar.update()
    bar.finish()
            
    return LogData(numClasses, descs, kmeans, hists, train, testes, total, numTestes, numAcertos, classHits, imgHits, imgMisses)

######################################################################################################
# Main
######################################################################################################
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse the output of visualbags.py to generate a more latex friendly data report.")
    parser.add_argument("logfiles", nargs='+', help="Name of log files to parse.")

    args = parser.parse_args()
        
    data = {}
    for filename in args.logfiles:
        data[filename] = parse(filename)
        
#teste1   numClas  descs  kmeans  hists  train testes total testesQuePassaram% 
#teste2   numClas  descs  kmeans  hists  train testes total testesQuePassaram%
    print "\\begin{tabular}{ | l | c | c | c | c | c | c | c | r | }"
    print "\\hline"
    print "Teste       & Classes & Descs & KMeans & Hists & Treino & Testes & Total & AT \\\\"
    print "\\hline"
    logs = data.keys()
    logs.sort()
    for log in logs:
        d = data[log]
        AT = 100.0 * float(d.numAcertos) / d.numTestes
        s = log[7:-4].replace("_", "\\_")
        print "%s & %s & %.2fs & %.2fs & %.2fs & %.2fs & %.2fs & %.2fs & %.1f\\%% \\\\" % (s, d.numClasses, d.descs, d.kmeans, d.hists, d.train, d.testes, d.total, AT)
        print "\\hline"
    print "\\end{tabular}"
    
    print ""
    
#teste1   class1hits class2hits ...
    print "\\begin{tabular}{ | l | c | c | c | c | c | c | c | c | c | c | }"
    print "\\hline"
    print "Teste       & A & B & C & D & E & F & G & H & I & J \\\\"
    print "\\hline"
    for log in logs:
        d = data[log].classHits
        s = log[7:-4].replace("_", "\\_")
        def f(name):
            if name in d:
                return "%.1f\\%%" % (d[name])
            return "--"
        print ("%s" + (" & %s"*10) + " \\\\") % (s, f("leonberger"), f("english_setter"), f("pug"), f("basset_hound"), f("saint_bernard"),
                     f("beagle"), f("shiba_inu"), f("english_cocker_spaniel"), f("great_pyrenees"), f("newfoundland"))
        print "\\hline"
    print "\\end{tabular}"
    
    print ""
# quais imagens que acertaram em todos testes

    def getInAll(doHits):
        imgBoas = None
        for log in logs:
            if doHits:
                d = data[log].imgHits
            else:
                d = data[log].imgMisses
            if imgBoas == None:
                imgBoas = d
                continue
            for k in d.keys():
                imgs = d[k]
                boas = list(imgBoas[k])
                for i in boas:
                    if not i in imgs:
                        imgBoas[k].remove(i)
        return imgBoas

    print "PERFECT HITS:", getInAll(True)
    print "\nPERFECT MISSES:", getInAll(False)
    
