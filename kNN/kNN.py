from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir


def createdataset():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inx, dataset, labels, k):
    datasetsize = dataset.shape[0]
    diffmat = tile(inx, (datasetsize, 1)) - dataset
    sqdiffmat = diffmat**2
    sqdistances = sqdiffmat.sum(axis=1)
    distances = sqdistances**0.5
    sorteddistances = distances.argsort()
    classcount = {}
    for i in range(k):
        votelabel = labels[sorteddistances[i]]
        classcount[votelabel] = classcount.get(votelabel, 0) + 1
    sortedclass = sorted(classcount.items(),
                         key=operator.itemgetter(1), reverse=True)
    return sortedclass[0][0]


def file2matrix(filename):
    with open(filename) as fr:
        arrayolines = fr.readlines()
        numberlines = len(arrayolines)
        returnmat = zeros((numberlines, 3))
        classlabel = []
        index = 0
        for line in arrayolines:
            line = line.strip()
            listfromline = line.split('\t')
            returnmat[index, :] = listfromline[0:3]
            classlabel.append(int(listfromline[-1]))
            index += 1
        return returnmat, classlabel


def autonorm(dataset):
    minvals = dataset.min(0)
    maxvals = dataset.max(0)
    ranges = maxvals - minvals
    m = dataset.shape[0]
    normdataset = dataset - tile(minvals, (m, 1))
    normdataset = normdataset / tile(ranges, (m, 1))
    return normdataset, ranges, minvals


def datingclasstest():
    horatio = 0.1
    datingdatamat, datinglabels = file2matrix('datingTestSet2.txt')
    normmat, ranges, minvals = autonorm(datingdatamat)
    m = normmat.shape[0]
    errorcount = 0.0
    numtest = int(m * horatio)
    for i in range(numtest):
        classifierresult = classify0(normmat[i, :], normmat[numtest:m, :],
                                     datinglabels[numtest:m], 3)
        print('the classifier came back with: %d, the real answer is %d' %
              (classifierresult, datinglabels[i]))
        if (classifierresult != datinglabels[i]):
            errorcount += 1.0
    print('the total error rate is: %f' % (errorcount / float(numtest)))


def classifyperson():
    resultlist = ['not at all', 'in small doses', 'in large doses']
    percenttats = float(input('percentage of time spent playing video games:'))
    ffmiles = float(input('frequent flier miles earned per year:'))
    icecream = float(input('liters of ice cream consumed per year:'))
    datingdatamat, datinglabels = file2matrix('datingTestSet2.txt')
    normmat, ranges, minvals = autonorm(datingdatamat)
    intarr = array([ffmiles, percenttats, icecream])
    classifierresult = classify0((intarr - minvals) / ranges, normmat,
                                 datinglabels, 3)
    print('you will probably like this person:',
          resultlist[classifierresult - 1])


def img2vector(filename):
    returnvec = zeros((1, 1024))
    with open(filename) as fr:
        for i in range(32):
            line = fr.readline()
            for j in range(32):
                returnvec[0, 32 * i + j] = int(line[j])
        return returnvec


def handwriteclasstest():
    hwlabels = []
    trainingfilelist = listdir('trainingDigits')
    m = len(trainingfilelist)
    trainingmat = zeros((m, 1024))
    for i in range(m):
        filename = trainingfilelist[i]
        filestr = filename.split('.')[0]
        classnum = int(filestr.split('_')[0])
        hwlabels.append(classnum)
        trainingmat[i, :] = img2vector('trainingDigits/%s' % filename)
    testfilelist = listdir('testDigits')
    mtest = len(testfilelist)
    errorcount = 0.0
    for i in range(mtest):
        filename = testfilelist[i]
        filestr = filename.split('.')[0]
        classnum = int(filestr.split('_')[0])
        vectortest = img2vector('testDigits/%s' % filename)
        classifierresult = classify0(vectortest, trainingmat, hwlabels, 3)
        print('the classifier came back with: %d, the real answer is: %d' %
              (classifierresult, classnum))
        if (classifierresult != classnum):
            errorcount += 1.0
    print('\nthe total number of errors is: %d' % errorcount)
    print('\nthe total error rate is: %f' % (errorcount / float(mtest)))

