from math import log
import operator
from treeplotter import *

def calcshannonent(dataset):
    numberentries=len(dataset)
    labelcount={}
    for vec in dataset:
        currentlabel=vec[-1]
        if currentlabel not in labelcount.keys():
            labelcount[currentlabel]=0
        labelcount[currentlabel]+=1
    shannonent=0.0
    for key in labelcount.keys():
        prob=float(labelcount[key])/numberentries
        shannonent-=prob*log(prob,2)
    return shannonent

def createdataset():
    dataset=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels=['no surfacing','flippers']
    return dataset,labels

def splitdataset(dataset,axis,value):
    retdataset=[]
    for featvec in dataset:
        if featvec[axis]==value:
            reducedfeatvec=featvec[:axis]
            reducedfeatvec.extend(featvec[axis+1:])
            retdataset.append(reducedfeatvec)
    return retdataset

def choosebestsplit(dataset):
    numfeatures=len(dataset[0])-1
    baseent=calcshannonent(dataset)
    bestinfogain=0.0
    bestfeature=-1
    for i in range(numfeatures):
        newent=0.0
        featlist=[elem[i] for elem in dataset]
        uniquevals=set(featlist)
        for value in uniquevals:
            subdataset=splitdataset(dataset,i,value)
            prob=float(len(subdataset))/float(len(dataset))
            newent+=prob*calcshannonent(subdataset)
        infogain=baseent-newent
        if(infogain>bestinfogain):
            bestinfogain=infogain
            bestfeature=i
    return bestfeature

def majoritycnt(classlist):
    classcount={}
    for vote in classlist:
        if vote not in classcount.keys():
            classcount[vote]=0
        classcount[vote]+=1
    sortedclasscount=sorted(classcount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedclasscount[0][0]

def createtree(dataset,labels):
    classlist=[elem[-1] for elem in dataset]
    if classlist.count(classlist[0])==len(classlist):
        return classlist[0]
    if len(dataset[0])==1:
        return majoritycnt(classlist)
    bestfeat=choosebestsplit(dataset)
    bestfeatlabel=labels[bestfeat]
    del(labels[bestfeat])
    mytree={bestfeatlabel:{}}
    featvals=[elem[bestfeat] for elem in dataset]
    uniquevals=set(featvals)
    for value in uniquevals:
        copylabels=labels[:]
        mytree[bestfeatlabel][value]=createtree(splitdataset(dataset,bestfeat,value),copylabels)
    return mytree

def classify(inputtree,featlabels,testvec):
    firststr=list(inputtree.keys())[0]
    seconddict=inputtree[firststr]
    featindex=featlabels.index(firststr)
    for key in seconddict.keys():
        if testvec[featindex]==key:
            if isinstance(seconddict[key],dict):
                classlabel=classify(seconddict[key],featlabels,testvec)
            else:
                classlabel=seconddict[key]
    return classlabel

def storetree(inputtree,filename):
    import pickle
    fw=open(filename,'wb')
    pickle.dump(inputtree,fw)
    fw.close()

def grabtree(filename):
    import pickle
    with open(filename,'rb') as fr:
        return pickle.load(fr)

def lenses(filename):
    with open(filename) as fr:
        lense=[line.strip().split('\t') for line in fr.readlines()]
        lenselabel=['age','prescript','astigmatic','tearrate']
        return lense,lenselabel

lense,lenselabel=lenses('lenses.txt')
lensetree=createtree(lense,lenselabel)
createplot(lensetree)
