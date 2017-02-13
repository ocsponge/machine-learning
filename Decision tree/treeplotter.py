import matplotlib.pyplot as plt

decisionnode = dict(boxstyle='sawtooth', fc='0.8')
leafnode = dict(boxstyle='round4', fc='0.8')
arrow = dict(arrowstyle='<-')


def plotnode(nodetext, centerpoint, parentpoint, nodetype):
    createplot.ax1.annotate(nodetext, xy=parentpoint, xycoords='axes fraction', xytext=centerpoint,
                            textcoords='axes fraction', va='center', ha='center', bbox=nodetype, arrowprops=arrow)


def createplot(intree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createplot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plottree.totalw = float(getnumleafs(intree))
    plottree.totald = float(gettreedepth(intree))
    plottree.xoff = -0.5 / plottree.totalw
    plottree.yoff = 1.0
    plottree(intree, (0.5, 1.0), '')
    plt.show()


def getnumleafs(mytree):
    numleafs = 0
    firstfeat = list(mytree.keys())[0]
    seconddict = mytree[firstfeat]
    for key in seconddict.keys():
        if isinstance(seconddict[key],dict):
            numleafs += getnumleafs(seconddict[key])
        else:
            numleafs += 1
    return numleafs


def gettreedepth(mytree):
    maxdepth = 0
    firstfeat = list(mytree.keys())[0]
    seconddict = mytree[firstfeat]
    for key in seconddict.keys():
        if isinstance(seconddict[key],dict):
            thisdepth = 1 + gettreedepth(seconddict[key])
        else:
            thisdepth = 1
        if thisdepth > maxdepth:
            maxdepth = thisdepth
    return maxdepth


def retrievetree(i):
    listoftrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
    return listoftrees[i]

def plotmidtext(cntrpt, parentpt, txtstr):
    xmid = (parentpt[0] - cntrpt[0]) / 2.0 + cntrpt[0]
    ymid = (parentpt[1] - cntrpt[1]) / 2.0 + cntrpt[1]
    createplot.ax1.text(xmid, ymid, txtstr)


def plottree(mytree, parentpt, nodetxt):
    numleafs = getnumleafs(mytree)
    depth = gettreedepth(mytree)
    firstfeat = list(mytree.keys())[0]
    cntrpt = (plottree.xoff + (1.0+ float(numleafs)) / 2.0 / plottree.totalw, plottree.yoff)
    plotmidtext(cntrpt, parentpt, nodetxt)
    plotnode(firstfeat, cntrpt, parentpt, decisionnode)
    seconddict = mytree[firstfeat]
    plottree.yoff = plottree.yoff - 1.0 / plottree.totald
    for key in seconddict.keys():
        if isinstance(seconddict[key],dict):
            plottree(seconddict[key], cntrpt, str(key))
            plottree.yoff = plottree.yoff + 1.0 / plottree.totald
        else:
            plottree.xoff = plottree.xoff + 1.0 / plottree.totalw
            plotnode(seconddict[key], (plottree.xoff, plottree.yoff), cntrpt, leafnode)
            plotmidtext((plottree.xoff, plottree.yoff), cntrpt, str(key))
    