from numpy import *

def load_data_set():
    data_mat=[]
    label_mat=[]
    with open('testSet.txt') as fr:
        for line in fr.readlines():
            line_arr=line.strip().split()
            data_mat.append([1.0,float(line_arr[0]),float(line_arr[1])])
            label_mat.append(int(line_arr[2]))
    return data_mat,label_mat

def sigmoid(intx):    
    return 1.0/(1.0+exp(-intx))

def grad_ascent(data_mat_in,label_mat_in):
    data_mat=mat(data_mat_in)
    label_mat=mat(label_mat_in).transpose()
    m,n=shape(data_mat)
    alpha=0.001
    max_cycles=500
    weights=ones([n,1])
    for i in range(max_cycles):
        h=sigmoid(data_mat*weights)
        error=label_mat-h
        weights=weights+alpha*data_mat.transpose()*error
    return weights

def plot_best_fit(weights):
    import matplotlib.pyplot as plt
    data_mat,label_mat=load_data_set()
    data_arr=array(data_mat)
    n=shape(data_arr)[0]
    xcord1=[]
    ycord1=[]
    xcord2=[]
    ycord2=[]
    for i in range(n):
        if label_mat[i]==1:
            xcord1.append(data_arr[i,1])
            ycord1.append(data_arr[i,2])
        else:
            xcord2.append(data_arr[i,1])
            ycord2.append(data_arr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

def stoc_grad_ascent0(data_arr,label_arr):
    m,n=shape(data_arr)
    alpha=0.01
    weights=ones(n)
    for i in range(m):
        h=sigmoid(sum(data_arr[i]*weights))
        error=label_arr[i]-h
        weights=weights+alpha*error*data_arr[i]
    return weights

def stoc_grad_ascent1(data_arr,label_arr,num_iter=150):
    m,n=shape(data_arr)
    weights=ones(n)
    for j in range(num_iter):
        data_index=list(range(m))
        for i in range(m):
            alpha=4.0/(1.0+i+j)+0.01
            ran_index=int(random.uniform(0,len(data_index)))
            h=sigmoid(sum(data_arr[data_index[ran_index]]*weights))
            error=label_arr[data_index[ran_index]]-h
            weights=weights+alpha*error*data_arr[data_index[ran_index]]
            del(data_index[ran_index])
    return weights

def classify_vec(inx,weights):
    prob=sigmoid(sum(inx*weights))
    if prob>0.5:
        return 1.0
    else:
        return 0.0

def colic_test():
    training_set=[]
    training_labels=[]
    with open('horseColicTraining.txt') as ftr:
        for line in ftr.readlines():
            line_vec=line.strip().split('\t')
            line_arr=[]
            for i in range(21):
                line_arr.append(float(line_vec[i])/10.0)
            training_set.append(line_arr)
            training_labels.append(float(line_vec[21]))
    weights=stoc_grad_ascent1(array(training_set),training_labels,500)
    error_count=0.0
    num_test=0.0
    with open('horseColicTest.txt') as fte:
        for line in fte.readlines():
            num_test+=1.0
            line_vec=line.strip().split('\t')
            line_arr=[]
            for i in range(21):
                line_arr.append(float(line_vec[i])/10.0)
            if int(classify_vec(array(line_arr),weights))!=int(line_vec[21]):
                error_count+=1.0
    error_rate=error_count/num_test
    print('the error rate of this test is: %f' % error_rate)
    return error_rate
            
def multi_test():
    num_tests=10
    error_sum=0.0
    for i in range(num_tests):
        error_sum+=colic_test()
    print('after %d iterations the average error is: %f' % (num_tests,error_sum/float(num_tests)))

multi_test()