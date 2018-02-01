from __future__ import print_function
import numpy as np
import math
from matplotlib import pyplot as plt

# This code is used for plotting results only

def mediansmooth(datain, kernelwidth):

    # note: this guy is only for plotting purposes and does what you'd expect: a median smooth

    padd_beg = int(math.ceil((float(kernelwidth) / 2)))
    padd_end = int(math.floor((float(kernelwidth) / 2)))

    padd_beg = np.empty((padd_beg))
    padd_beg[:] = np.NAN

    padd_end = np.empty((padd_end))
    padd_end[:] = np.NAN

    datatmp = np.concatenate((padd_beg, datain, padd_end))

    x = np.size(np.matrix(datatmp), 1)
    data_pre = np.empty((x, kernelwidth), dtype=np.float32)
    data_pre[:] = np.NAN

    for i in range(0, kernelwidth):
        data_pre[range(0, x - i), i] = datatmp[range(i, x)]

    data_pre = data_pre[range(1, x - kernelwidth + 1)]
    data_out = np.nanmedian(data_pre, axis=1)

    return data_out

def citationDescripitives():
    with open('citationsDict.txt') as file:
        lines = []
        next(file)
        for line in file:
            lines.append(float(line))


    hist_ = np.histogram(lines,577635)[0]
    up_p = np.percentile(lines,75)
    low_p = np.percentile(lines,25)

    x = range(200)
    x[200:] = range(400,600)
    x[400:] = range(2000,2200)

    y = hist_[x]

    f,(ax,ax2,ax3) = plt.subplots(1,3,sharey=True, facecolor='w')

    # plot the same data on both axes
    ax.plot(x, y)

    ax2.plot(x, y)

    ax3.plot(x, y)

    ax.set_xlim(0,200)
    ax2.set_xlim(400,600)
    ax3.set_xlim(2000,2200)

    # hide the spines between ax and ax2
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('prevalence')
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax.yaxis.tick_left()
    ax.tick_params(labelright='off')
    ax2.yaxis.tick_right()
    ax2.yaxis.tick_left()
    ax3.yaxis.tick_left()

    d = .015
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1-d,1+d), (-d,+d), **kwargs)
    ax.plot((1-d,1+d),(1-d,1+d), **kwargs)

    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d,+d), (1-d,1+d), **kwargs)
    ax2.plot((-d,+d), (-d,+d), **kwargs)
    ax2.plot((1-d,1+d), (-d,+d), **kwargs)
    ax2.plot((1-d,1+d),(1-d,1+d), **kwargs)

    kwargs.update(transform=ax3.transAxes)
    ax3.plot((-d,+d), (1-d,1+d), **kwargs)
    ax3.plot((-d,+d), (-d,+d), **kwargs)
    plt.title('distribution of number of citations')

    plt.show()

def plotforConv():

    with open('Accuracy_val.txt') as file:
        acc_val = []
        next(file)
        for line in file:
            acc_val.append(float(line))

    with open('Loss_val.txt') as file:
        loss = []
        next(file)
        for line in file:
            loss.append(float(line))

    _,(ax,ax3) = plt.subplots(2,1)
    ax2 = ax.twinx()

    ax.plot(range(1,len(acc_val)+1),acc_val,color=[0.8,0.2,0.2])
    ax.set_ylabel('Accuracy',color=[0.8,0.2,0.2])
    ax2.plot(range(1,len(loss)+1),loss,color=[0.2,0.2,0.8])
    ax2.set_ylabel('Loss',color=[0.2,0.2,0.8])
    plt.vlines(range(72,len(acc_val),72) ,min([ax.get_ylim()[0],ax2.get_ylim()[0]]),max([ax.get_ylim()[1],ax2.get_ylim()[1]]),label='onset new epoch')
    plt.title('Validation')
    plt.xticks([])
    plt.legend(loc=2)
    with open('Accuracy_train.txt') as file:
        acc = []
        next(file)
        for line in file:
            acc.append(float(line))

    with open('Loss_train.txt') as file:
        loss = []
        next(file)
        for line in file:
            loss.append(float(line))

    ax4 = ax3.twinx()
    ax3.plot(range(1,len(acc)+1),mediansmooth(acc,50),color=[0.8,0.2,0.2])
    ax3.set_ylabel('Accuracy',color=[0.8,0.2,0.2])
    ax3.set_xlabel('Number of Training Iterations')
    ax4.plot(range(1,len(loss)+1),mediansmooth(loss,50),color=[0.2,0.2,0.8])
    ax4.set_ylabel('Loss',color=[0.2,0.2,0.8])
    plt.title('Training')
    plt.xticks([2]+range(50*25,len(acc),50*25),[x*25 for x in [2]+range(50,len(acc_val),50)*25])

    plt.vlines(range(72*25,len(acc),72*25) ,min([ax3.get_ylim()[0],ax4.get_ylim()[0]]),max([ax3.get_ylim()[1],ax4.get_ylim()[1]]))

    plt.show()
    plt.savefig('results_ConvDict.eps', format='eps', dpi=600)

def plotforGAN():
    with open('Accuracy_trainGAN.txt') as file:
        acc = []
        next(file)
        for line in file:
            acc.append(float(line))

    with open('Loss_DisGAN.txt') as file:
        loss_dis = []
        next(file)
        for line in file:
            loss_dis.append(float(line))

    _, (ax, ax3) = plt.subplots(2, 1)
    ax2 = ax.twinx()

    ax.plot(range(1, len(acc) + 1), acc, color=[0.8,0.2,0.2])
    ax.set_ylabel('Accuracy', color=[0.8,0.2,0.2])
    ax2.plot(range(1, len(loss_dis) + 1), loss_dis, color=[0.2,0.2,0.8])
    ax2.set_ylabel('Loss', color=[0.2,0.2,0.8])
    plt.title('Discriminator')
    plt.xticks([])

    with open('Loss_GenGAN.txt') as file:
        loss_gen = []
        next(file)
        for line in file:
            loss_gen.append(float(line))
    ax3.plot(range(1, len(acc) + 1), mediansmooth(loss_gen, 50), color=[0.2,0.2,0.8])
    ax3.set_ylabel('Loss', color=[0.2,0.2,0.8] )
    ax3.set_title('Generator')
    ax3.yaxis.set_ticks_position('right')
    ax3.yaxis.set_label_position('right')
    ax3.set_xlabel('Number of Training Iterations')
    plt.show()
    plt.savefig('results_GANDict.eps', format='eps', dpi=600)

plotforConv()
plotforGAN()


