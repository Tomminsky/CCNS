#!/Users/Tommy/anaconda/bin python
from __future__ import print_function
import numpy as np
import chainer
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import link
from chainer import reporter
from chainer import optimizers
import chainer.functions as F
import chainer.links as L
import time
from matplotlib import pyplot as plt
import math

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

def get_dictionary(namefile):
    print('loading '+ namefile + '...')
    with open(namefile) as f:
        dict_={line.replace('"', '').replace('\n', ''): int(v) for v, line in enumerate(f)}
    dict_.update({'<eol>': len(dict_)+1})
    print('done.')
    return dict_

def get_titles(namefile,dictionary,shuffle=0):
    print('loading ' + namefile + '...')
    with open(namefile) as file:
        lines = ''
        next(file)
        for line in file:
            lines = lines + line.replace(',0', '').replace('\n', '').replace('"', '') + ',' + str(
                dictionary.get('<eol>')) + ','
    print('done.')
    lines = np.array(map(int, lines[:-1].split(',')))
    if shuffle:
        endoftitles = [x for x in range(len(lines)) if lines[x] == dictionary.get('<eol>')]
        startoftitles = [0] + list(np.add(endoftitles[:-1], 1))
        idx = np.random.permutation(len(endoftitles))
        endoftitles=[endoftitles[x] for x in idx]
        startoftitles = [startoftitles[x] for x in idx]
        lines = [lines[range(startoftitles[x], endoftitles[x] + 1)] for x in range(len(endoftitles))]
        lines = np.hstack(lines)
    return lines

def get_embeddedwords(namefile='word2vec.model'):
    print('loading ' + namefile + '...')
    with open(namefile, 'r') as f:
        ss = f.readline().split()

        n_vocab, n_units = int(ss[0]), int(ss[1])
        word2index = {}
        index2word = {}
        w = np.empty((n_vocab, n_units), dtype=np.float32)
        for i, line in enumerate(f):
            ss = line.split()
            if len(ss)<201:
                ss = [' ']+ss

            word = ss[0]
            word2index[word] = i
            index2word[i] = word
            w[i] = np.array([float(s) for s in ss[1:]], dtype=np.float32)
    w[word2index[' ']] = np.zeros((1, 200))
    print('done.')
    return word2index,index2word,w

def get_max_words_over_titles(titles_raw,dictionary):
    endoftitles = [x for x in range(len(titles_raw)) if titles_raw[x] == dictionary.get('<eol>')]
    startoftitles = [0] + list(np.add(endoftitles[:-1], 1))
    max_title_length_in_batch = max(np.abs(np.subtract(startoftitles, endoftitles))) + 1
    return max_title_length_in_batch

def createtitlebatch(titles_raw,dictionary,skipbatches=0,numtitles=80,testpart=0.1):

    skip_=numtitles*skipbatches

    endoftitles = [x for x in range(len(titles_raw)) if titles_raw[x] == dictionary.get('<eol>')]
    startoftitles = [0] + list(np.add(endoftitles[:-1], 1))

    max_title_length_in_batch=max(np.abs(np.subtract(startoftitles,endoftitles)))+1

    max_skip=len(endoftitles)
    if max_skip<(numtitles+skip_):
        print('maximum requested number of titles is '+ str(numtitles+skip_)+'; dataset only has ' + str(max_skip) + ' titles')
        print('maximum number of batches at ' + str(numtitles) + ' titles per batch: ' + str(max_skip/numtitles-1))
    else:
        title_matrix=[]
        for n in range(skip_,numtitles+skip_):
            title_num=n
            title_vec=titles_raw[range(startoftitles[title_num],endoftitles[title_num]+1)]
            title_matrix.append([w[x-1] for x in title_vec])

        randidx=np.random.permutation(len(title_matrix))
        idx_train=randidx[:-int(np.floor(len(randidx)*(testpart)))]
        idx_test=randidx[int(np.floor(len(randidx)*(1-testpart))):]



        train = [title_matrix[x] for x in idx_train]
        test = [title_matrix[x] for x in idx_test]

        train = [np.concatenate((x,np.zeros((max_title_length_in_batch-len(x),200))),0) for x in train]
        test = [np.concatenate((x, np.zeros((max_title_length_in_batch - len(x), 200))), 0) for x in test]

        return np.asarray(train).reshape((len(train),1,max_title_length_in_batch,200)),np.asarray(test).reshape((len(test),1,max_title_length_in_batch,200))

def vec2title(vec_,dict_trans,index2word):

    title_recon=''
    for i in range(len(vec_)):
        word_ = vec_.data[i]
        word_ = np.tile(word_,(len(w),1))
        dist_=np.sqrt(np.sum((dict_trans-word_)**2,1))
        title_recon=title_recon+index2word[dist_.argmin()]+' '

    return title_recon

class Classifier(link.Chain):

    compute_accuracy = True

    def __init__(self, predictor,
                 lossfun=softmax_cross_entropy.softmax_cross_entropy,
                 accfun=accuracy.accuracy):
        super(Classifier, self).__init__()
        self.lossfun = lossfun
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None

        with self.init_scope():
            self.predictor = predictor

    def __call__(self, *args):
        assert len(args) >= 2
        x = args[:-1]
        t = args[-1]
        self.y = None
        self.loss = None
        self.accuracy = None
        self.y = self.predictor(*x)
        self.loss = self.lossfun(self.y, t)
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y, t)
            reporter.report({'accuracy': self.accuracy}, self)
        return self.loss

class MLPConv(chainer.Chain):
    def __init__(self,words_per_title):
        super(MLPConv, self).__init__()
        with self.init_scope():
            self.words_per_title = words_per_title
            self.l0 = L.Linear(None,words_per_title*200)
            self.conv = L.Convolution2D(in_channels=1, out_channels=1, ksize=3)
            self.l1 = L.Linear(None, 50)
            self.l2 = L.Linear(None, 2)

    def __call__(self, x):
        x1=F.relu(self.l0(x))
        x1 = F.reshape(x1,(x.data.shape[0], 1, self.words_per_title, 200))
        x2 = F.relu(self.conv(x1))
        x3 = F.max_pooling_2d(x2, 3)
        y = F.sigmoid(self.l2(F.dropout(F.relu(self.l1(x3)))))
        return y

word2index,index2word,w=get_embeddedwords()

dictionary=get_dictionary('dictionary.txt')

titles_high_raw=get_titles('titlesDict_high.txt',dictionary,shuffle=1)
titles_low_raw=get_titles('titlesDict_low.txt',dictionary,shuffle=1)

words_per_title = max([get_max_words_over_titles(titles_high_raw,dictionary),get_max_words_over_titles(titles_low_raw,dictionary)])

model = MLPConv(words_per_title)
classifier_model = Classifier(model)

# Setup an optimizer
optimizer = optimizers.AdaGrad()  # Using Stochastic Gradient Descent
optimizer.setup(classifier_model)

n_epoch = 5

num_validation_titles=800

maxiter=1800 # maximum is [number of titles -1600 (2 x 800 for both groups that are used for validation)]/ 80 (default number of titles in one batch) - 80


test_batch_high_raw, _ = createtitlebatch(titles_high_raw, dictionary, skipbatches=0,numtitles=num_validation_titles)
test_batch_low_raw, _ = createtitlebatch(titles_low_raw, dictionary, skipbatches=0,numtitles=num_validation_titles)

shape_diff=test_batch_high_raw.shape[2]-test_batch_low_raw.shape[2]

padd_test=np.zeros((test_batch_high_raw.shape[0],test_batch_high_raw.shape[1],np.abs(shape_diff),test_batch_high_raw.shape[3]))

test_batch_high=test_batch_high_raw
test_batch_low=test_batch_low_raw

if shape_diff != 0:
    padd_=np.zeros((test_batch_high_raw.shape[0],test_batch_high_raw.shape[1],np.abs(shape_diff),test_batch_high_raw.shape[3]))
    if shape_diff>0:
        test_batch_low = np.concatenate((test_batch_low_raw, padd_test), 2)
    elif shape_diff<0:
        test_batch_high = np.concatenate((test_batch_high_raw, padd_test), 2)
test_labels = np.ones((len(test_batch_high), 1))
test_labels = np.concatenate((test_labels, np.zeros((len(test_batch_low), 1))), 0)

test_batch = np.concatenate((test_batch_high, test_batch_low), 0)

N_test = len(test_batch)

start_timer = time.time()

acc_train = []
loss_train = []

acc_val = []
loss_val = []

for epoch in range(n_epoch):

    for iteration in range(10,maxiter):  # start with epoch 10 due to the skip for creating the validation dataset
        print('epoch' , epoch, ' - iteration ', iteration)

        train_batch_high, _ = createtitlebatch(titles_high_raw, dictionary, skipbatches=iteration)

        train_batch_low, _ = createtitlebatch(titles_low_raw, dictionary, skipbatches=iteration)

        shape_diff=train_batch_high.shape[2]-train_batch_low.shape[2]

        if shape_diff != 0:
            padd_=np.zeros((train_batch_high.shape[0],train_batch_high.shape[1],np.abs(shape_diff),train_batch_high.shape[3]))
            if shape_diff>0:
                train_batch_low = np.concatenate((train_batch_low,padd_),2)
            elif shape_diff<0:
                train_batch_high = np.concatenate((train_batch_high, padd_), 2)

        train_labels_high = np.ones((len(train_batch_high), 1))
        train_labels = np.concatenate((train_labels_high, np.zeros((len(train_batch_low), 1))), 0)

        train_batch = np.concatenate((train_batch_high, train_batch_low), 0)

        input = chainer.Variable(train_batch.astype('float32'))
        target = chainer.Variable(train_labels.astype('int32').ravel())

        model.cleargrads()

        predictions = model(input)

        loss = softmax_cross_entropy.softmax_cross_entropy(predictions, target)

        acc = accuracy.accuracy(predictions, target)

        loss_train.append(float(loss.data))
        acc_train.append(float(acc.data))

        loss.backward()
        optimizer.update()

        print('Training current loss =', (float(loss.data)), ',Training current Accuracy =',
              (float(acc.data)))


        #### for Validation the same dataset is used without updating the network ####


        if iteration%25==0:

            perm = np.random.permutation(N_test)  # permutation for the indices
            input = chainer.Variable(test_batch.astype('float32'))
            target = chainer.Variable(test_labels.astype('int32').ravel())

            model.cleargrads()

            predictions = model(input)

            loss = softmax_cross_entropy.softmax_cross_entropy(predictions, target)  # For multi class predictions

            acc = accuracy.accuracy(predictions, target)

            loss_val.append(float(loss.data))
            acc_val.append(float(acc.data))

            print('Validation current loss =', (float(loss.data)), ', Validation current Accuracy =',
                  (float(acc.data)))

    print('time elapsed: ' + str((time.time() - start_timer) / 60) + 'm')
    print('iterations per minute: ' + str((maxiter-10)/((time.time() - start_timer) / 60 / (epoch + 1))))
    print('time per epoch: ' + str((time.time() - start_timer) / 60 / (epoch + 1)) + 'm')

with open('Accuracy_val.txt', 'w') as file_handler:
    for item in acc_val:
        file_handler.write("{}\n".format(item))

with open('Loss_val.txt', 'w') as file_handler:
    for item in loss_val:
        file_handler.write("{}\n".format(item))

with open('Accuracy_train.txt', 'w') as file_handler:
    for item in acc_train:
        file_handler.write("{}\n".format(item))

with open('Loss_train.txt', 'w') as file_handler:
    for item in loss_train:
        file_handler.write("{}\n".format(item))

fig = plt.figure()
plt.plot(acc_val)
plt.title('Acc Val')
plt.savefig('acc_val.eps', format='eps', dpi=600)

fig2 = plt.figure()
plt.plot(loss_val)
plt.title('Loss Val')
plt.savefig('loo_val.eps', format='eps', dpi=600)

fig3 = plt.figure()
plt.plot(mediansmooth(acc_train,50))
plt.title('Acc Train')
plt.savefig('acc_train.eps', format='eps', dpi=600)

fig4 = plt.figure()
plt.plot(mediansmooth(loss_train,50))
plt.title('Loss Train')
plt.savefig('loss_train.eps', format='eps', dpi=600)
plt.show()