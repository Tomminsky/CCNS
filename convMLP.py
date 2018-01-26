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
from chainer.datasets import TupleDataset

def get_dictionary(namefile):
    print('loading '+ namefile + '...')
    with open(namefile) as f:
        dict_={line.replace('"', '').replace('\n', ''): int(v) for v, line in enumerate(f)}
    dict_.update({'<eol>': len(dict_)+1})
    print('done.')
    return dict_

def get_titles(namefile,dictionary):
    print('loading ' + namefile + '...')
    with open(namefile) as file:
        lines = ''
        next(file)
        for line in file:
            lines = lines + line.replace(',0', '').replace('\n', '').replace('"', '') + ',' + str(
                dictionary.get('<eol>')) + ','
    print('done.')
    return np.array(map(int, lines[:-1].split(',')))

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
    print('done.')
    return word2index,index2word,w

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
    def __init__(self):
        super(MLPConv, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(in_channels=1, out_channels=1, ksize=3)
            self.l1 = L.Linear(None, 50)
            self.l2 = L.Linear(None, 2)

    def __call__(self, x):
        x2 = F.relu(self.conv(x))
        x3 = F.max_pooling_2d(x2, 3)
        y = self.l2(F.dropout(F.relu(self.l1(x3))))
        return y

### GAN ###
# only take correctly classified high rank
# inout noise matrix of length of one title vector
# let it work
# get title
word2index,index2word,w=get_embeddedwords()

dictionary=get_dictionary('dictionary.txt')

titles_high_raw=get_titles('titlesDict_high.txt',dictionary)
titles_low_raw=get_titles('titlesDict_low.txt',dictionary)

epoch = 20 # for the convolutionary network 50 training epochs are used
unit = 200
model = MLPConv()
classifier_model = Classifier(model)

# Setup an optimizer
optimizer = optimizers.AdaDelta()  # Using Stochastic Gradient Descent
optimizer.setup(classifier_model)

n_epoch = epoch

accplot_train = np.zeros((n_epoch, 1), dtype=float) # Store train accuracy for plot
lossplot_train = np.zeros((n_epoch, 1), dtype=float)  # Store train loss for plot

epocheloss=[]

num_validation_titles=800

test_batch_raw, _ = createtitlebatch(titles_high_raw, dictionary, skipbatches=0,numtitles=num_validation_titles)
test_batch2_raw, _ = createtitlebatch(titles_low_raw, dictionary, skipbatches=0,numtitles=num_validation_titles)
shape_diff=test_batch_raw.shape[2]-test_batch2_raw.shape[2]
padd_test=np.zeros((test_batch_raw.shape[0],test_batch_raw.shape[1],np.abs(shape_diff),test_batch_raw.shape[3]))

numtitles = len([x for x in range(len(titles_high_raw)) if titles_high_raw[x] == dictionary.get('<eol>')])-num_validation_titles+\
              len([x for x in range(len(titles_low_raw)) if titles_low_raw[x] == dictionary.get('<eol>')])-num_validation_titles

maxiter=1500 # maximum is [number of titles -1600 (2 x 800 for both groups that are used for validation)]/ 80 (default number of titles in one batch) - 80

accplot = np.zeros((maxiter, 1), dtype=float)  # Store  test accuracy for plot
lossplot = np.zeros((maxiter, 1), dtype=float)  # Store test loss for plot

for epoch in range(n_epoch):
    for iteration in range(10,maxiter):  # start with epoch 1 (instead of 0)
        print('epoch' , epoch, ' - iteration ', iteration)  # prompting the word 'epoch ' and the coresponding training epoch to the Python Consol

        # training the MLP with the last chainer method from guide; no cleargrads()!

        train_batch, _ = createtitlebatch(titles_high_raw, dictionary, skipbatches=iteration)

        train_labels=np.ones((len(train_batch),1))
        test_labels = np.ones((len(test_batch_raw), 1))

        train_batch2, _ = createtitlebatch(titles_low_raw, dictionary, skipbatches=iteration)

        shape_diff=train_batch.shape[2]-train_batch2.shape[2]

        test_batch=test_batch_raw
        test_batch2 = test_batch2_raw
        if shape_diff != 0:
            padd_=np.zeros((train_batch.shape[0],train_batch.shape[1],np.abs(shape_diff),train_batch.shape[3]))
            if shape_diff>0:
                train_batch2 = np.concatenate((train_batch2,padd_),2)
                test_batch2 = np.concatenate((test_batch2, padd_test), 2)
            elif shape_diff<0:
                train_batch = np.concatenate((train_batch, padd_), 2)
                test_batch = np.concatenate((test_batch, padd_test), 2)

        train_labels = np.concatenate((train_labels, np.zeros((len(train_batch2), 1))), 0)
        test_labels = np.concatenate((test_labels, np.zeros((len(test_batch2), 1))), 0)

        train_batch = np.concatenate((train_batch, train_batch2), 0)
        test_batch = np.concatenate((test_batch, test_batch2), 0)

        N = len(train_batch)  # training data size
        N_test = len(test_batch)  # test data size

        batchsize = len(train_batch)  # Training batchsize, blackboard specified 32

        sum_accuracy_train = 0  # Creating a staring variable
        sum_loss_train = 0

        input = chainer.Variable(train_batch.astype('float32'))
        target = chainer.Variable(train_labels.astype('int32').ravel())

        model.cleargrads()


        # print(predictions)

        predictions = model(input)
        loss = softmax_cross_entropy.softmax_cross_entropy(predictions, target)  # For multi class predictions

        loss.backward()
        acc = accuracy.accuracy(predictions, target)

        optimizer.update()

        sum_loss_train += float(loss.data) * len(
            target.data)  # Times length of current batch for relative impact
        sum_accuracy_train += float(acc.data) * len(target.data)

        print('Training mean loss =', (sum_loss_train / N), ',Training Accuracy =',
              (sum_accuracy_train / N))  # To check values during process.

        # Testing the model
    sum_accuracy = 0  # Creating a staring variable
    sum_loss = 0
    perm = np.random.permutation(N_test)  # permutation for the indices
    input = chainer.Variable(test_batch.astype('float32'))
    target = chainer.Variable(test_labels.astype('int32').ravel())

    model.cleargrads()

    predictions = model(input)
    # print(predictions)

    loss = softmax_cross_entropy.softmax_cross_entropy(predictions, target)  # For multi class predictions
    # fakeloss = 1-accuracy.accuracy(predictions, target)
    # fakeloss.backward()

    acc = accuracy.accuracy(predictions, target)

    sum_loss += float(loss.data) * len(target.data)  # Times length of current batch for relative impact
    sum_accuracy += float(acc.data) * len(target.data)
    print('Validation mean loss =', (sum_loss / N_test), ', Validation Accuracy =',
          (sum_accuracy / N_test))  # To check values during process.

    accplot_train[epoch] = sum_accuracy / N_test
    lossplot_train[epoch] = sum_loss / N_test



