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

# median smooth function used for plotting only
def mediansmooth(datain, kernelwidth):

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

# load and arrange the dictionary file
def get_dictionary(namefile):
    print('loading '+ namefile + '...')
    with open(namefile) as f:
        # all " strings are removed and the data is split at each line break
        dict_={line.replace('"', '').replace('\n', ''): int(v) for v, line in enumerate(f)}
    # end of line statement <eol> is added to the dictionary
    dict_.update({'<eol>': len(dict_)+1})
    print('done.')
    return dict_

# load and arrange the title file
def get_titles(namefile,dictionary,shuffle=0):
    print('loading ' + namefile + '...')
    with open(namefile) as file:
        lines = ''
        next(file)
        for line in file:
            # all zeros are removed and the data is split at each line break
            # <eol> is added at the end of each title
            lines = lines + line.replace(',0', '').replace('\n', '').replace('"', '') + ',' + str(
                dictionary.get('<eol>')) + ','
    print('done.')
    lines = np.array(map(int, lines[:-1].split(',')))
    # if desired the titles are shuffled
    if shuffle:
        endoftitles = [x for x in range(len(lines)) if lines[x] == dictionary.get('<eol>')]
        startoftitles = [0] + list(np.add(endoftitles[:-1], 1))
        idx = np.random.permutation(len(endoftitles))
        endoftitles=[endoftitles[x] for x in idx]
        startoftitles = [startoftitles[x] for x in idx]
        lines = [lines[range(startoftitles[x], endoftitles[x] + 1)] for x in range(len(endoftitles))]
        lines = np.hstack(lines)
    # the function returns a vector containing all dictionary indices of all titles
    return lines

# load and arrange the word embedding file
def get_embeddedwords(namefile='word2vec.model'):
    print('loading ' + namefile + '...')
    with open(namefile, 'r') as f:
        ss = f.readline().split()
        # each line is split and the respective human readable word and the vector embedding vector is extracted
        n_vocab, n_units = int(ss[0]), int(ss[1])
        word2index = {}
        index2word = {}
        w = np.empty((n_vocab, n_units), dtype=np.float32)
        # the embedding matrix is created by sorting all word vectors according to the dictionary index
        # the resulting matrix is of size NumIndices x 200
        # note that due to splitting white space got removed. For that reason it is added again and a vector of
        # zeros is used within w
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
    # word2index transforms a dictionary index to a human readable word
    # index2wordtransforms a human readable word to a dictionary index
    return word2index,index2word,w

# this function is used to obtain the maximum number of words accross all titles by finding <eol> statements
def get_max_words_over_titles(titles_raw,dictionary):
    endoftitles = [x for x in range(len(titles_raw)) if titles_raw[x] == dictionary.get('<eol>')]
    startoftitles = [0] + list(np.add(endoftitles[:-1], 1))
    max_title_length_in_batch = max(np.abs(np.subtract(startoftitles, endoftitles))) + 1
    return max_title_length_in_batch

# this function creates batch data used to train the network
def createtitlebatch(titles_raw,dictionary,w,skipbatches=0,numtitles=80,testpart=0.1):

    # skip_ is used to select parts of the title vector
    # skip_ = 10 given numtitles = 80 would mean that the first 800 titles in the vector are skipped and
    # all following operations are performed on data that comes after that

    skip_=numtitles*skipbatches

    endoftitles = [x for x in range(len(titles_raw)) if titles_raw[x] == dictionary.get('<eol>')]
    startoftitles = [0] + list(np.add(endoftitles[:-1], 1))

    max_title_length_in_batch=max(np.abs(np.subtract(startoftitles,endoftitles)))+1

    max_skip=len(endoftitles)
    if max_skip<(numtitles+skip_):
        print('maximum requested number of titles is '+ str(numtitles+skip_)+'; dataset only has ' + str(max_skip) + ' titles')
        print('maximum number of batches at ' + str(numtitles) + ' titles per batch: ' + str(max_skip/numtitles-1))
    else:
        # extraction of the data from w given the amount of titles selected
        title_matrix=[]
        for n in range(skip_,numtitles+skip_):
            title_num=n
            title_vec=titles_raw[range(startoftitles[title_num],endoftitles[title_num]+1)]
            title_matrix.append([w[x-1] for x in title_vec])

        # shuffling the selected batch
        randidx=np.random.permutation(len(title_matrix))
        idx_train=randidx[:-int(np.floor(len(randidx)*(testpart)))]
        idx_test=randidx[int(np.floor(len(randidx)*(1-testpart))):]

        train = [title_matrix[x] for x in idx_train]
        test = [title_matrix[x] for x in idx_test]

        train = [np.concatenate((x,np.zeros((max_title_length_in_batch-len(x),200))),0) for x in train]
        test = [np.concatenate((x, np.zeros((max_title_length_in_batch - len(x), 200))), 0) for x in test]

        # train and test are returned in a shape optimized for the use with convolutional networks
        # the respective shape is numExamples x channel x numWords x ndimEmbedding

        return np.asarray(train).reshape((len(train),1,max_title_length_in_batch,200)),np.asarray(test).reshape((len(test),1,max_title_length_in_batch,200))

# this function can be used to transform a title matrix (words x ndim) back into a human readable title
# the argmin() L2 norm between each word vector of the title and all word vectors in w serves as the
# index for the respective word in the dictionary
def vec2title(vec_,dict_trans,index2word):

    title_recon=''
    for i in range(len(vec_)):
        word_ = vec_.data[i]
        word_ = np.tile(word_,(len(w),1))
        dist_=np.sqrt(np.sum((dict_trans-word_)**2,1))
        title_recon=title_recon+index2word[dist_.argmin()]+' '

    return title_recon

# classifier to compute loss based on softmax cross entropy and accuracy
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

# Convolutional neuronal network to do the classification
# respective architecture choices are explained in the report
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

# loading the respective data
word2index,index2word,w=get_embeddedwords()

dictionary=get_dictionary('dictionary.txt')

titles_high_raw=get_titles('titlesDict_high.txt',dictionary,shuffle=1)
titles_low_raw=get_titles('titlesDict_low.txt',dictionary,shuffle=1)

# get maximum number of words in all titles
words_per_title = max([get_max_words_over_titles(titles_high_raw,dictionary),get_max_words_over_titles(titles_low_raw,dictionary)])

# setup networks
model = MLPConv(words_per_title)
classifier_model = Classifier(model)

# Setup an optimizer
optimizer = optimizers.AdaGrad()  # Using Adaptive Gradient Decent
optimizer.setup(classifier_model)

# setting number of training epochs
n_epoch = 5

# setting size of validation dataset (per condition)
num_validation_titles=800

maxiter=1800

# create validation batches
test_batch_high_raw, _ = createtitlebatch(titles_high_raw, dictionary, w,skipbatches=0,numtitles=num_validation_titles)
test_batch_low_raw, _ = createtitlebatch(titles_low_raw, dictionary, w,skipbatches=0,numtitles=num_validation_titles)

# compensate for inequality in word count
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

# constructing test labels
test_labels = np.ones((len(test_batch_high), 1))
test_labels = np.concatenate((test_labels, np.zeros((len(test_batch_low), 1))), 0)

test_batch = np.concatenate((test_batch_high, test_batch_low), 0)

# preparing some start values for logging
N_test = len(test_batch)

start_timer = time.time()

acc_train = []
loss_train = []

acc_val = []
loss_val = []

# initiate epochs
for epoch in range(n_epoch):
    # initiate iterations
    # Note that due to the extraction of 800 titles beforehand, the first 10 x 80 titles are skipped during each epoch
    for iteration in range(10,maxiter):  # start with epoch 10 due to the skip for creating the validation dataset
        print('epoch' , epoch, ' - iteration ', iteration)

        # obtaining training batches
        train_batch_high, _ = createtitlebatch(titles_high_raw, dictionary,w, skipbatches=iteration)

        train_batch_low, _ = createtitlebatch(titles_low_raw, dictionary,w, skipbatches=iteration)

        # compensate for difference in word count by padding with zeros
        shape_diff=train_batch_high.shape[2]-train_batch_low.shape[2]

        if shape_diff != 0:
            padd_=np.zeros((train_batch_high.shape[0],train_batch_high.shape[1],np.abs(shape_diff),train_batch_high.shape[3]))
            if shape_diff>0:
                train_batch_low = np.concatenate((train_batch_low,padd_),2)
            elif shape_diff<0:
                train_batch_high = np.concatenate((train_batch_high, padd_), 2)

        # contruct training labels
        train_labels_high = np.ones((len(train_batch_high), 1))
        train_labels = np.concatenate((train_labels_high, np.zeros((len(train_batch_low), 1))), 0)

        train_batch = np.concatenate((train_batch_high, train_batch_low), 0)

        input = chainer.Variable(train_batch.astype('float32'))
        target = chainer.Variable(train_labels.astype('int32').ravel())

        # clear gradients of the model
        model.cleargrads()

        # produce predictions based on input data
        predictions = model(input)

        # compute loss based on those predictions
        loss = softmax_cross_entropy.softmax_cross_entropy(predictions, target)

        # compute accuracy based on those predictions
        acc = accuracy.accuracy(predictions, target)

        # log values for later plotting
        loss_train.append(float(loss.data))
        acc_train.append(float(acc.data))

        # update the model
        loss.backward()
        optimizer.update()

        print('Training current loss =', (float(loss.data)), ',Training current Accuracy =',
              (float(acc.data)))


        #### for Validation the test dataset is used without updating the network ####

        # the test set is evaluated every 25 iterations
        # explanations are corresponding to the training set
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

# saving data to .txt files

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