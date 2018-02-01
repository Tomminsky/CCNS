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

# load and arrange the dictionary file
def get_dictionary(namefile):
    print('loading '+ namefile + '...')
    # all " strings are removed and the data is split at each line break
    with open(namefile) as f:
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
    lines=np.array(map(int, lines[:-1].split(',')))
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
    w[word2index[' ']]=np.zeros((1, 200))
    print('done.')
    # word2index transforms a dictionary index to a human readable word
    # index2word transforms a human readable word to a dictionary index
    return word2index,index2word,w

# this function is used to obtain the maximum number of words across all titles by finding <eol> statements
def get_max_words_over_titles(titles_raw,dictionary):
    endoftitles = [x for x in range(len(titles_raw)) if titles_raw[x] == dictionary.get('<eol>')]
    startoftitles = [0] + list(np.add(endoftitles[:-1], 1))
    max_title_length_in_batch = max(np.abs(np.subtract(startoftitles, endoftitles))) + 1
    return max_title_length_in_batch

# this function creates batch data used to train the network
def createtitlebatch(titles_raw,dictionary,skipbatches=0,numtitles=10,testpart=0.05):

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
        title_matrix=[]
        # extraction of the data from w given the amount of titles selected
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

        return np.asarray(train).astype('float32').reshape((len(train),max_title_length_in_batch/7,max_title_length_in_batch/7,200)),np.asarray(test).astype('float32').reshape((len(test),1,max_title_length_in_batch,200))

# this function can be used to transform a title matrix (words x ndim) back into a human readable title
# the argmin() L2 norm between each word vector of the title and all word vectors in w serves as the
# index for the respective word in the dictionary
def vec2title(vec_,w,index2word):

    dict_trans=w#(w-w.min())/(w-w.min()).max()
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

# Convolutional neuronal network to do the discrimination
# respective architecture choices are explained in the report
class MLPConv(chainer.Chain):
    def __init__(self,words_per_title):
        super(MLPConv, self).__init__()
        with self.init_scope():
            self.words_per_title = words_per_title
            self.conv = L.Convolution2D(in_channels=1, out_channels=1, ksize=3)
            self.l2 = L.Linear(None, 2)

    def __call__(self, x):
        x2 = F.relu(self.conv(F.reshape(x,(x.data.shape[0], 1,self.words_per_title ,200))))
        x3 = F.max_pooling_2d(x2, 3)
        y = F.sigmoid(self.l2(F.dropout(x3,0.2)))
        return y

# Deconvolutional neuronal network to do the generation
# respective architecture choices are explained in the report
class generator(chainer.Chain):
    def __init__(self, words_per_title):
        super(generator, self).__init__()
        with self.init_scope():
            self.words_per_title = words_per_title
            self.l1 = L.Linear(None, words_per_title*200)  # linear input layer
            self.l2 = L.Deconvolution2D(in_channels=1, out_channels=1, ksize=3)  # applying deconvolution
            self.l3 = L.Linear(None, words_per_title * 200)  # linear input layer

    def __call__(self, x):
        h = F.relu(self.l1(x))  # rectified activation function
        h = F.reshape(h, (x.data.shape[0], 1,self.words_per_title,200))
        h = F.relu(self.l2(h))
        return F.reshape(self.l3(h),(x.data.shape[0], 1, self.words_per_title, 200))

# loading the respective data
word2index,index2word,w=get_embeddedwords()

dictionary=get_dictionary('dictionary.txt')

titles_high_raw=get_titles('titlesDict_high.txt',dictionary,shuffle=1)

# get maximum number of words in all titles
words_per_title = get_max_words_over_titles(titles_high_raw,dictionary)

# setup networks
dis = MLPConv(words_per_title)
gen = generator(words_per_title)

# Setup an optimizer
opti_gen = optimizers.MomentumSGD(lr=0.01) # Using Stochastic Gradient Decent employing momentum
opti_gen.setup(gen)

opti_dis = optimizers.MomentumSGD(lr=0.001) # Using Stochastic Gradient Decent employing momentum
opti_dis.setup(dis)

# due to hardware limitations only one training epoch is used
n_epoch = 1

maxiter=14400

# preparing some start values for logging
start_timer = time.time()

overall_acc = []
overall_loss_dis = []
overall_loss_gen = []
title_creations = []

# initiate epochs
for epoch in range(n_epoch):
    for iteration in range(maxiter):  # initiate iterations
        print('epoch' , epoch, ' - iteration ', iteration)  # prompting the word 'epoch ' and the coresponding training epoch to the Python Consol

        # obtaining original training titles
        train_batch_orig, _ = createtitlebatch(titles_high_raw, dictionary, skipbatches=iteration,testpart=0.1)

        # creating random value distribution for title generation
        train_batch_gen = chainer.Variable(np.random.uniform(w.min(), w.max(),(len(train_batch_orig),words_per_title*200) ).astype('float32'))

        # obtaining generated training titles
        train_batch_gen = gen(train_batch_gen)

        # evaluate created titles
        judge_fake = dis(train_batch_gen)

        # compute loss for the respective generated titles
        Loss_gen = F.softmax_cross_entropy(judge_fake, chainer.Variable(
            np.zeros(len(train_batch_orig), dtype=np.int32)))  # obtain generator loss

        # evaluate real titles
        judge_real = dis(chainer.Variable(train_batch_orig))

        # compute combined loss for the respective generated titles and real titles
        Loss_dis = F.softmax_cross_entropy(judge_fake, chainer.Variable(np.ones(len(train_batch_orig), dtype=np.int32))) + \
                   F.softmax_cross_entropy(judge_real, chainer.Variable(np.zeros(len(train_batch_orig), dtype=np.int32)))

        # compute discriminator accuracy
        acc = (sum(list(np.int32(judge_fake[:, 0].data < 0.5)) + list(np.int32(judge_real[:, 0].data > 0.5))) / (len(train_batch_orig) * 2.))

        # log values for later plotting
        overall_acc.append(acc)
        overall_loss_dis.append(float(Loss_dis.data))
        overall_loss_gen.append(float(Loss_gen.data))

        # update discriminator
        dis.cleargrads()
        Loss_dis.backward()
        opti_dis.update()

        # update generator
        gen.cleargrads()
        Loss_gen.backward()
        opti_gen.update()


        print('Training current loss Discriminator =', (float(Loss_dis.data)),
              ', Training current loss Generator =', (float(Loss_gen.data)), ',Training current Accuracy =',
              (acc),'Fake detected = ',np.float32(judge_fake[:, 0].data < 0.5).sum()/ len(train_batch_orig))

        # after every 10th training iteration an example title is plotted
        if iteration%10==0:
            print(vec2title(train_batch_gen[np.random.randint(len(train_batch_gen))][0], w, index2word))
            title_creations.append(vec2title(train_batch_gen[np.random.randint(len(train_batch_gen))][0], w, index2word))

# saving data to .txt files
with open('Accuracy_trainGAN.txt', 'w') as file_handler:
    for item in overall_acc:
        file_handler.write("{}\n".format(item))

with open('Loss_DisGAN.txt', 'w') as file_handler:
    for item in overall_loss_dis:
        file_handler.write("{}\n".format(item))

with open('Loss_GenGAN.txt', 'w') as file_handler:
    for item in overall_loss_gen:
        file_handler.write("{}\n".format(item))

with open('Titles_GAN.txt', 'w') as file_handler:
    for item in title_creations:
        file_handler.write("{}\n".format(item))