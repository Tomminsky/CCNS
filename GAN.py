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
    w[word2index[' ']]=np.zeros((1, 200))
    print('done.')
    return word2index,index2word,w

def get_max_words_over_titles(titles_raw,dictionary):
    endoftitles = [x for x in range(len(titles_raw)) if titles_raw[x] == dictionary.get('<eol>')]
    startoftitles = [0] + list(np.add(endoftitles[:-1], 1))
    max_title_length_in_batch = max(np.abs(np.subtract(startoftitles, endoftitles))) + 1
    return max_title_length_in_batch

def createtitlebatch(titles_raw,dictionary,skipbatches=0,numtitles=80,testpart=0.1,shuffleinit=0):

    skip_=numtitles*skipbatches

    endoftitles = [x for x in range(len(titles_raw)) if titles_raw[x] == dictionary.get('<eol>')]
    startoftitles = [0] + list(np.add(endoftitles[:-1], 1))

    if shuffleinit:
        idx = np.random.permutation(len(endoftitles))

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

        return np.asarray(train).astype('float32').reshape((len(train),1,max_title_length_in_batch,200)),np.asarray(test).astype('float32').reshape((len(test),1,max_title_length_in_batch,200))

def vec2title(vec_,w,index2word):

    dict_trans=(w-w.min())/[(w-w.min()).max()]*2-1
    title_recon=''
    for i in range(len(vec_)):
        word_ = vec_.data[i]
        word_ = np.tile(word_,(len(w),1))
        dist_=np.sqrt(np.sum((dict_trans-word_)**2,1))
        title_recon=title_recon+index2word[dist_.argmin()-1]+' '

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
    def __init__(self):
        super(MLPConv, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(in_channels=1, out_channels=5, ksize=5,stride=1,pad=0)
            self.l1 = L.Linear(None, 50)
            self.l2 = L.Linear(None, 2)

    def __call__(self, x):
        x2 = F.relu(self.conv(x))
        x3 = F.max_pooling_2d(x2, 3)
        y = self.l2(F.dropout(F.relu(self.l1(x3))))
        return y

class generator(chainer.Chain):
    def __init__(self, words_per_title):
        super(generator, self).__init__()
        with self.init_scope():
            self.words_per_title = words_per_title
            self.l1 = L.Linear(None, words_per_title*200)  # linear input layer
            self.l2 = L.BatchNormalization(words_per_title*200)  # applying batch normalization
            self.l3 = L.Deconvolution2D(None, 1, 3, pad=1)  # applying deconvolution

    def __call__(self, x):
        h = F.relu(self.l1(x))  # rectified activation function
        h = self.l2(h)
        h = F.reshape(h, (x.data.shape[0], 1, self.words_per_title, 200))
        h = F.sigmoid(self.l3(h))  # sigmoid activation function
        return h

word2index,index2word,w=get_embeddedwords()

dictionary=get_dictionary('dictionary.txt')

titles_high_raw=get_titles('titlesDict_high.txt',dictionary)

epoch = 20 # for the convolutionary network 50 training epochs are used
words_per_title = get_max_words_over_titles(titles_high_raw,dictionary)
dis = MLPConv()
gen = generator(words_per_title)

opti_gen = optimizers.AdaGrad(lr=0.001)
opti_gen.setup(gen)

opti_dis = optimizers.AdaDelta()
opti_dis.setup(dis)

n_epoch = epoch

accplot_train = np.zeros((n_epoch, 1), dtype=float) # Store train accuracy for plot
lossplot_train = np.zeros((n_epoch, 1), dtype=float)  # Store train loss for plot

epocheloss=[]

num_validation_titles=800

test_batch_raw, _ = createtitlebatch(titles_high_raw, dictionary, skipbatches=0,numtitles=num_validation_titles)

maxiter=1500 # maximum is [number of titles -1600 (2 x 800 for both groups that are used for validation)]/ 80 (default number of titles in one batch) - 80



accplot = np.zeros((maxiter, 1), dtype=float)  # Store  test accuracy for plot
lossplot = np.zeros((maxiter, 1), dtype=float)  # Store test loss for plot
start_timer = time.time()
for epoch in range(n_epoch):
    sum_accuracy_train = 0  # Creating a staring variable
    sum_loss_train_dis = 0
    sum_loss_train_gen = 0
    for iteration in range(10,maxiter):  # start with epoch 1 (instead of 0)
        print('epoch' , epoch, ' - iteration ', iteration)  # prompting the word 'epoch ' and the coresponding training epoch to the Python Consol

        # training the MLP with the last chainer method from guide; no cleargrads()!

        train_batch, _ = createtitlebatch(titles_high_raw, dictionary, skipbatches=iteration,testpart=0.1)

        train_batch2 = chainer.Variable(np.random.uniform(-1, 1,(len(train_batch),words_per_title*200) ).astype('float32'))

        train_batch2 = gen(train_batch2)

        judge_fake = dis(train_batch2)  # classify generated examples

        Loss_gen = F.softmax_cross_entropy(judge_fake, chainer.Variable(
            np.zeros(len(train_batch), dtype=np.int32)))  # obtain generator loss

        judge_real = dis(train_batch)
        # obtain discriminator loss
        Loss_dis = F.softmax_cross_entropy(judge_fake, chainer.Variable(np.ones(len(train_batch), dtype=np.int32))) + \
                   F.softmax_cross_entropy(judge_real, chainer.Variable(np.zeros(len(train_batch), dtype=np.int32)))

        # update generator
        gen.cleargrads()
        Loss_gen.backward()
        opti_gen.update()

        # update discriminator
        dis.cleargrads()
        Loss_dis.backward()
        opti_dis.update()

        sum_loss_train_dis += float(Loss_dis.data) * len(train_batch)  # Times length of current batch for relative impact
        sum_loss_train_gen += float(Loss_gen.data) * len(train_batch)
        sum_accuracy_train += np.sum(np.int32(judge_fake[:, 0].data < 0) + np.int32(judge_real[:, 0].data > 0)) / (len(train_batch) * 2.)

        print('Training mean loss Discriminator =', (sum_loss_train_dis / len(train_batch)),
              ', Training mean loss Generator =', (sum_loss_train_gen / len(train_batch)), ',Training Accuracy =',
              (sum_accuracy_train / iteration))  # To check values during process.

    print(vec2title(train_batch2[0][0],w,index2word))


