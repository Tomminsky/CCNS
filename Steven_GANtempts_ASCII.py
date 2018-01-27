from __future__ import print_function
import numpy as np
import chainer
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import link
from chainer import reporter
from chainer import optimizers
from chainer.dataset import iterator
import chainer.functions as F
import chainer.links as L
from chainer.datasets import TupleDataset
import matplotlib.pyplot as plt
import simplejson
import os
import random
import time


###### ASCIII IMPORT ######
# filename_titleshigh = 'titlesASCII_high.txt' # Top 25% cited titles
# filename_titleslow = 'titlesASCII_low.txt' # Lower bound 25% cited titles

filename_titleshigh = 'titlesASCII_high_DictComp.txt' # Top 25% cited titles - better quality dictionary
filename_titleslow = 'titlesASCII_low_DictComp.txt' # Lower bound 25% cited titles - better quality dictionary

######## WORDS IMPORT ###########
# filename_titleshigh = 'titlesDict_high.txt' # Top 25% cited titles - better quality dictionary
# filename_titleslow = 'titlesDict_low.txt' # Lower bound 25% cited titles - better quality dictionary

# Dictionary
filename_dict = 'dictionary.txt'
with open(filename_dict) as file:
    dictionary = []
    for line in file:
        # The rstrip method gets rid of the "\n" at the end of each line
        dictionary.append(line.rstrip())  # split(",")

### Functions 'N Classes ####
def get_title_target(namefile, t_label):
    """
        Imports the titles from file and converts its numerical [ASCII/normalized word] representation from string to float.
        Computes the corresponding target label 0 (non-popular) or 1 (popular)
    """
    with open(namefile) as file:
        lines = []
        for line in file:
            # The rstrip method gets rid of the "\n" at the end of each line
            lines.append(line.rstrip())  # split(","))

    titles = []
    count = 0
    for line in lines: # For every title
        if count == 0: # Skip first, because this is header info.
            count += 1
        else:
            line = line.split(',') # Split the mega string into a list of strings
            for i in range(len(line)):
                line[i] = np.float32(line[i]) # String to float
            titles.append(line)

    # Delete title values outside ASCII range
    titleshape = np.shape(titles)
    for t in range(titleshape[0]):
        for i in range(titleshape[1]):
            if titles[t][i] > 256:
                titles[t][i] = np.float32(32)


    if t_label ==0: # Label non-popular titles
        # targets = np.zeros((len(titles),1),dtype=np.int32)
        targets = np.zeros(len(titles), dtype=np.int32)
    if t_label ==1: # Label popular titles
        # targets = np.ones((len(titles), 1),dtype=np.int32)
        targets = targets = np.ones(len(titles), dtype=np.int32)
    return titles, targets

def ascii2title(title):
    """Traslates ascii code to titles. If there are any non-ascii noise numbers, this is converted to a space"""
    L = title
    L= np.asarray(L)
    L[L>256] = 32
    y = ''.join(chr(int(i)) for i in L)
    return y

def showrand_ASCII(rand_titles, n=5):
    """ Just for fun: Show n random titles that are ASCII encoded"""
    for i in range(n):
        u=int(np.random.randint(len(titles), size=(1)))
        print(ascii2title(rand_titles[u]))

def dict2title(title):
    """Traslates dictionary code to titles. """
    L = title
    y = ''.join(dictionary[int(i)] + " " for i in L if i > 0)
    y=y.replace('"', "")
    return y

def showrand_dict(rand_titles, n=5):
    """ Just for fun: Show n random titles that are word-dictionary encoded"""
    for i in range(n):
        u=int(np.random.randint(len(titles), size=(1)))
        print(dict2title(rand_titles[u]))

def titlehist(titles):
    """"
    Shows histogram of all ASCII values.
    Just to make sens eof the data.
    """
    duh = np.shape(titles)
    values = np.zeros(duh[0] * duh[1])
    idx=0
    for t in range(duh[0]):
        for i in range(duh[1]):

            values[idx] = titles[t][i]
            idx+=1

    plt.hist(values,bins=100, color='blue'); plt.xlabel("ASCII #"); plt.ylabel("Occurance"); plt.title("Histogram of ASCII numbers");

def norm_title(titles, asciimax=255.0):
    """"
    Normalize titles between 0 and 1
    """
    shapetitles=np.shape(titles)
    for t in range(shapetitles[0]):
        for i in range(shapetitles[1]):
            if titles[t][i] > asciimax: # If character exceeds max allowed character
                titles[t][i] = np.float32(32)
            titles[t][i] = np.float32(titles[t][i] / asciimax)

    return titles

def unnorm_title(titles, asciimax=255.0):
    """"
    Unormalize titles from between 0 and 1 to 0 and asciimax
    """
    shapetitles=np.shape(titles)
    for t in range(shapetitles[0]):
        for i in range(shapetitles[1]):
            titles[t][i] = np.float32(round(titles[t][i] * asciimax))

    return titles

def gentitle(generator, normalize = 0, printoff=0):
    """
    Makes the generator generate one title.
    Used as follow: gentitle(generator, normalize=normalize)
    """
    gen_batch = generator(Gnoise(1))
    if normalize == 1:  # If title normalized, conert back to ASCII style for show-off
        gen_title = ascii2title(
            unnorm_title(np.atleast_2d(gen_batch._data[0]), asciimax=maxscii)[0][:])
    else:
        gen_title = ascii2title(gen_batch._data[0])  # Take random generated title


    if printoff:
        return gen_title
    else:
        print(repr(gen_title))

####### MODELS ########
# Discriminators
class MLP(chainer.Chain):
    """MLP from exercise 1 for testing"""

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__(
            # No need for input number, it can infer this.
            l1 = L.Linear(None, n_units), # Input to layer 1
            l2 = L.Linear(None, n_out), # Layer out
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        y = self.l2(h1)
        return y

class Discriminator(chainer.Chain):
    """Multilayered Perceptron with 1 hidden layer. It takes n_units as input
        , which is the number of hidden layer units. In addition, it takes n_out as input, specifying output unit of
        last layer"""

    def __init__(self, n_units, n_out):
        super(Discriminator, self).__init__(
            # No need for input number, it can infer this.
            l1 = L.Linear(None, n_units), # Layer 1
            l2 = L.Linear(n_units, n_units), # Layer 2
            l3 = L.Linear(n_units, n_out), # Layer 3; out
        )

    def __call__(self, x):
        h1 = F.elu(self.l1(x))
        h2 = F.elu(self.l2(h1))
        y = F.sigmoid(self.l3(h2))
        return y

class MLP2(chainer.Chain):
    """MLP from exercise 1 for testing"""

    def __init__(self, n_out):
        super(MLP2, self).__init__(
            # No need for input number, it can infer this.
            l1 = L.Linear(None, 194), # Input to layer 1
            l2 = L.Linear(None, 200), # Layer out
            l3 = L.Linear(None, 120),
            l4 = L.Linear(None, 60),
            l5 = L.Linear(None, 20),
            l6 = L.Linear(None,n_out),
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.sigmoid(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = F.sigmoid(self.l4(h3))
        h5 = F.relu(self.l5(h4))
        y = F.sigmoid(self.l6(h5))
        return y

# Generators
class Generator(chainer.Chain):
    """Multilayered Perceptron with 1 hidden layer. It takes n_units as input
        , which is the number of hidden layer units. In addition, it takes n_out as input, specifying output unit of
        last layer"""

    def __init__(self, n_out):
        super(Generator, self).__init__(
            # No need for input number, it can infer this.
            l1 = L.Linear(None, n_out * 3), # Layer 1
            l2 = L.Linear(None, n_out * 2), # Layer 2
            l3 = L.Linear(None, n_out), # Layer 3 out
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = F.sigmoid(self.l3(h2))
        return y


class Generator2(chainer.Chain):
    """Multilayered Perceptron with 1 hidden layer. It takes n_units as input
        , which is the number of hidden layer units. In addition, it takes n_out as input, specifying output unit of
        last layer"""

    def __init__(self, n_out):
        super(Generator2, self).__init__(
            # No need for input number, it can infer this.
            l1 = L.Linear(None, n_out * 10), # Layer 1
            l2 = L.Linear(None, n_out * 8), # Layer 2
            l3 = L.Linear(None, n_out * 6),
            l4 = L.Linear(None, n_out * 4),
            l5 = L.Linear(None, n_out * 2),
            l6 = L.Linear(None, n_out), # Last layer
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.sigmoid(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = F.sigmoid(self.l4(h3))
        h5 = F.relu(self.l5(h4))
        y = F.sigmoid(self.l6(h5))
        return y

####### EXTRA USED FOR NETWORK ####
def to_tuple(x):
    if hasattr(x, '__getitem__'):
        return x
    return x,

class UniformNoiseGenerator(object):
    def __init__(self, low, high, size):
        self.low = low
        self.high = high
        self.size = to_tuple(size)

    def __call__(self, batch_size):
        return np.random.uniform(self.low, self.high, (batch_size,) +
                                    self.size).astype(np.float32)

class BatchIter(object):
    """
    Generates random mega set of data and returns next batch when called.
    """

    def __init__(self, data, batch_size=1, iters=1):
        self.data = data
        self.original_data_size =np.shape(titles)[0]
        self.idx = 0
        self.batch_size = batch_size
        self.max_iter=iters
        self.bigdata=[]
        self.range = range(0, self.batch_size * self.max_iter, self.batch_size)
        self.batch = None

        # Create a mega dataset of minimal size batch_size*iter: we can iterate over this
        self.goal_data_size = self.max_iter * self.batch_size
        self.times_enlarge = (self.goal_data_size // self.original_data_size) + (self.goal_data_size % self.original_data_size >0)

        # The loop ensures all data is used and not potentially lost in randomization if iter is large
        counter=0
        while counter < (self.times_enlarge+1):
            rand_ind = np.random.permutation(self.original_data_size)
            rand_titles = [self.data[i] for i in rand_ind]
            self.bigdata += rand_titles
            counter+=1

    def __call__(self):

        self.batch = self.bigdata[self.range[self.idx]: self.range[self.idx]+self.batch_size]
        self.idx +=1
        return np.asarray(self.batch)

def update_model(opt, loss):
    """"
    Updates the model of the input optimizer using the input loss.
    Makes the code shorter and clear.
    """
    opt.target.cleargrads()
    loss.backward()
    opt.update()


# Dataset
titles, targets  = get_title_target(filename_titleshigh, 1) # Get popular titles only

# Set if normalize titles or not
normalize = 1 # 1 is yes normalize
maxscii = 122.0 # Could also be 255, just tryin'
if normalize == 1:
    titles = norm_title(titles,asciimax=maxscii)


# Create noise input for generator
titlelength = np.shape(titles)[1]
if normalize:
    Gnoise = UniformNoiseGenerator(0,1,titlelength) # This creates uniform noise of size titlelength. When called do Gnoise(batchsize)
else:
    Gnoise = UniformNoiseGenerator(0, maxscii, titlelength)

# Parameters
N = np.shape(titles)[0] # Number of titles
batchsize = 30
epochsize = 1000 # One epoch is i iterations. It is an artificial epoch in order to create only one for loop
N_iter = 30000 # Number of iterations. One iteration goes over one batch

# Set up models
n_out = 1 # Number of units out of discriminator: must be 1 because of probability output
discriminator = MLP2(n_out)
# generator = Generator(titlelength) # Generates samples of size titlelength
generator = Generator2(titlelength)

# Set up optimizers for models
opt_discriminator = optimizers.MomentumSGD(lr=0.001,momentum=0.9)
opt_generator = optimizers.MomentumSGD(lr=0.001,momentum=0.9)

# opt_discriminator = optimizers.RMSprop(lr=0.001)
# opt_generator = optimizers.RMSprop(lr=0.001)

# opt_discriminator = optimizers.AdaDelta()
# opt_generator =optimizers.AdaDelta()


opt_discriminator.setup(discriminator)
opt_generator.setup(generator)

# Batch iterator. Only need it once: it creates enough batches for N_iterations. The iterator automatically goes to the next batch when called.
real_batch = BatchIter(titles, batch_size=batchsize, iters=N_iter) # Batches for real titles


# Loop specifics
epochs = range(1000, N_iter, epochsize)
epoch_i = 0 # Counter for epoch
N_epoch = len(epochs)

total_Dloss = np.zeros(N_iter)
total_Gloss = np.zeros(N_iter)
total_Dacc = np.zeros(N_iter)

epoch_Dloss = np.zeros(N_epoch); epoch_Dloss[:]=np.nan # isnan because N_iter might not scale to epochs so there could be one extra index.
epoch_Gloss = np.zeros(N_epoch); epoch_Gloss[:]=np.nan
epoch_Dacc = np.zeros(N_epoch); epoch_Dacc[:]=np.nan

start_time = time.time()
####### GAN #######
for i in range(N_iter): # For N_iter do the training
    # Generated images for discriminator
    gen_batch = generator(Gnoise(batchsize))

    # Update discriminator
    Dreal = discriminator(real_batch())
    Dgen = discriminator (gen_batch)

    Dloss = 0.5 * (F.sum((Dreal - 1.0)**2) + F.sum(Dgen**2)) / batchsize # Loss for discriminator.
    update_model(opt_discriminator, Dloss)

    Dacc = np.sum(np.int32(Dgen[:,0].data<0.5)+np.int32(Dreal[:,0].data>0.5)) / (batchsize*2.0) # Accuracy in prediction of fake/real


    # Update generator
    gen_batch = generator(Gnoise(batchsize))
    Gloss = 0.5 * F.sum((discriminator(gen_batch) - 1.0) ** 2) / batchsize # Loss for generator
    update_model(opt_generator, Gloss)


    # Store all the values for later plots
    total_Dloss[i] = Dloss.data
    total_Gloss[i] = Gloss.data
    total_Dacc[i] = Dacc

    # Epoch
    if i in epochs: # Every artifical epoch show some parameters
        gen_title = gentitle(generator, normalize=normalize, printoff=1)

        epoch_Dloss[epoch_i] = np.sum(total_Dloss[i-epochsize:i])
        epoch_Gloss[epoch_i] = np.sum(total_Gloss[i-epochsize:i])
        epoch_Dacc[epoch_i] = np.sum(total_Dacc[i-epochsize:i]) / epochsize # Mean accuracy per epoch

        # Time progress
        timespent = time.time() - start_time
        percent_progress = float(i)/N_iter
        time_left = (timespent / percent_progress)  - timespent

        # Print stuff
        print("Epoch ", str(epoch_i+1))
        print("* Progress:", str(percent_progress*100) + "%", "|", "Time spent:", str(timespent/60) + "minutes","|", "Estimated time left:", str(time_left/60)+ "minutes")
        print("* Discriminator accuracy:", epoch_Dacc[epoch_i],"|","Discriminator loss:", str(epoch_Dloss[epoch_i]),
              "|", "Generator loss:", str(epoch_Gloss[epoch_i]))
        print("* Random generated title: \n ", repr(gen_title), "\n")

        #end epoch
        epoch_i += 1




######### PLOTS 'n WOTS ##########
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
plt.subplot(211)
plt.plot(epoch_Gloss, color=[0.2,0.4,0.8])
plt.ylabel('Loss Generator', color=[0.2,0.2,0.8])

ax2 = ax1.twinx()
ax2.plot(epoch_Dloss, color=[0.8,0.2,0.2])
ax2.set_ylabel('Loss Discriminator', color=[0.8,0.2,0.2])
plt.show()

plt.subplot(212)
plt.plot(epoch_Dacc, color=[0.8,0.2,0.2], linestyle= "--", linewidth=2.0)
plt.ylabel('Accuracy Discriminator', color=[0.8,0.2,0.2])
plt.xlabel("Epoch")

ax2.set_title("GAN fight")
f.tight_layout()



