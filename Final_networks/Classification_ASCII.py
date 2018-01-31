from __future__ import print_function
import numpy as np
import chainer
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import optimizers
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
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
    """Translates dictionary code to titles. """
    L = title
    y = ''.join(dictionary[int(i)] + " " for i in L if i > 0)
    y=y.replace('"', "")
    return y

def showrand_dict(rand_titles, n=5):
    """ Just for fun: Show n random titles that are word-dictionary encoded"""
    for i in range(n):
        u=int(np.random.randint(len(titles), size=(1)))
        print(dict2title(rand_titles[u]))

class MLP(chainer.Chain):
    """MLP for binary classification"""

    def __init__(self, n_out):
        super(MLP, self).__init__(
            # No need for input number, it can infer this.
            l1 = L.Linear(None, 194), # Input to layer 1
            l2 = L.Linear(None, 1000), # Layer out
            l3 = L.Linear(None, 1000),
            l4 = L.Linear(None, 500),
            l5 = L.Linear(None, 500),
            l6 = L.Linear(None, 250),
            l7 = L.Linear(None, 250),
            l8 = L.Linear(None, 200),
            l9 = L.Linear(None, 150),
            l10 = L.Linear(None, 100),
            l11 = L.Linear(None, 50),
            l12 = L.Linear(None, 25),
            l13 = L.Linear(None,n_out), # Layer out
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.tanh(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = F.tanh(self.l4(h3))
        h5 = F.relu(self.l5(h4))
        h6 = F.tanh(self.l6(h5))
        h7 = F.relu(self.l7(h6))
        h8 = F.tanh(self.l8(h7))
        h9 = F.relu(self.l9(h8))
        h10 = F.tanh(self.l10(h9))
        h11 = F.relu(self.l11(h10))
        h12 = F.tanh(self.l12(h11))
        y = F.softmax(self.l13(h12))
        return y

# Pre define dataset
titles_high, targets_high = get_title_target(filename_titleshigh, 1)
titles_low, targets_low = get_title_target(filename_titleslow, 0)


titles = titles_low + titles_high
targets = np.concatenate((targets_low, targets_high))

# Randomizing for mini-batches
rand_ind = np.random.permutation(np.shape(targets)[0])
rand_targets = targets[rand_ind]
rand_titles = [titles[i] for i in rand_ind]

# Split into training and test batches
percent_test = 0.1  # 10% of our data is test data
N_split = int(len(rand_titles) * percent_test)
titles_test = rand_titles[0:N_split]; targets_test = rand_targets[0:N_split]  # Tests
titles_train = rand_titles[N_split:]; targets_train = rand_targets[N_split:]  # Trains


###################MLP Part######################

batchsize = 30
N_train = batchsize*5000 # Can max max len(titles_train)

epochs = 101
n_out = 2 # The number of classes out. If 2 then it's similar to how it was done for MNIST. If 1, then it's a popularity probabiliy.

# Network selection
model = MLP(n_out)

# optimizer = optimizers.SGD()  # Using Stochastic Gradient Descent
# optimizer = optimizers.MomentumSGD(lr=0.001,momentum=0.9)
# optimizer = optimizers.NesterovAG(lr=0.0001,momentum=0.9)
optimizer = optimizers.AdaGrad(lr=0.001)
optimizer.setup(model)



# MLP storage
total_loss = np.zeros(epochs)
total_acc = np.zeros(epochs)
loss_batches = np.zeros(epochs*N_train/batchsize); bcount=0
acc_batches = np.zeros(epochs*N_train/batchsize)

test_loss = np.zeros(epochs)
test_acc = np.zeros(epochs)

# MLP start
time_start = time.time()
for epoch in range(epochs):
    # Shuffle training dataset
    rand_ind = np.random.permutation(np.shape(targets_train)[0])
    targets_train = targets_train[rand_ind]
    titles_train_tmp = [titles_train[i] for i in rand_ind]
    titles_train = titles_train_tmp

    ###### Training loop
    for i in range(0, N_train, batchsize):
        input = chainer.Variable(np.asarray(titles_train[i: i+batchsize]))
        target = targets_train[i: i+batchsize]

        model.cleargrads()
        predictions = model(input)

        if n_out == 1:
            loss = F.sigmoid_cross_entropy(predictions, np.atleast_2d(target).T) #Sigmoid for two class prediction (i.e.  one number output)
        if n_out > 1:
            loss = softmax_cross_entropy.softmax_cross_entropy(predictions,target) # For multi class predictions

        loss_batches[bcount] = loss.data
        total_loss[epoch] += loss.data
        loss.backward()
        acc = accuracy.accuracy(predictions, target)
        acc_batches[bcount]= acc.data; bcount+=1
        total_acc[epoch] += acc.data

        optimizer.update()
    print("Epoch: ", str(epoch+1), "| Training accuracy: ", str(total_acc[epoch] / (N_train/batchsize)) + "%", "| Time elapsed: ", str((time.time()-time_start)/60)+ " minutes | actual loss: ",str(total_loss[epoch]))


    ###### Testing: Doesn't really need a loop as model doesn't update here anyway. It is order independent.
    model.cleargrads()
    predictions = model(chainer.Variable(np.asarray(titles_test)))

    acc = accuracy.accuracy(predictions, targets_test)
    if n_out == 1:
        loss = F.sigmoid_cross_entropy(predictions, np.atleast_2d(
            targets_test).T)  # Sigmoid for two class prediction (i.e.  one number output)
    if n_out > 1:
        loss = softmax_cross_entropy.softmax_cross_entropy(predictions, targets_test)  # For multi class predictions

    test_loss[epoch] = loss.data
    test_acc[epoch] = acc.data
    print("Epoch: ", str(epoch+1), "| Testing accuracy: ", str(test_acc[epoch]) + "%", "| Time elapsed: ", str((time.time()-time_start)/60)+ " minutes | actual loss: ",str(test_loss[epoch]))




##### Plot desciptive statistics #########
fig, ax1 = plt.subplots()
plt.plot(total_acc/(N_train/batchsize),color=[0.8,0.2,0.2], label="Train")
plt.plot(test_acc, color=[0.2,0.4,0.8], label="Test")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
# fig.tight_layout()
plt.show()


plt.savefig("RESULTS_ASCII_Accuracy.eps", format='eps', dpi=1000)