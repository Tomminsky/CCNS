# CCNS

## preprocessdata.sh
merges parts of data (that otherwise would be to big to be transferred via email) and removes white space

result: allcrawspreproc.txt

## preprocessdata.m
takes allcrawspreproc.txt and performs the respective analysis

- ASCII conversion + saving
- Dictionary conversion + saving
- ASCII re-conversion based on titles used to build up the dictionary version + saving

result: all kinds of .mat and .txt files (contained by titlesASCII_all.zip and titlesASCII_split_and_citations.zip)

### copyright:
-------
the function /scriptsPi/scholarcrawler.py was adapted from ckreibich (https://github.com/ckreibich/scholar.py) according to the standard [BSD license](http://opensource.org/licenses/BSD-2-Clause)

the function word_embedding.py was adapted from the chainer word2vec example (https://github.com/chainer/chainer/tree/master/examples/word2vec)

### License:
-------

all contents within this repository are using the standard [BSD license](http://opensource.org/licenses/BSD-2-Clause).
