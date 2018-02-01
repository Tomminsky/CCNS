#!/bin/bash

# wrapper function for doing the word embedding
# the title file is given as first and the dictionary as second argument

/Users/Tommy/anaconda/bin/python word_embedding.py -u 200 -e 20 -it $1 -id $2
