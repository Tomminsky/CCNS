#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#removes white space lines and puts all parts together
awk 'NF' $DIR/allcrawls_part*.txt > $DIR/allcrawlspreproc.txt
