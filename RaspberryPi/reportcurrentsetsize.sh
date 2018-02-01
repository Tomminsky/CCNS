#!/bin/bash
echo $(cat ../txtfiles/allcrawls.txt | wc -l)/2 | bc -s
