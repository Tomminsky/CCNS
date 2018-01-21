#!/bin/bash
mkdir tmp
cd tmp
zip -r titlesASCII.zip ../titlesASCII*.txt
zip -r citations.zip ../citations*.txt
zip -r titlesDict.zip ../*titlesDict*.txt ../dictionary.txt
zip -r rawData.zip ../allcrawls_part*.txt ../allcrawlspreproc.txt

scp -r * /Users/Tommy/Github_repositories/CCNS/
scp -r ../preprocessdata.m /Users/Tommy/Github_repositories/CCNS/preprocessdata.m
scp -r ../preprocessdata.sh /Users/Tommy/Github_repositories/CCNS/preprocessdata.sh
scp -r ../updateGitHub.sh  /Users/Tommy/Github_repositories/CCNS/updateGitHub.sh
