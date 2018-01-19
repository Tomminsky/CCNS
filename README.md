# CCNS

## preprocessdata.sh
merges parts of data (that otherwise would be to big to be transferred via email) and removes white space

result: allcrawspreproc.txt

## preprocessdata.m
takes allcrawspreproc.txt and performs the respective analysis

- ASCII conversion + saving
- Dictionary conversion + saving (not correct yet)

result: all kinds of .mat and .txt files (contained by titlesASCII_all.zip and titlesASCII_split_and_citations.zip)
