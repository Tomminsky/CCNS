#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
currnum=$(echo $(($(ls ../txtfiles/allcrawls_part* | sed -e s/[^0-9]//g | tail -1) +1)) | bc)
scp -r $DIR/../txtfiles/allcrawls.txt $DIR/../txtfiles/allcrawls_part$currnum.txt
sh sendData.sh
sleep 15s
rm $DIR/../txtfiles/allcrawls.txt
touch $DIR/../txtfiles/allcrawls.txt
