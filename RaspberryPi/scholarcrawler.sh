#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIR=$DIR/../txtfiles

logfile=$DIR/crawler.log
exec > $logfile 2>&1

sh $DIR/../scripts/killalljobs.sh
sh $DIR/../scripts/autoreport.sh &

cat $3 | while read CMD
do
echo $(date) >> $DIR/timestamplastIteration.txt
rm "$DIR/currcrawl.txt"
touch "$DIR/currcrawl.txt"
olddata=""
timeoutind=0

for i in `seq $1 $2`; do
searchvar="$CMD neuro"
echo $searchvar

if [ "$i" -eq 5 ]
then
sh $DIR/../scripts/updateStatus.sh &
echo date current search variable $searchvar >> $DIR/../txtfiles/current_status.txt
echo "$(sed '/$CMD/d' $DIR/search_items.txt)">$DIR/tmp.txt
mv $DIR/tmp.txt $DIR/search_items.txt
fi

newdata=$(echo "$(python "$DIR/../scripts/scholarcrawler.py" -c $i -s $searchvar --no-patents | grep -e "Citations [0-9]" -e "Title" | awk '{$1=""; print $0}')")
echo "$newdata"  >> "$DIR/currcrawl.txt"
if [ "$olddata" == "$newdata" ]
then
timeoutind=1
echo $(date) not crawling >> $DIR/../txtfiles/current_status.txt
break
fi
olddata=$newdata
r=$(( $RANDOM % 300 ))
r=$(echo $r/100+0.01 | bc -l)
sleep $r
done
cat $DIR/currcrawl.txt >> $DIR/allcrawls.txt
if [ "$timeoutind" -eq 1 ]
then
r=$(( $RANDOM % 1800 ))
r=$(echo $r+60 | bc -s)
echo "timeout for $r seconds"
echo $(date) timeout detected... waiting $r seconds before reconnecting
sleep 10s
sh $DIR/../scripts/reconfDNS.sh &
sh $DIR/../scripts/reloginscript.sh &
sleep 300s
timeoutind=0
echo $(date) start new crawling attempt >> $DIR/current_status.txt
sh $DIR/../scripts/updateStatus.sh &
fi
done
