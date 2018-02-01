#!/bin/bash
if pidof -x "scholarcrawler.sh" >/dev/null
then
sudo killall scholarcrawler.sh
fi
~/CCNS_project/scripts/scholarcrawler.sh 0 10 ~/CCNS_project/txtfiles/search_items.txt
