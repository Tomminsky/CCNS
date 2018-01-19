#!/bin/bash

if pidof -x "scholarcrawler.sh" >/dev/null
then
echo $(date) crawler online >> ~/CCNS_project/txtfiles/current_status.txt
else
~/CCNS_project/scripts/goCrawl.sh
echo $(date) try to boot crawler... >> ~/CCNS_project/txtfiles/current_status.txt

fi

if pidof -x "ikec" >/dev/null
then
echo $(date) VPN online >> ~/CCNS_project/txtfiles/current_status.txt
else
echo $(date) VPN offline >> ~/CCNS_project/txtfiles/current_status.txt

fi
