#!/bin/bash
sudo killall ikec
sudo killall iked
sleep 15s
sudo iked
sleep 15s
curr_date=$(date)
echo "##### NEW SESSION $curr_date #####" >> ../txtfiles/current_status.txt
old_ip=$(sh getIP.sh)
echo "\nold IP: $old_ip\n" >>  ../txtfiles/current_status.txt
sh connectVPN.sh  >>  ../txtfiles/current_status.txt &
sleep 15s
new_ip=$(sh getIP.sh)
echo "\nnew IP: $new_ip\n" >>  ../txtfiles/current_status.txt
