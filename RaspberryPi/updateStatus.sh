#!/bin/bash
curr_size="$(sh reportcurrentsetsize.sh)"
echo $(date)current size of dataset: $curr_size >> ../txtfiles/current_status.txt
echo current IP "$(sh getIP.sh)\n">> ../txtfiles/current_status.txt
