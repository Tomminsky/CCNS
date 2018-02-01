#!/bin/bash

sh updateStatus.sh

cp ../txtfiles/templatemail.txt ../txtfiles/lastmail.txt
cat ../txtfiles/current_status.txt >> ../txtfiles/lastmail.txt

address=t.clausner@student.ru.nl
echo sent to $address
echo Sent to $address from Raspberry Pi at $(sh getIP.sh) "\non " $(date)  >> ../txtfiles/lastmail.txt

ssmtp $address < ../txtfiles/lastmail.txt
