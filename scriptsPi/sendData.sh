#!/bin/bash
sh updateStatus.sh
address=t.clausner@student.ru.nl
mpack -s $(echo "$(sh reportcurrentsetsize.sh )") ../txtfiles/allcrawls.txt $address
sleep 1s
echo "sh sendData.sh" | at 23:00

