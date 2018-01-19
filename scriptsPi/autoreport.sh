#!/bin/bash

echo "sh sendstatus.sh" | at 09:00
echo "sh sendstatus.sh" | at 13:00
echo "sh sendstatus.sh" | at 19:00
echo "sh sendstatus.sh" | at 23:00
echo "sh sendData.sh" | at 23:00

while sleep 3600;do sh checkonline.sh; done
