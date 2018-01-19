#!/bin/bash
case $1 in
size)
  echo current file size: $(sh reportcurrentsetsize.sh)
  ;;
IP)
  echo current IP: $(sh getIP.sh)
  ;;
*)
  echo "invalid argument. Use: sh getI.sh <argument> : size, IP"
  ;;
esac
