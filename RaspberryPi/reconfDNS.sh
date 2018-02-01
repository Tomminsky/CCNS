#!/bin/bash
sudo sh -c 'echo "nameserver 8.8.8.8" | cat - /etc/resolv.conf > tmp.txt && mv tmp.txt /etc/resolv.conf'
