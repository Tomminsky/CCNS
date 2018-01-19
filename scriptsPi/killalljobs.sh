#!/bin/bash

maxjid=$(at -l | cut -f1)

for i in $maxjid
do
echo killed $i
at -r $i
done
