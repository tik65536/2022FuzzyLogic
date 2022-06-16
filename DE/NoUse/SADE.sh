#!/bin/bash

n=10
for ((i=1;i<=$n;i++));
do
	./SADE_range.py > ./log/SADE_R_$i.log
done
