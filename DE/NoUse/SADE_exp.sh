#!/bin/bash

n=10
for ((i=1;i<=$n;i++));
do
	./SADE_expDist.py > ./log/SADE_expDist_R_$i.log
done
