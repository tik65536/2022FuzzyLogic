#!/bin/bash

n=10
for ((i=1;i<=$n;i++));
do
	./SADE_uniformPopulation.py > ./log/SADE_uniDist_R_$i.log
done
