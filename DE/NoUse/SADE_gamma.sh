#!/bin/bash

n=10
for ((i=1;i<=$n;i++));
do
	./SADE_gamma.py > ./log/SADE_gamma_R_$i.log
done
