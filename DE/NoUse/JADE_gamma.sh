#!/bin/bash

n=10
for ((i=1;i<=$n;i++));
do
	./JADE_gamma.py > ./log/JADE_gamma_R_$i.log
done
