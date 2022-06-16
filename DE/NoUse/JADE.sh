#!/bin/bash

n=10
for ((i=1;i<=$n;i++));
do
	./JADE.py > ./log/JADE_R_$i.log
done
