#!/usr/bin/env bash


n_proc=8
day=1


t_warm_start=60
i=0
echo 'Generating DoE'
mpiexec -n $n_proc python3 Gen_DoE_PHES.py $i $day $t_warm_start
alg="turbo"
echo $alg
time mpiexec -n $n_proc python main_PHES.py $i $day $t_warm_start $alg

