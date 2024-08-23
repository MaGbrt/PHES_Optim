#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 17:32:42 2023

@author: gobertm
"""

from mpi4py import MPI

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
n_proc = comm.Get_size()
print('From ', my_rank, ' : Running main with ', n_proc, 'proc')
import numpy as np
import sys

import Global_Var
from Global_Var import *
import torch
from DataSets.DataSet import DataBase
from random import random
from scipy.stats import qmc
from time import time
import pandas

# Budget parameters
DoE_num = int(sys.argv[1]); 
batch_size = n_proc;
budget = Global_Var.budget
t_max = Global_Var.max_time; # seconds
n_init = Global_Var.n_init #(int) (min((0.2*budget)*batch_size, 128));
n_cycle = Global_Var.n_cycle #(int) (0.8*budget);


  
print('The maximum budget for this run is:', budget)
    

dim = 30
from Problems.PHES import PHES
day = int(sys.argv[2])  # used scenario
t_warm = int(sys.argv[3]) # time dedicated to warm start
fill_ratio = 0.5
f = PHES(dim, day, fill_ratio)

#%%
if my_rank ==0:
    t_start = time()
    DB = DataBase(f, n_init)
    if (t_warm == 0):
        x_warm = None
        n_warm=0
    else:
        file_warm_start = './DoE_PHES/Initial_PHES_' + str(t_warm) + 'sec_8pts_day' + str(day) + '.csv'
        df = pandas.read_csv(file_warm_start, dtype=float, header=None, sep=' ')
        x_warm_ = df.values
        x_warm = DB.my_map(x_warm_)
        n_warm = len(x_warm)

    print(x_warm)
    # Initialize Data
    folder = 'DoE_PHES/'
    ext = '.txt'
    id_name = '_PHES_D' + str(dim) + '_day' + str(day) + '_t_warm' + str(t_warm) + '_batch' + str(batch_size)

#%%
if my_rank == 0:
    DB._X = None
    seed = torch.rand(1)
    torch.manual_seed(seed)
    sampler = qmc.LatinHypercube(d=dim)
    sample = sampler.random(n=n_init-n_warm)
    X = torch.tensor(sample)
    if (t_warm!=0):
        X_init = torch.concatenate([X, torch.tensor(x_warm)], axis = 0)
    else:
        X_init = X.clone()
    ## send to workers
    n_cand = np.zeros(n_proc, dtype = 'i')
    for c in range(len(X_init)):
        send_to = c%n_proc
        n_cand[send_to] += 1
    
    ## Broadcast n_cand
    comm.Bcast(n_cand, root = 0)
    for c in range(len(X_init)):
        send_cand = X_init[c].numpy()
        send_to = c%n_proc
        if (send_to != 0):
            comm.send(send_cand, dest = send_to, tag = c)
    
    ## Evaluate
    for c in range(int(n_cand[0])):
        y_new = DB.eval_f(X_init[n_proc*c].numpy())
        if DB._X == None :
            DB._X = X_init[n_proc*c].unsqueeze(0)
            DB._y = torch.tensor(y_new)
        else :
            DB.add(X_init[n_proc*c].unsqueeze(0), torch.tensor(y_new))
            print(c, DB._X[c], DB._y[c])
    
    ## Gather
    for c in range(len(X_init)):
        get_from = c%n_proc
        if (get_from != 0):
            recv_eval = comm.recv(source = get_from, tag = c)
            DB.add(X_init[c].unsqueeze(0), torch.tensor(recv_eval))

    if(torch.isnan(DB._X).any() == True):
        print('DB create X ', DB._X)
    if(torch.isnan(DB._y).any() == True):
        print('DB create y ', DB._y)
#            print('Try distance in creation')
    t_end = time()
    t_tot = t_end - t_start
    print('Time for generate complementary DoE: ', t_tot)
    f = open("DoE_PHES/Time_init_DB" + id_name + '_run' + str(DoE_num) + ".txt", "w")
    f.write(str(t_tot))
    f.close()
    DB.try_distance()
    full_name = 'Initial_DoE' + id_name + '_run' + str(DoE_num) + ext

    DB.save_txt(folder + full_name)

    
else :
    DB_worker = DataBase(f, n_init)

    n_cand = np.zeros(n_proc, dtype = 'i')
    comm.Bcast(n_cand, root = 0)
    cand = []
    for c in range(n_cand[my_rank]):
        cand.append(comm.recv(source = 0, tag = my_rank + c * n_proc))

    ## Evaluate
    y_new = []
    for c in range(n_cand[my_rank]):
        y_new.append(DB_worker.eval_f(cand[c]))
    ## Send it back
    for c in range(n_cand[my_rank]):
        comm.send(y_new[c], dest = 0, tag = my_rank + c * n_proc)
  
