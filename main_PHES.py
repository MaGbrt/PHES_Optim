#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 17:42:28 2022

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
from Problems.PHES import PHES

from Full_loops.parallel_EGO_cycle import par_EGO_run, par_MC_qEGO_run, par_MCbased_qEGO_run, par_Lanczos_MCbased_qEGO_run
from Full_loops.parallel_g1_BSP_EGO_cycle import par_g1_BSP_EGO_run, par_g1_BSP_qEGO_run, par_g1_BSP2_EGO_run
from Full_loops.parallel_l1_BSP_EGO_cycle import par_l1_BSP_EGO_run, par_lg1_BSP_EGO_run, par_lg2_BSP_EGO_run, par_l2_BSP_EGO_run
from Full_loops.parallel_SAGA_SaaF import par_SAGA_SaaF_run
from Full_loops.parallel_TuRBO1_cycle import par_Turbo1_run, par_fast_Turbo1_run
from Full_loops.parallel_MACE import par_MACE_run
from Full_loops.parallel_Hybrid_TuRBO_SAGA import par_Hybrid_TuRBO_SAGA_run
from Full_loops.parallel_GA import par_GA_run


from DataSets.DataSet import DataBase
from random import random

# Budget parameters
DoE_num = int(sys.argv[1]); 
day = int(sys.argv[2])  # used scenario
t_warm = int(sys.argv[3]) # time dedicated to warm start


dim = 30;
fill_ratio = 0.5
f = PHES(dim, day, fill_ratio)
batch_size = n_proc;

id_name = '_PHES_D' + str(dim) + '_day' + str(day) + '_t_warm' + str(t_warm) + '_batch' + str(batch_size)
f_init = open("DoE_PHES/Time_init_DB" + id_name + '_run' + str(DoE_num) + ".txt", "r")
t_init_DB = int(np.floor(float(f_init.read())))
print('Initialisation of the DB took ', t_init_DB, 's')
budget = Global_Var.budget
t_max = (Global_Var.max_time - t_warm - t_init_DB) #Global_Var.max_time; # seconds
print('Remaining time for optimization: ', t_max, 's')
n_init = Global_Var.n_init #(int) (min((0.2*budget)*batch_size, 128));
n_cycle = Global_Var.n_cycle #(int) (0.8*budget);

size_Lanczos = Global_Var.size_Lanczos


if my_rank == 0:
    print('The budget for this run is:', budget, ' cycles.')
n_leaves = 4*n_proc
tree_depth = int(np.log(n_leaves)/np.log(2))
n_init_leaves = pow(2, tree_depth)
#max_leaves = 4 * n_init_leaves
n_learn = min(n_init, 128)
n_TR = 2 * batch_size

threshold = Global_Var.threshold



run_random = False
run_eShotgun = False
run_ABAFMo = False
run_MACE = False
run_Hybrid_TuRBO_SAGA = False
run_SAGA_SaaF = False
run_SAPSO_SaaF = False
run_GA = False
run_PSO = False
run_turbo_ei = False
run_fast_turbo_ei = False
run_turbo_m = False
run_qEGO = False
run_MC_qEGO = False 
run_MCbased_qEGO = False
run_Lanczos_MCbased_qEGO = False
run_gBSP_EGO = False
run_gBSP_qEGO = False
run_gBSP2_EGO = False
run_lBSP_EGO = False
run_lg1BSP_EGO = False
run_l2BSP_EGO = False
run_Skip_qEGO = False
run_Sparse_qEGO = False

if (sys.argv[4] == "random"):
    run_random = True
if (sys.argv[4] == "turbo"):
    run_turbo_ei = True
if (sys.argv[4] == "fast_turbo"):
    run_fast_turbo_ei = True
if (sys.argv[4] == "turbo_m"):
    run_turbo_m = True
# if (sys.argv[4] == "turbo_2m"):
#     run_turbo_m = True
if (sys.argv[4] == "KBqEGO"):
    run_qEGO = True
if (sys.argv[4] == "MCqEGO"):
    run_MC_qEGO = True
if (sys.argv[4] == "MCbasedqEGO"):
    run_MCbased_qEGO = True
if (sys.argv[4] == "LanczosMCbasedqEGO"):
    run_Lanczos_MCbased_qEGO = True
if (sys.argv[4] == "gBSPEGO"):
    run_gBSP_EGO = True
if (sys.argv[4] == "gBSPqEGO"):
    run_gBSP_qEGO = True
if (sys.argv[4] == "gBSP2EGO"):
    run_gBSP2_EGO = True
if (sys.argv[4] == "lBSPEGO"):
    run_lBSP_EGO = True
if (sys.argv[4] == "lg1BSPEGO"):
    run_lg1BSP_EGO = True
if (sys.argv[4] == "l2BSPEGO"):
    run_l2BSP_EGO = True
if (sys.argv[4] == "SKIP_qEGO"):
    run_Skip_qEGO = True
if (sys.argv[4] == "Sparse_qEGO"):
    run_Sparse_qEGO = True
if (sys.argv[4] == "eShotgun"):
    run_eShotgun = True
if (sys.argv[4] == "ABAFMo"):
    run_ABAFMo = True
if (sys.argv[4] == "MACE"):
    run_MACE = True
if (sys.argv[4] == "Hybrid_TuRBO_SAGA"):
    run_Hybrid_TuRBO_SAGA = True
if (sys.argv[4] == "SAGA_SaaF"):
    run_SAGA_SaaF = True
if (sys.argv[4] == "SAPSO_SaaF"):
    run_SAPSO_SaaF = True
if (sys.argv[4] == "GA"):
    run_GA = True
if (sys.argv[4] == "PSO"):
    run_PSO = True

# Initialize Data
#rep_vec = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
rep_vec = np.array([DoE_num])
n_rep = len(rep_vec)
folder = 'Results_PHES/'
ext = '.txt'

if (my_rank == 0):
    target_EGO = np.zeros((n_rep, n_cycle+1))
    target_MC_qEGO = np.zeros((n_rep, n_cycle+1))
    target_eShotgun = np.zeros((n_rep, n_cycle+1))
    target_ABAFMo = np.zeros((n_rep, n_cycle+1))
    target_MACE = np.zeros((n_rep, n_cycle+1))
    target_Hybrid_TuRBO_SAGA = np.zeros((n_rep, n_cycle+1))
    target_SAGA_SaaF = np.zeros((n_rep, n_cycle+1))
    target_SAPSO_SaaF = np.zeros((n_rep, n_cycle+1))
    target_GA = np.zeros((n_rep, n_cycle+1))
    target_PSO = np.zeros((n_rep, n_cycle+1))
    target_Skip_qEGO = np.zeros((n_rep, n_cycle+1))
    target_Sparse_qEGO = np.zeros((n_rep, n_cycle+1))
    target_MCbased_qEGO = np.zeros((n_rep, n_cycle+1))
    target_Lanczos_MCbased_qEGO = np.zeros((n_rep, n_cycle+1))
    target_Turbo_ei = np.zeros((n_rep, n_cycle+1))
    target_fast_Turbo_ei = np.zeros((n_rep, n_cycle+1))
    target_Turbo_m = np.zeros((n_rep, n_cycle+1))
    target_g1_BSP_EGO = np.zeros((n_rep, n_cycle+1))
    target_g1_BSP_qEGO = np.zeros((n_rep, n_cycle+1))
    target_g1_BSP2_EGO = np.zeros((n_rep, n_cycle+1))
    target_l1_BSP_EGO = np.zeros((n_rep, n_cycle+1))
    target_lg1_BSP_EGO = np.zeros((n_rep, n_cycle+1))
    target_l2_BSP_EGO = np.zeros((n_rep, n_cycle+1))
    target_random = np.zeros((n_rep, n_cycle+1))

#    for i_rep in range(n_rep):
    for i_rep in range(n_rep):
        k_rep = rep_vec[i_rep]
        # Input data scaled in [0, 1]^d
        DB = DataBase(f, n_init)
        r = random()*1000
        input_file = 'Initial_DoE' + id_name + '_run' + str(k_rep) + ext #None

        if (input_file == None):
            par_create = np.ones(1, dtype = 'i')
            comm.Bcast(par_create, root = 0)
            DB.par_create(comm = comm, seed = r)
            #            DB.create(seed=r)
#            DB.create_lhs(seed=r)
            full_name = 'Initial_DoE' + id_name + '_run' + str(k_rep) + ext
            DB.save_txt('DoE_PHES/' + full_name)
        else :
            par_create = np.zeros(1, dtype = 'i')
            comm.Bcast(par_create, root = 0)
            DB.read_txt('DoE_PHES/' + input_file)
            n_init = DB._size
            n_learn = min(n_init, 128)
            comm.Bcast(np.array(n_init, dtype='i'), root = 0)

        comm.Barrier()
        if run_random:
            print('\n Synchronize before running Random')
            DB_random = DB.copy()
            target_random[i_rep, :], time_random = par_random_run(DB_random, n_cycle, t_max, batch_size, DoE_num, comm)
            full_name = 'Random' + id_name + '_t_max' + str(t_max) + '_run' + str(k_rep) + ext
            DB_random.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_random, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            del DB_random
        
        comm.Barrier()
        if run_eShotgun:
            print('\n Synchronize before running eShotgun')
            DB_eShotgun = DB.copy() 
            target_eShotgun[i_rep, :], time_eShotgun = par_eShotgun_run(DB_eShotgun, n_cycle, t_max, batch_size, DoE_num, comm)
            full_name = 'eShotgun' + id_name + '_t_max' + str(t_max) + '_run' + str(k_rep) + ext
            DB_eShotgun.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_eShotgun, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            del DB_eShotgun

        comm.Barrier()
        if run_ABAFMo:
            print('\n Synchronize before running ABAFMo')
            DB_ABAFMo = DB.copy() 
            target_ABAFMo[i_rep, :], time_ABAFMo = par_ABAFMo_run(DB_ABAFMo, n_cycle, t_max, batch_size, DoE_num, comm)
            full_name = 'ABAFMo' + id_name + '_t_max' + str(t_max) + '_run' + str(k_rep) + ext
            DB_ABAFMo.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_ABAFMo, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            del DB_ABAFMo

        comm.Barrier()
        if run_MACE:
            print('\n Synchronize before running MACE')
            DB_MACE = DB.copy() 
            target_MACE[i_rep, :], time_MACE = par_MACE_run(DB_MACE, n_cycle, t_max, batch_size, DoE_num, comm)
            full_name = 'MACE' + id_name + '_t_max' + str(t_max) + '_run' + str(k_rep) + ext
            DB_MACE.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_MACE, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            np.savetxt(fname = folder + 'Target_per_cycle_' + full_name, X = target_MACE, fmt ='%.6f', delimiter= '\t')
            del DB_MACE

        comm.Barrier()
        if run_Hybrid_TuRBO_SAGA:
            print('\n Synchronize before running Hybrid_TuRBO_SAGA')
            DB_Hybrid_TuRBO_SAGA = DB.copy() 
            file_id = id_name + '_DoE_' + str(DoE_num)
            target_Hybrid_TuRBO_SAGA[i_rep, :], time_Hybrid_TuRBO_SAGA = par_Hybrid_TuRBO_SAGA_run(DB_Hybrid_TuRBO_SAGA, n_cycle, t_max, batch_size, threshold, file_id, comm)
            full_name = 'Hybrid_TuRBO_SAGA' + id_name + '_t_max' + str(t_max) + '_run' + str(k_rep) + ext
            DB_Hybrid_TuRBO_SAGA.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_Hybrid_TuRBO_SAGA, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            np.savetxt(fname = folder + 'Target_per_cycle_' + full_name, X = target_Hybrid_TuRBO_SAGA, fmt ='%.6f', delimiter= '\t')
            del DB_Hybrid_TuRBO_SAGA

        comm.Barrier()
        if run_SAGA_SaaF:
            print('\n Synchronize before running SAGA_SaaF')
            DB_SAGA_SaaF = DB.copy() 
            file_id = id_name + '_DoE_' + str(DoE_num)
            target_SAGA_SaaF[i_rep, :], time_SAGA_SaaF = par_SAGA_SaaF_run(DB_SAGA_SaaF, n_cycle, t_max, batch_size, file_id, comm)
            print('Optimization done')
            full_name = 'SAGA_SaaF' + id_name + '_t_max' + str(t_max) + '_run' + str(k_rep) + ext
            DB_SAGA_SaaF.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_SAGA_SaaF, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            np.savetxt(fname = folder + 'Target_per_cycle_' + full_name, X = target_SAGA_SaaF, fmt ='%.6f', delimiter= '\t')
            del DB_SAGA_SaaF

        comm.Barrier()
        if run_SAPSO_SaaF:
            print('\n Synchronize before running SAPSO_SaaF')
            DB_SAPSO_SaaF = DB.copy()
            file_id = id_name + '_DoE_' + str(DoE_num)
            target_SAPSO_SaaF[i_rep, :], time_SAPSO_SaaF = par_SAPSO_SaaF_run(DB_SAPSO_SaaF, n_cycle, t_max, batch_size, file_id, comm)
            print('Optimization done')
            full_name = 'SAPSO_SaaF' + id_name + '_t_max' + str(t_max) + '_run' + str(k_rep) + ext
            DB_SAPSO_SaaF.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_SAPSO_SaaF, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            np.savetxt(fname = folder + 'Target_per_cycle_' + full_name, X = target_SAPSO_SaaF, fmt ='%.6f', delimiter= '\t')
            del DB_SAPSO_SaaF

        comm.Barrier()
        if run_GA:
            print('\n Synchronize before running GA')
            DB_GA = DB.copy()
            file_id = id_name + '_DoE_' + str(DoE_num)
            target_GA[i_rep, :], time_GA = par_GA_run(DB_GA, n_cycle, t_max, file_id, comm)
            print('Optimization done')
            full_name = 'GA' + id_name + '_t_max' + str(t_max) + '_run' + str(k_rep) + ext
            DB_GA.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_GA, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            np.savetxt(fname = folder + 'Target_per_cycle_' + full_name, X = target_GA, fmt ='%.6f', delimiter= '\t')
            del DB_GA

        comm.Barrier()
        if run_PSO:
            print('\n Synchronize before running PSO')
            DB_PSO = DB.copy()
            file_id = id_name + '_DoE_' + str(DoE_num)
            target_PSO[i_rep, :], time_PSO = par_PSO_run(DB_PSO, n_cycle, t_max, file_id, comm)
            print('Optimization done')
            full_name = 'PSO' + id_name + '_t_max' + str(t_max) + '_run' + str(k_rep) + ext
            DB_PSO.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_PSO, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            np.savetxt(fname = folder + 'Target_per_cycle_' + full_name, X = target_PSO, fmt ='%.6f', delimiter= '\t')
            del DB_PSO

        comm.Barrier()
        if run_turbo_ei:
            print('\n Synchronize before running Turbo')
            DB_Turbo_ei = DB.copy()
            target_Turbo_ei[i_rep, :], time_turbo_ei = par_Turbo1_run(DB_Turbo_ei, n_cycle, t_max, batch_size, "ei", DoE_num, comm)
            full_name = 'TuRBO' + id_name + '_t_max' + str(t_max) + '_run' + str(k_rep) + ext
            DB_Turbo_ei.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_turbo_ei, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            np.savetxt(fname = folder + 'Target_per_cycle_' + full_name, X = target_Turbo_ei, fmt ='%.6f', delimiter= '\t')
            del DB_Turbo_ei

        comm.Barrier()
        if run_fast_turbo_ei:
            print('\n Synchronize before running Turbo')
            DB_fast_Turbo_ei = DB.copy()
            target_fast_Turbo_ei[i_rep, :], time_fast_Turbo_ei = par_fast_Turbo1_run(DB_fast_Turbo_ei, n_cycle, t_max, batch_size, "ei", DoE_num, comm)
            full_name = 'Fast_TuRBO' + id_name + '_t_max' + str(t_max) + '_run' + str(k_rep) + ext
            DB_fast_Turbo_ei.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_fast_Turbo_ei, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            del DB_fast_Turbo_ei

        comm.Barrier()
        if run_turbo_m:
            print('\n Synchronize before running Turbo_m')
            DB_Turbo_m = DB.copy()
            target_Turbo_m[i_rep, :], time_turbo_m = par_Turbom_run(DB_Turbo_m, n_cycle, t_max, batch_size, n_TR, "ei", DoE_num, comm)
            full_name = 'TuRBO_' + str(n_TR) + id_name + '_t_max' + str(t_max) + '_run' + str(k_rep) + ext
            DB_Turbo_m.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_turbo_m, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            del DB_Turbo_m

        comm.Barrier()
        if run_qEGO:
            print('\n Synchronize before running qEGO from Ginsbourger et al.')
            DB_EGO = DB.copy()
            target_EGO[i_rep, :], time_EGO = par_EGO_run(DB_EGO, n_cycle, t_max, batch_size, DoE_num, 'ei', comm)
            full_name = 'KB_qEGO_ei' + id_name + '_t_max' + str(t_max) + '_run' + str(k_rep) + ext
            DB_EGO.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_EGO, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            del DB_EGO
            
        comm.Barrier()
        if run_MC_qEGO:
            print('\n Synchronize before running multi criteria qEGO')   
            DB_MC_qEGO = DB.copy()
            target_MC_qEGO[i_rep, :], time_MC_qEGO = par_MC_qEGO_run(DB_MC_qEGO, n_cycle, t_max, batch_size, DoE_num, comm)
            full_name = 'MC_qEGO' + id_name + '_t_max' + str(t_max) + '_run' + str(k_rep) + ext
            DB_MC_qEGO.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_MC_qEGO, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            np.savetxt(fname = folder + 'Target_per_cycle_' + full_name, X = target_MC_qEGO, fmt ='%.6f', delimiter= '\t')
            del DB_MC_qEGO
            
        comm.Barrier()
        if run_MCbased_qEGO:
            print('\n Synchronize before running Monte Carlo based qEGO')
            DB_MCbased_qEGO = DB.copy()
            target_MCbased_qEGO[i_rep, :], time_MCbased_qEGO = par_MCbased_qEGO_run(DB_MCbased_qEGO, n_cycle, t_max, batch_size, DoE_num, comm)
            full_name = 'MCbased_qEGO' + id_name + '_t_max' + str(t_max) + '_run' + str(k_rep) + ext
            DB_MCbased_qEGO.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_MCbased_qEGO, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            del DB_MCbased_qEGO
            
        comm.Barrier()
        if run_Lanczos_MCbased_qEGO:
            print('\n Synchronize before running Monte Carlo based qEGO with Lanczos fast variance')
            DB_Lanczos_MCbased_qEGO = DB.copy()
            target_Lanczos_MCbased_qEGO[i_rep, :], time_Lanczos_MCbased_qEGO = par_Lanczos_MCbased_qEGO_run(DB_Lanczos_MCbased_qEGO, n_cycle, t_max, batch_size, size_Lanczos, DoE_num, comm)
            full_name = 'Lanczos_MCbased_qEGO' + id_name + '_t_max' + str(t_max) + '_run' + str(k_rep) + ext
            DB_Lanczos_MCbased_qEGO.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_Lanczos_MCbased_qEGO, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            np.savetxt(fname = folder + 'Target_per_cycle_' + full_name, X = target_Lanczos_MCbased_qEGO, fmt ='%.6f', delimiter= '\t')
            del DB_Lanczos_MCbased_qEGO

        comm.Barrier()
        if run_Skip_qEGO:
            print('\n Synchronize before running Monte Carlo based SKIP-qEGO')
            DB_Skip_qEGO = DB.copy()
            target_Skip_qEGO[i_rep, :], time_Skip_qEGO = par_MCbased_SKIP_qEGO_run(DB_Skip_qEGO, n_cycle, t_max, batch_size, DoE_num, comm)
            full_name = 'SKIP_qEGO' + id_name + '_t_max' + str(t_max) + '_run' + str(k_rep) + ext
            DB_Skip_qEGO.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_Skip_qEGO, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            del DB_Skip_qEGO
            
        comm.Barrier()
        if run_Sparse_qEGO:
            print('\n Synchronize before running Monte Carlo based Sparse-qEGO')
            DB_Sparse_qEGO = DB.copy()
            target_Sparse_qEGO[i_rep, :], time_Sparse_qEGO = par_MCbased_Sparse_qEGO_run(DB_Sparse_qEGO, n_cycle, t_max, batch_size, DoE_num, comm)
            full_name = 'Sparse_qEGO' + id_name + '_t_max' + str(t_max) + '_run' + str(k_rep) + ext
            DB_Sparse_qEGO.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_Sparse_qEGO, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            del DB_Sparse_qEGO
            
        comm.Barrier()
        if run_gBSP_EGO:
            print('\n Synchronize before running BSP-EGO with global model')
            DB_g1_BSP_EGO = DB.copy()
            target_g1_BSP_EGO[i_rep, :], time_g1_BSP_EGO = par_g1_BSP_EGO_run(DB_g1_BSP_EGO, n_cycle, t_max, batch_size, tree_depth, DoE_num, comm)
            full_name = 'Global_BSP_EGO' + id_name + '_t_max' + str(t_max) + '_depth' + str(tree_depth) + '_run' + str(k_rep) + ext
            DB_g1_BSP_EGO.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_g1_BSP_EGO, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            np.savetxt(fname = folder + 'Target_per_cycle_' + full_name, X = target_g1_BSP_EGO, fmt ='%.6f', delimiter= '\t')
            del DB_g1_BSP_EGO

        comm.Barrier()
        if run_gBSP_qEGO:
            print('\n Synchronize before running BSP-qEGO with global model')
            DB_g1_BSP_qEGO = DB.copy()
            target_g1_BSP_qEGO[i_rep, :], time_g1_BSP_qEGO = par_g1_BSP_qEGO_run(DB_g1_BSP_qEGO, n_cycle, t_max, batch_size, tree_depth, DoE_num, comm)
            full_name = 'Global_BSP_qEGO' + id_name + '_t_max' + str(t_max) + '_depth' + str(tree_depth) + '_run' + str(k_rep) + ext
            DB_g1_BSP_qEGO.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_g1_BSP_qEGO, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            del DB_g1_BSP_qEGO

        comm.Barrier()
        if run_gBSP2_EGO:
            print('\n Synchronize before running BSP2-EGO with global model')
            DB_g1_BSP2_EGO = DB.copy()
            target_g1_BSP2_EGO[i_rep, :], time_g1_BSP2_EGO = par_g1_BSP2_EGO_run(DB_g1_BSP2_EGO, n_cycle, t_max, batch_size, (tree_depth + 2), DoE_num, comm)
            full_name = 'Global_BSP22_EGO' + id_name + '_t_max' + str(t_max) + '_depth' + str((tree_depth+2)) + '_run' + str(k_rep) + ext
            DB_g1_BSP2_EGO.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_g1_BSP2_EGO, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            del DB_g1_BSP2_EGO

        comm.Barrier()
        if run_lBSP_EGO:
            print('\n Synchronize before running BSP-EGO with local models')
            DB_l1_BSP_EGO = DB.copy()
            target_l1_BSP_EGO[i_rep, :], time_l1_BSP_EGO = par_l1_BSP_EGO_run(DB_l1_BSP_EGO, n_cycle, t_max, batch_size, tree_depth, n_learn, DoE_num, comm)
            full_name = 'Local_BSP_EGO' + id_name + '_t_max' + str(t_max) + '_depth' + str(tree_depth) + '_run' + str(k_rep) + ext
            DB_l1_BSP_EGO.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_l1_BSP_EGO, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            np.savetxt(fname = folder + 'Target_per_cycle_' + full_name, X = target_l1_BSP_EGO, fmt ='%.6f', delimiter= '\t')
            del DB_l1_BSP_EGO
    
        comm.Barrier()
        if run_lg1BSP_EGO:
            print('\n Synchronize before running lBSP-EGO with local models and one global model')
            DB_lg1_BSP_EGO = DB.copy()
            target_lg1_BSP_EGO[i_rep, :], time_lg1_BSP_EGO = par_lg1_BSP_EGO_run(DB_lg1_BSP_EGO, n_cycle, t_max, batch_size, tree_depth, n_learn, DoE_num, comm)
            full_name = 'LG1_BSP_EGO' + id_name + '_t_max' + str(t_max) + '_depth' + str(tree_depth) + '_run' + str(k_rep) + ext
            DB_lg1_BSP_EGO.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_lg1_BSP_EGO, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            del DB_lg1_BSP_EGO
            
        comm.Barrier()
        if run_l2BSP_EGO:
            print('\n Synchronize before running l2BSP-EGO with local models')
            DB_l2_BSP_EGO = DB.copy()
            target_l2_BSP_EGO[i_rep, :], time_l2_BSP_EGO = par_l2_BSP_EGO_run(DB_l2_BSP_EGO, n_cycle, t_max, batch_size, tree_depth, n_learn, DoE_num, comm)
            full_name = 'L2_BSP_EGO' + id_name + '_t_max' + str(t_max) + '_depth' + str(tree_depth) + '_run' + str(k_rep) + ext
            DB_l2_BSP_EGO.save_txt(folder + full_name)
            np.savetxt(fname = folder + 'Times_per_cycle_' + full_name, X = time_l2_BSP_EGO, fmt ='%.6f', delimiter= '\t', header='model \t ap \t model+ap \t eval \t total')
            del DB_l2_BSP_EGO

        del DB
        
        print('END - Master')
else:
    for i_rep in range(n_rep):
        par_create = np.zeros(1, dtype = 'i')
        comm.Bcast(par_create, root = 0)
        if par_create[0] == 1:
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            DB_worker.par_create(comm = comm)
        else:
            n_in = np.zeros(1, dtype = 'i')
            comm.Bcast(n_in, root = 0)
            n_init = n_in[0]
    
        comm.Barrier()
        if run_random:
            print('\n Synchronize before running random search - workers')
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            par_random_run(DB_worker, n_cycle, None, batch_size, None, comm)
            del DB_worker

        comm.Barrier()
        if run_eShotgun:
            print('\n Synchronize before running eShotgun search - workers')
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            par_eShotgun_run(DB_worker, n_cycle, None, batch_size, None, comm)
            del DB_worker            

        comm.Barrier()
        if run_ABAFMo:
            print('\n Synchronize before running ABAFMo search - workers')
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            par_ABAFMo_run(DB_worker, n_cycle, None, batch_size, None, comm)
            del DB_worker            

        comm.Barrier()
        if run_MACE:
            print('\n Synchronize before running MACE search - workers')
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            par_MACE_run(DB_worker, n_cycle, None, batch_size, None, comm)
            del DB_worker            

        comm.Barrier()
        if run_Hybrid_TuRBO_SAGA:
            print('\n Synchronize before running Hybrid_TuRBO_SAGA search - workers')
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            par_Hybrid_TuRBO_SAGA_run(DB_worker, n_cycle, None, batch_size, threshold, None, comm)
            del DB_worker            

        comm.Barrier()
        if run_SAGA_SaaF:
            print('\n Synchronize before running SAGA_SaaF search - workers')
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            par_SAGA_SaaF_run(DB_worker, n_cycle, None, batch_size, None, comm)
            del DB_worker            

        comm.Barrier()
        if run_SAPSO_SaaF:
            print('\n Synchronize before running SAPSO_SaaF search - workers')
            DB_worker = DataBase(f, n_init) # in order to access function evaluation  
            par_SAPSO_SaaF_run(DB_worker, n_cycle, None, batch_size, None, comm)
            del DB_worker

        comm.Barrier()
        if run_GA:
            print('\n Synchronize before running GA search - workers')
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            par_GA_run(DB_worker, n_cycle, None, None, comm)
            del DB_worker

        comm.Barrier()
        if run_PSO:
            print('\n Synchronize before running PSO search - workers')
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            par_PSO_run(DB_worker, n_cycle, None, None, comm)
            del DB_worker

        comm.Barrier()
        if run_turbo_ei:
            print('\n Synchronize before running turbo - workers')
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            par_Turbo1_run(DB_worker, n_cycle, None, batch_size, None, None, comm)
            del DB_worker

        comm.Barrier()
        if run_fast_turbo_ei:
            print('\n Synchronize before running fast turbo - workers')
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            par_fast_Turbo1_run(DB_worker, n_cycle, None, batch_size, None, None, comm)
            del DB_worker
            
        comm.Barrier()
        if run_turbo_m:
            print('\n Synchronize before running turbo_m - workers')
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            par_Turbom_run(DB_worker, n_cycle, None, batch_size, n_TR, None, None, comm)
            del DB_worker

        comm.Barrier()
        if run_qEGO:
            print('\n Synchronize before running qEGO - workers')
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            par_EGO_run(DB_worker, n_cycle, None, batch_size, None, None, comm)
            del DB_worker
    
        comm.Barrier()
        if run_MC_qEGO:
            print('\n Synchronize before running mic qEGO - workers')
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            par_MC_qEGO_run(DB_worker, n_cycle, None, batch_size, None, comm)
            del DB_worker
        
        comm.Barrier()
        if run_MCbased_qEGO:
            print('\n Synchronize before running MC based qEGO - workers')
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            par_MCbased_qEGO_run(DB_worker, n_cycle, None, batch_size, None, comm)
            del DB_worker

        comm.Barrier()
        if run_Lanczos_MCbased_qEGO:
            print('\n Synchronize before running Monte Carlo based qEGO with Lanczos fast variance - workers')
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            par_Lanczos_MCbased_qEGO_run(DB_worker, n_cycle, None, batch_size, size_Lanczos, None, comm)
            del DB_worker

        comm.Barrier()
        if run_Skip_qEGO:
            print('\n Synchronize before running SKIP qEGO - workers')
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            par_MCbased_SKIP_qEGO_run(DB_worker, n_cycle, None, batch_size, None, comm)
            del DB_worker

        comm.Barrier()
        if run_Sparse_qEGO:
            print('\n Synchronize before running Sparse qEGO - workers')
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            par_MCbased_Sparse_qEGO_run(DB_worker, n_cycle, None, batch_size, None, comm)
            del DB_worker

        comm.Barrier()
        if run_gBSP_EGO:
            print('\n Synchronize before running BSP-EGO with global model - workers')
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            par_g1_BSP_EGO_run(DB_worker, n_cycle, None, batch_size, None, None, comm)
            del DB_worker

        comm.Barrier()
        if run_gBSP_qEGO:
            print('\n Synchronize before running BSP-qEGO with global model - workers')
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            par_g1_BSP_qEGO_run(DB_worker, n_cycle, None, batch_size, None, None, comm)
            del DB_worker

        comm.Barrier()
        if run_gBSP2_EGO:
            print('\n Synchronize before running BSP2-EGO with global model - workers')
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            par_g1_BSP2_EGO_run(DB_worker, n_cycle, None, batch_size, None, None, comm)
            del DB_worker

        comm.Barrier()
        if run_lBSP_EGO:
            print('\n Synchronize before running BSP-EGO with local models - workers')
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            par_l1_BSP_EGO_run(DB_worker, n_cycle, None, batch_size, None, n_learn, None, comm)
            del DB_worker
            
        comm.Barrier()
        if run_lg1BSP_EGO:
            print('\n Synchronize before running lg1BSP-EGO with local models and one global model - workers')
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            par_lg1_BSP_EGO_run(DB_worker, n_cycle, None, batch_size, None, n_learn, None, comm)
            del DB_worker

        comm.Barrier()
        if run_l2BSP_EGO:
            print('\n Synchronize before running l2BSP-EGO with local models and one global model - workers')
            DB_worker = DataBase(f, n_init) # in order to access function evaluation
            par_l2_BSP_EGO_run(DB_worker, n_cycle, None, batch_size, None, n_learn, None, comm)
            del DB_worker
            
        print('END', my_rank)
