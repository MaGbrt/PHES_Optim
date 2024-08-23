#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 15:49:22 2023

@author: maxime
"""
from julia.api import Julia
import os

print(os.getcwd())
print(os.listdir())

from julia.api import Julia
import numpy as np
jl = Julia(compiled_modules=False)

DAM_d = np.zeros(24)
reserves_d = np.zeros(6)

jl.eval('include("evaluate.jl")')

instruction = 'evaluate_d(' + str(DAM_d) + ', ' + str(reserves_d) + ')'
p = jl.eval(instruction)

print('Python received value of profit p = ', p)