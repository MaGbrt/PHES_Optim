#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 08:03:41 2023

@author: maxime
"""

import pandas as pd
import numpy as np 

data = pd.read_csv('Warm_start_init.csv', sep='\t')
days = np.array([1, 2, 3, 4, 5, 6, 7, 8, 12, 15, 17])
t_warm = np.array([0, 60, 120, 300])
n_pts = 8
t = 60

for day in days:
    df_day = data.loc[data['Day']==day]
    DoE = df_day[['design point', 'simulator']]
    print(DoE)

    design = df_day['design point']

    assert len(design)==n_pts
    name = 'Initial_PHES_' + str(t) + 'sec_' + str(n_pts) + 'pts_day' + str(day) + '.csv'
    design.to_csv(name, float_format="%.8f", header=False, index=False)

