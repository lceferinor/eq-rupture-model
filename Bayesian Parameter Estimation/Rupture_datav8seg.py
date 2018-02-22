#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
%% Bayesian parameter estimation for rupture occurrence model: (Ceferino et al. BSSA (2018))
% Developed by: Luis Ceferino
% Date: 02/21/2017
---------------------------------------------
% Script to store the earthquake catalog data
---------------------------------------------
"""

# Required imports
import pickle
import numpy as np


# Data
N = 8 # Number of sections
current_year = 2017
year_start = 1500 # Start of the catalog
n_years = current_year - year_start + 1  # Number of years of rupture data
# Rupture History
Rupture_history = np.zeros((n_years,N)) # Rows: number of years, N: number of units (from south to north)
Rupture_history[1586-year_start,2:4] = 1
Rupture_history[1664-year_start,1] = 1
Rupture_history[1678-year_start,6:8] = 1
Rupture_history[1687-year_start,:4] = 1
Rupture_history[1725-year_start,7] = 1
Rupture_history[1746-year_start,3:7] = 1
Rupture_history[1940-year_start,3:6] = 1
Rupture_history[1966-year_start,5:7] = 1
Rupture_history[1974-year_start,2:5] = 1
Rupture_history[2007-year_start,:2] = 1


# Time since the last rupture
Years_no_EQ = np.zeros((n_years,N)) # Rows: number of years, N: number of units
Years_no_EQ[0,:] = 1000 # Initializing all previous ruptures as constant: Assuming last rupture occurred long time ago)
for i in range(1,n_years):
    Years_no_EQ[i,:] = Years_no_EQ[i-1,:]*(1-Rupture_history[i-1,:]) + 1


# Units coordinates
L_fault = 620
delta_L = L_fault/N
loc_rup_units = np.arange(N)*delta_L


# Save variables
f = open('Rupture_data_historic', 'wb')
pickle.dump(N, f)
pickle.dump(n_years, f)
pickle.dump(Rupture_history, f)
pickle.dump(Years_no_EQ, f)
pickle.dump(loc_rup_units, f)
f.close()