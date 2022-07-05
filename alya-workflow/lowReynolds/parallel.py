#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 18:28:05 2022

@author: al
"""

import numpy as np
import subprocess
import os
import itertools
from analysis import *


"""
Parameters that define the number of simulations to run and
the values to use on them
"""

# Define the ranges for variables to simulate
angRange    = [0, 45]       # degrees
vRange      = [-2, 0]       # order of magnitude
poroRange   = [3, 5]        # order of magnitude

# Define the number of angles that will be swept and the number of different
# simulations (vel, poro) that will be run at each
# [nAng, nSim_ang] for a total of nAng * nSim_ang simulations
nSimNums = [2, 4]       


"""
function definitions
"""

# Runs the 'case' in directory 'fold' using the angle, velocity and porosity
# in *varAVP = (ang, vel, poro). The temporary case is run in a copy of the
# 'fold' directory with the 'caseNum' appended to it ("fold-caseNum") 
def runParallelCase(fold, case, caseNum, varAVP):
    # read run variables from varAVP tuple
    ang, vel, por = varAVP

    # create a copy of the folder to run the simulation there in parallel
    # without interfering other processes
    newFold = fold + "-" + str(caseNum)
    subprocess.run(["cp", "-r", "./" + fold, "./" + newFold])

    # run the case with ang and vel values  
    runCase(newFold, case, ang, vel, por, HPC=True)     
    
    # clean the temporary case folders    
    subprocess.run(["rm", "-r", "./" + newFold])


# Give the ranges for the variables: angle, velocity and porosity as 2 components
# lists.
# nList is a 2 components list with [num angles, num simulations at each angle]
# ----!
# TODO: right now, although not probable, arrays could have repeated values
#       -> Change to implementation that guarantees unique sets.
# ----!
def getVarList(angR, velR, porR, nList):

    # seed random generator with a repetible constant to give each parallel
    # process the same variables list
    seed = int.from_bytes("RAISE".encode(), 'big')
    rng = np.random.default_rng(seed)

    ang = np.linspace(angRange[0], angRange[1], nList[0])   # constant degrees steps
    # random sampling for velocities and porosity
    vel = 10**rng.uniform(vRange[0], vRange[1], nList[0]*nList[1])
    por = 10**rng.uniform(poroRange[0], poroRange[1], nList[0]*nList[1])

    # list of conditions to simulate as [(ang, vel, poro), ... ]
    varL = [(ang[i//nList[1]], vel[i], por[i]) for i in range(nList[0]*nList[1])]    

    return varL


# Get Total Number of processes based on whether PBS
# environment variables are present
if ('PBS_NUM_NODES' in os.environ):
   ID = int(os.environ['PBS_TASKNUM']) - 2
   NT = int(os.environ['PBS_NUM_NODES']) * int(os.environ['PBS_NUM_PPN'])
else:
   ID = 0
   NT = 1


# Gather the combination of variable values in a single list
varList = getVarList(angRange, vRange, poroRange, nSimNums)
nComb = len(varList)      # List length: total number of combinations to run


# Distribute the cases to simulate amongst the different processes
for caseNum in range(nComb):
    if (ID == (caseNum % NT)):
        print(f"{ID} in {NT}: process case {caseNum} -> {varList[caseNum]}")
        runParallelCase("surrogate", "matSurr", caseNum, varList[caseNum])
        
            
