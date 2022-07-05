#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import glob
import subprocess
import os.path

"""
Point and node class definitions
--------------------------------
"""

# 2D point definition.
class point:
    def __init__(self, pX, pY):
        self.x = pX
        self.y = pY

# Node definition
class node:
    def __init__(self, num, point):
        self.N = num
        self.p = point

#%%
"""
Witness Points data processing
------------------------------
"""


"""
Class that reads witness points from a rectangular mesh on files after being
processed with the alya2pos script
The result is stored in an object with the data structure:
    .nodeNum            -> Number of nodes in the witness mesh
    .geoNodes[N]
        .node           -> Index of the node N in the mesh
        .p.x  .p.y      -> Coordinates for the node N
    .velNodes[N]
        .node           -> Index of the node N in the mesh
        .p.x  .p.y      -> Components of the velocity vector in the node N
"""
class witnessPointsRect:   

    # Inside the 'case' 'folder', there is a directory with the name of the mesh
    # ('wpMeshName') where the required files will be found.
    # By default the last step of the run is analyzed ('nstep').
    def __init__(self, folder, case, wpMeshName, nstep=-1):
        self.nodeNum = 0
        self.geoNodes = []      # List to store the geometry information for wp nodes
        self.velNodes = []      # List to store the velocity field for wp nodes
        # Read the geometry data for the witness points captured
        name = "./" + folder + "/" + wpMeshName + "/" + case + "-" + wpMeshName + ".ensi.geo"
        self.getGeoData(name)       # Function to get the data from the file
        # Read the velocity field data from the witness points
        name = "./" + folder + "/" + wpMeshName + "/" + case + "-" + wpMeshName + ".ensi.VELOC*"
        # only the final step is taken into account from all the possible captures
        name = glob.glob(name)[nstep]
        self.getVelData(name)       # Function to get the data from the file

    # Get the location of the points in the witness mesh
    def getGeoData(self, fName):
        f = open(fName, "r")        # Open file for reading
        readPhase = 0               # Set initial reading phase
        self.nodeNum = 0            # Initialize variables to get
        nodeList = []
        nodeX = []
        nodeY = []
        # Read the different lines and store the values accordingly
        for line in f.readlines():
            if (readPhase == 0):                # Skip lines till "coordinates" tag
                if (-1 != line.find("coordinates")): readPhase = 1
            elif (readPhase == 1):              # Get the total coordinates number
                self.nodeNum = int(line)
                readPhase = 2
            elif (readPhase == 2):              # Get node numbers 
                nodeList.append(int(line))                     
                if ( self.nodeNum == len(nodeList) ): readPhase = 3
            elif (readPhase == 3):              # Get nodes X positions
                nodeX.append(float(line))
                if ( self.nodeNum == len(nodeX) ): readPhase = 4
            elif (readPhase == 4):              # Get nodes Y positions
                nodeY.append(float(line))
                if ( self.nodeNum == len(nodeY) ): readPhase = 5
        f.close()       # Close file here
        if (readPhase != 5):
            raise SystemExit("Geo Data couldn't be correctly read")
        else:
            # Add a node class object to 'geoNodes' for each node read
            for i in range(self.nodeNum):
                self.geoNodes.append( node(nodeList[i], point(nodeX[i], nodeY[i])) )

    # Get the components X, Y of the velocity vectors on the witness point grid
    def getVelData(self, fName):
        f = open(fName, "r")            # Open file for reading
        readPhase = 0                   # Set initial reading phase
        velX = []                       # Initialize variables to get
        velY = []
        # Read the different lines and store the values accordingly
        for line in f.readlines():
            if (readPhase == 0):                # Skip lines till "coordinates" tag
                if (-1 != line.find("coordinates")): readPhase = 1
            elif (readPhase == 1):              # Get the X component of the velocity
                velX.append(float(line))
                if ( self.nodeNum == len(velX) ): readPhase = 2
            elif (readPhase == 2):              # Get the Y component of the velocity
                velY.append(float(line))
                if ( self.nodeNum == len(velY) ): readPhase = 3
        f.close()       # Close file here
        if (readPhase != 3):
            raise SystemExit("Vel Data couldn't be correctly read")
        else:
            # Add a node class object to 'velNodes' for each node read
            for i in range(self.nodeNum):
                self.velNodes.append( node(self.geoNodes[i].N, point(velX[i], velY[i])) )


"""
Function that writes the data inside a 'wp' witnessPointRect object to a 
'fileName' file.
Stored as lines of the tab separated data in the form:
    CoordX      CoordY      VelCompX        VelCompY
with an initial line header. 
"""
def writeWpRect(wp, fileName):
    
    # Function to get unique elements in a list
    def getUnique(origList):
        orderedList = [x for x in set(origList)]
        orderedList.sort(key=lambda x: float(x))
        return orderedList
    
    # Sort and complete the WP grid
    # Read lines in results summary file
    dataList = [[wp.geoNodes[i].p.x, wp.geoNodes[i].p.y, 
              wp.velNodes[i].p.x, wp.velNodes[i].p.y] for i in range(wp.nodeNum)]
    Xlist = getUnique([d[0] for d in dataList])     # X and Y coordinates
    Ylist = getUnique([d[1] for d in dataList])
    nX = len(Xlist)                                 # number of X and Y coordinates
    nY = len(Ylist)
    Vmatrix = np.zeros([nX, nY, 2])                 # Vmatrix creation
    # mask for present data: To keep account of what coordinates have an entry in
    # the witness point saved file
    for d in dataList:         # Store V vector in corresponding matrix coordinates
        iX = Xlist.index(d[0])             # X and Y index in the matrix
        iY = Ylist.index(d[1])
        Vmatrix[iX][iY][:] = [d[2], d[3]]           # Vx and Vy assignation
        
    # Open the filename and set the data header
    f = open(fileName, 'w')
    line ="Coordinates (X,Y)\t\tVelocity (Vx, Vy)\n"
    f.write(line)
    # writes a line for each velocity vector captured at each coordinate
    for iX in range(nX):
        for iY in range(nY):
            line = (str(Xlist[iX]) + "\t" + str(Ylist[iY]) + "\t" + 
                   str(Vmatrix[iX][iY][0]) + "\t" + str(Vmatrix[iX][iY][1]) + "\n")
            f.write(line)
    f.close()    


#%%
    
"""
Code to automate the Alya run of different cases inside different folders
"""

# Change the Cosinus 'value' of the angle where the boundary condition changes
# from impossed velocity to free condition (needed to avoid ouptut flow artifacts)
# 'fold' is the name of the folder where 'case' is located

def setCos(fold, case, value):
    # Store file lines as list of strings
    nsiFile = "./" + fold + "/" + case + ".nsi.dat"
    f = open(nsiFile, "r")
    lines = f.readlines()
    f.close()
    # Search for the line defining cosinus value and modify it with 'value'
    for i,line in enumerate(lines):
        if ( -1 != line.find("INFLOW_COSINE") ):
            lines[i] = "    " + "INFLOW_COSINE=" + str(round(value, 6)) + "\n"
    # Rewrite the whole file including the modified line
    f = open(nsiFile, "w")
    f.writelines(lines)
    f.close()


# Sets the angle to rotate the mesh in the .ker.dat file
# the effect is like rotating the velocity vector the same amount in the
# simulation configuration files

def setVAngle(kerFile, angle):
    # Store file lines as list of strings
    f = open(kerFile, "r")
    lines = f.readlines()
    f.close()
    # Search for the line defining cosinus value and modify it with 'value'
    for i,line in enumerate(lines):
        if ( -1 != line.find("ROTATION:") ):
            lines[i] = "    " + "ROTATION: Z, ANGLE=-" + angle + "\n"
    # Rewrite the whole file including the modified line
    f = open(kerFile, "w")
    f.writelines(lines)
    f.close()


# sets the modulus of the velocity in the .ker.dat file

def setVmod(kerFile, mod):
    f = open(kerFile, "r")
    lines = f.readlines()
    f.close()
    for i,line in enumerate(lines):
        if ( -1 != line.find("FUNCTION=INFLOW, DIMENSION=2") ):
            lines[i+1] = "      " + mod + "*cos(0/180*pi(1))\n"
            lines[i+2] = "      " + mod + "*sin(0/180*pi(1))\n"
    f = open(kerFile, "w")
    f.writelines(lines)
    f.close()


# sets the porosity of the surrogate material mesh in the .ker.dat file

def setPorosity(kerFile, poro):
    # Store file lines as list of strings
    f = open(kerFile, "r")
    lines = f.readlines()
    f.close()
    # Search for the line defining porosity and modify it with the 'poro' value
    found = False
    for i,line in enumerate(lines):
        if ( -1 != line.find("POROSITY:") ):
            lines[i] = "      " + "POROSITY:  CONSTANT: VALUE=" + poro + "\n"
            found = True
    if not found: print("No POROSITY definition found")
    # Rewrite the whole file including the modified line
    f = open(kerFile, "w")
    f.writelines(lines)
    f.close()


# sets the variable values for the 'case' inside 'folder'
    
def setVars(fold, case, ang, vel, poro=0):
    angS = str(round(ang, 6))                           # angle in degrees as a string
    velS = str(round(vel, 6))                           # velocity as a string
    porS = str(round(poro, 6))                          # porosity as a string
    fileName = "./" + fold + "/" + case + ".ker.dat"    # file to be modified
    setVAngle(fileName, angS)                           # update angle in the .ker.dat file
    setVmod(fileName, velS)                             # update velocity mod in the .ker.dat file
    setPorosity(fileName, porS)                         # update porosity in the .ker.dat file

    
# Processes the file case.msh inside fold and prepares it for alya 

def getMesh(fold, case):
    # Check for existence of file 'case.geo' and 'case.msh' inside of 'fold'
    geoFileIs = os.path.exists(fold + "/" + case + ".geo")
    mshFileIs = os.path.exists(fold + "/" + case + ".msh")
    if ( not geoFileIs and not mshFileIs ):
        # There is no geometry related files in the 'fold' directory
        sys.exit("No geometry necessary files inside " + fold)
    elif ( geoFileIs and not mshFileIs ):        
        # Creates file case.msh from case.geo using the program gmsh in 2D with format msh2
        print("Creating " + case + ".msh from " + case + ".geo")
        subprocess.run(["gmsh", case + ".geo", "-2", "-format", "msh2"], cwd=fold)  
        
    mshFileIs = os.path.exists(fold + "/" + case + ".msh")
    if ( mshFileIs ):
        # process file case.msh into case.dims.dat, case.fix.bou, case.geo.dat
        subprocess.run(["../gmsh2alya.pl", case, "--bcs=boundaries", "--bulkcodes"], cwd=fold)
    else:
        sys.exit("File " + case + ".msh could not be found or generated")


# Run alya for case inside folder and postprocess the data obtained

def runAlya(fold, case, output=False, HPC=False):

    # HPC environment modifications
    if HPC:
        exe = "mpiexec"         # mpirun was giving me problems in Marenostrum
        numProc = "1"           # one process per simulation          
    else:
        exe = "mpirun"
        numProc = "4"

    # stdout for Alya is captured to avoid too much clogging unless specified otherwise
    if output:
        subprocess.run([exe, "-np", numProc, "../../bin/alya", case], cwd=fold)
    else:
        subprocess.run([exe, "-np", numProc, "../../bin/alya", case], stdout=subprocess.PIPE, cwd=fold)

    # processes the ouput data from alya inside 'fold' and 'fold/UPW', 'fold/DOWNW'
    subprocess.run(["cp", "./" + case + ".post.alyadat", 
                    "./UPW/" + case + "-UPW.post.alyadat"], cwd=fold)
    subprocess.run(["cp", "./" + case + ".post.alyadat", 
                    "./DOWNW/" + case + "-DOWNW.post.alyadat"], cwd=fold)
    subprocess.run(["../../bin/alya2pos", case], cwd=fold)
    subprocess.run(["../../../bin/alya2pos", case + "-UPW"], cwd=fold+"/UPW")
    subprocess.run(["../../../bin/alya2pos", case + "-DOWNW"], cwd=fold+"/DOWNW")


# Process and write the output WP meshes from alya run to './results' folder
        
def writeResults(fold, case, ang, vel, por):
    if ( not os.path.exists("./results") ):     # Create results folder
        subprocess.run(["mkdir", "results"])

    angStr = str(round(ang, 5))             # rotation angle as a string
    velStr = str(round(vel, 5))             # velocity as a string
    porStr = str(round(por, 5))             # porosity as a string
    
    # Get the witnessPoints data (velocity field) from 'fold/UPWP` and 'fold/DOWNWP'
    # and store it inside './results'
    for wpName in ["UPW", "DOWNW"]:
        # Process the ouptut WP mesh files into the wp class
        wp = witnessPointsRect(fold, case, wpName)
        # filename: ./results/folder-WP-mesh-ang-vel-por.txt
        fileName = ("./results/" + fold + "-" + wpName + "-" + angStr + "-" + 
                    velStr + "-" + porStr + ".txt")
        writeWpRect(wp, fileName)   # Write data inside 'wp' to 'fileName'    


# Runs a 'case' inside 'fold' with the angle 'ang' (degrees) and velocity 
# modulus 'vel' indicated (passed as strings). The value 'poro' is passed as
# porosity for the surrogate cases where it applies (can be set as 0 by default)

def runCase(fold, case, ang, vel, poro=0, output=False, HPC=False):
    # Create the mesh if it doesn't exist yet
    getMesh(fold, case)
    
    # Modify case.ker.dat to set values for velocity angle and modulus and porosity 
    setVars(fold, case, ang, vel, poro)     
    runAlya(fold, case, output, HPC)             # run alya and postprocess data
    writeResults(fold, case, ang, vel, poro)     # write captured data in './results'


# set the limits for angle (degrees), velocity (exponent of order of magnitude)
# and porosity (for the surrogate model simulations) that will be used when 
# running a case battery with 'runCaseBattery()'
    
def setLimits(ang1, ang2, vel1, vel2, por1=0, por2=0):
    global Lims
    Lims = [ang1, ang2, vel1, vel2, por1, por2]


# Runs a battery of alya simulations for 'case' inside 'fold' with 'nAng' angles between 
# 0 and 45 and 'nVel' velocity modulus between 0.01 and 0.1 by default, or the limits
# defined in the global variable 'Lims', set with 'setLimits()'.
# Writes the witnesspoints results in the results folder

def runCaseBattery(fold, case, nAng, nVel, nPor=1):

    global Lims   
    # Check for defined limit values or set them by default
    # Velocity limits are defined by log10(v) (order of magnitude)    
    if (not 'Lims' in globals()):   Lims = [0, 45, -2, 0, 0, 0]
    print(Lims)
    
    # Create lists for the variables (Angles and velocity order are trivial)
    angL        = np.linspace(Lims[0], Lims[1], nAng)
    velL        = np.linspace(Lims[2], Lims[3], nVel)
    # Porosity could include a 0, but the distribution is defined to be logaritmic.
    # A log2 based distribution from the last value is implemented.
    porL        = np.array([Lims[4]] + 
                [Lims[4]+(Lims[5]-Lims[4])/(2**(nPor-2-i)) for i in range(nPor-1)])
    
    ListVars    = [i.flatten() for i in np.meshgrid(angL, velL, porL, indexing='ij')]

    # Iterate for all the possible variables combinations
    for n in range(nAng * nVel * nPor):
        
        # get values for angle, velocity and porosity to use
        ang, vel, por = (ListVars[0][n], 10**ListVars[1][n], ListVars[2][n])
        angStr = str(round(ang,5))              # rotation angle as a string
        velStr = str(round(vel, 5))             # velocity as a string
        porStr = str(round(por, 5))             # porosity as a string
        
        print("-----------")                    # print progress of the run
        print("run: " + angStr + " , " + velStr + " , " + porStr)
        print("completion: " + str( round( n/(1.0*nAng*nVel*nPor)*100, 0 ) ) + "%")
        print("-----------")
        
        runCase(fold, case, ang, vel, por)      # run the case with ang and vel values
        
        subprocess.run(["./clean.sh"], cwd=fold)    # clean the directory

#%%
"""
Print help about commands
"""

def commands(*argv):
    helpFile = ("./analysis.help")      # Read contents of help file...
    f = open(helpFile, "r")
    lines = f.readlines()               # ...and store its lines
    f.close()
    
    # Print a list of commands if no argument is provided
    if ( not argv ):
        print("\nList of functions available")
        print("\nTo see more details about any of them use 'commands(name)'")
        print("\n\texample: commands(\"runCase\")")
        print("\n----------------------------------------\n")
        pDesc = False           # Print description flag
        for i,line in enumerate(lines):
            if ( -1 != line.find("FUNCTION") ):
                print("Function:\t\t", lines[i+1])
            elif ( -1 != line.find("DESCRIPTION") ):
                descStart = i + 1   # Start of description 1 line below DESCRIPTION
            elif ( -1 != line.find("EXAMPLE") ):
                descStop = i        # Stop description at EXAMPLE
                pDesc = True        # set print flag as true
            if ( pDesc ):
                desc = ""           # Store description lines here
                for l in lines[descStart:descStop]:
                    desc = desc + l
                print(desc)         # Print description
                print("----------------------------------------\n")
                pDesc = False       # Reset print flag as false
    # Give details on a particular command when name is passed as argument
    else:
        foundCom = False
        pParam = False
        # Get the details for the function which name is passed as first argument
        for i, line in enumerate(lines):
            if ( -1 != line.find(argv[0]+"(") and -1 != lines[i-1].find("FUNCTION") ):
                foundCom = True
                pParam = True
                print("\nFunction:\t\t", line)
            if ( -1 != line.find("PARAMETERS") and pParam):
                paramStart = i + 1
            if ( -1 != line.find("DESCRIPTION") and pParam):
                paramStop = i
                descStart = i + 1
            if ( -1 != line.find("EXAMPLE") and pParam):
                descStop = i
                examp = i + 1
                pParam = False
        if ( foundCom ):
            text = ""
            for l in lines[descStart:descStop]:
                text = text + l
            print(text)
            text = ""
            for l in lines[paramStart:paramStop]:
                text = text + l
            print(text)
            print("Example:\t\t" + lines[examp])
            print("----------------------------------------\n")
        else:
            print("That function name was not found.")



    
#%%
"""
Usage examples
"""
# Run the case 'matrix' inside 'matrix' folder with the following parameters
#   ang = 30                30 degrees of inclination
#   velOrd = 0              v = 10**0 = 1
#   poro =                  No porosity defined
#
#runCase("matrix", "matrix", 30, 0)
#
# Run the case 'matSurr' inside 'surrogate' folder with the following parameters
#   ang = 30                30 degrees of inclination
#   velOrd = 0              v = 10**0 = 1
#   poro = 100
#
#runCase("surrogate", "matSurr", 30, 0, 100)

# Set the limits for a battery of runs with the following values:
#   ang1 = 0 degrees        ang2 = 45 degrees
#   vel1 = 0.01 = 10**(-2)  -> velOrd1 = -2
#   vel2 = 1    = 10**(0)   -> velOrd2 = 0
#
#setLimits(0, 45, -2, 0)

# Run a battery of different velocity conditions (nAng and nVel) for the case
# 'matSurr' inside 'surrogate' folder. The limits for angles and velocity modulus
# are set with 'setLimits()'
#   nAng = 5
#   nVel = 10
#   poro = 300
#runCaseBattery("surrogate", "matSurr", 5, 10, poro=300)
