#################################
##     tutorial2_mesh.py      ##
#################################

import matplotlib.pyplot as plt
import numpy as np
import dill
import os

from sfiabp.base import base
from sfiabp.base import base2pmesh
from sfiabp.general import addfunction
from sfiabp.vectorial import multicore
from sfiabp.display.sweep import sfidisp_sweep
from sfiabp.display.versus import sfidisp_versus
from sfiabp.vectorial.analytics import function1rN

## general parameters
# The data to analyze is the same as in tutorial 1
# data file
PathDataFile = ('tutorial/data/Sim_1r4_npar_82_u_6_k_2000_5000f.pkl')
# starting frame 
istar = 0
# number of frame 
nfra = 400
# get the data
with open( PathDataFile, 'rb' ) as inp:    
    data0 = dill.load(inp)
list_data = data0['X_Raw'][istar:istar+nfra]

# additional GWN on orientation 
# standard deviation (deg)
stdGWN = 0
# add noise 
addfunction.addGWnoise(list_data,stdGWN*(np.pi/180))

## parameters for sfi algo
# time between two frames (s)
dtframe = data0['dtframe']
# box dimension (um) along x, where to perform the sfi 
xboxlim = data0['xframelim'] #  np.array([0,320])
# box dimension (um) along y, where to perform the sfi 
yboxlim = data0['yframelim'] #  np.array([0,320])
# cell length (um) for the sfi algo
# must be a common divisor of xboxlim and yboxlim
lcell = data0['lcell']
# core number
ncore = 8

# choice of the drift basis functions
basis_name = 'Mesh'
if basis_name == 'Mesh': 
    ## 1 particle force 
    def cstx(k = 1): # constant velocity along x
        return lambda X :  np.array([ k*1, 0, 0 ]) 
    def csty(k = 1): # constant velocity along y
        return lambda X :  np.array([ 0, k*1, 0 ])
    def active(U = 1): # active velocity
        return lambda X : np.array([ U * np.cos(X[2]), U * np.sin(X[2]), 0 ])
    def active_perp(U = 1): # perpendicular active velocity
        return lambda X : np.array([ U * np.sin(X[2]), -U * np.cos(X[2]),0])
    fun1p = [ cstx(), csty(), active(), active_perp() ]
    ## 2 particles function
    # The 'mesh' basis requires to specify the grid in coordinates (r,theta_1,theta_2)
    # A function is related to each bin of the grid. 
    # radial bin
    vecr = np.array([3,4,5,6,8,11,15,20])
    # angular bin
    veca = np.linspace(0,360,6)*np.pi/180
    # The bin function is here a simple 'Step' function
    FuncKern = 'Step'
    lbase = base2pmesh.polarmesh( vecr, veca, veca, FuncKern )[0]
    fun2p = [ base.convcart(lbase) ]

# drift_mode = 'Ito'
drift_mode = 'Stratonovich'
inverse_mode = {'name':'pinv'}
# inverse_mode = {'name':'tiko','alpha':1e-5}

# Sub folder to save the ouput results 
SubDir = 'tutorial/tuto2'
os.makedirs(SubDir,exist_ok=True)
# output file
PathOutFile = SubDir + '/'  + 'S_%s_'%(basis_name) + os.path.basename(PathDataFile)[:-4] + \
                    '_%s_%s_Gwn_%d_%df.pkl' % (drift_mode[0:6],inverse_mode['name'],stdGWN,nfra)

#################################
##                             ##
#################################

# The function multicore.sfi processes, at the same time and by default, an histogram 
# for the distribution of neighbors around a particle. It allows to assess the reliability of the inference. 
# In case of a basis of type 'mesh', it is useful to set the grid of the mesh identical to the grid of the histogram 
# also to assess the reliability of each bin (see parameters histo_mode, histo_vecr, histo_veca ).

# process sfi algo
cof1p, cof2p, D_average, psfi = multicore.sfi(list_data,dtframe,fun1p,fun2p,xboxlim,yboxlim,lcell,ncore,
                                                    drift_mode=drift_mode,inverse_mode=inverse_mode,verbose=True,
                                                        histo_mode=True, histo_vecr=vecr, histo_veca=veca)

Dr = D_average[2,2]
print("inferred rotational diffusion, Dr=",Dr)
active_vel = cof1p[2][0]
print("inferred active velocity, U=",active_vel)

# create Sabp dict specific to our study
Sabp = dict( cof1p = cof1p, cof2p = cof2p,
            active_vel = cof1p[2][0],
            D_average = D_average, psfi = psfi,
            data_file = os.path.basename(PathDataFile),
            frame_init = list_data[0],
            iid = np.array([istar,istar+nfra]),
            basis_name = basis_name, 
            FuncKern = FuncKern,
            vecr = vecr,  
            veca = veca )

# save Sabp dict
with open( PathOutFile, 'wb') as outp:
    dill.dump( Sabp, outp )

#################################
##                             ##
#################################

## we now compare the inferred results with the exact pair interactions
# first get the analytical function
lfftheo = function1rN(2000,4)
# mask_thres parameter, available only if basis_name=='mesh': 
# if the histogram gives a count number below this threshold 
# the coeffecient is set to zero. This improves the readibility of the 
# graph by removing results with poor statistics.
mask_thres = 100
# plot tutorial
plt.ion()
# fig = sfidisp_sweep(SubDir, exact_fun = lfftheo, d = 3.17, rlim = [0,10],
#                                    tishift=0, tjshift = -np.pi, mask_thres = mask_thres, Prefix = 'S_' )
fig, _ = sfidisp_versus(SubDir, SubDir, d = 3.17, rlim = [1,10], 
                                     tishift=0, tjshift = -np.pi, mask_thres = mask_thres, Prefix = 'S_' )
# FigManager = plt.get_current_fig_manager()
# FigManager.full_screen_toggle()
plt.show(block=True)
print('ok')

