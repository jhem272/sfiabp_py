#################################
##     tutorial1_trigo.py      ##
#################################

import numpy as np
import dill
import os

import matplotlib.pyplot as plt

from sfiabp.base import base
from sfiabp.base import base2ptrigo
from sfiabp.general import addfunction
from sfiabp.vectorial import multicore
from sfiabp.display.sweep import sfidisp_sweep
from sfiabp.vectorial.analytics import function1rN

## general parameters
# The data to analyze is a set of synthetic frames of 82 particles per frame, 
# with an active velocity : 6 um/s, a rotational diffusion Dr = 0.1 /s^2 
# and a simple radial pair interaction of form v_er = 2000/r**4 um/s. 
PathDataFile = ('tutorial/data/Sim_1r4_npar_82_u_6_k_2000_5000f.pkl')
# starting frame 
istar = 0
# number of frame 
nfra = 200
# get the data
with open( PathDataFile, 'rb' ) as inp:    
    data0 = dill.load(inp)
list_data = data0['X_Raw'][istar:istar+nfra]

# To evaluate the algo performance in the presence of possible error detection 
# it is possible to blur the data with an additional GWN on the particle orientation 
stdGWN = 0 # standard deviation (deg) 
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
lcell = data0['lcell'] # 20
# Core number
ncore = 8

# choice of the drift basis functions
basis_name = 'Trigo'
if basis_name == 'Trigo': 
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
    # note fun1p also accepts the concatenated function base.stdfun1p(), such as fun1p = [ base.stdfun1p() ] 
    #fun1p = [ base.stdfun1p() ] 
    ## 2 particles force
    # Order, FuncRad, VectorRad
    Order = 1
    FuncRad = 'PolyExp'
    VectorRad = np.arange(0,16,2)
    lbase = base2ptrigo.polartrigo( Order, FuncRad, VectorRad )[0]
    # lbase is a list of trigonometric functions that depend on the coordinates 
    # (r,theta_1,theta_2) -> see notation in the draft
    # The next line convert lbase to a more relevant format for the algo :
    # a function that accepts the cartesian coordinates Xi, Xj of two particles 
    # and return dv the velocity for each basis function.  
    fun2p = [ base.convcart( lbase ) ]

# drift_mode = 'Ito'
drift_mode = 'Stratonovich'
# In case of pinv, the matrix inversion uses
# the pseudo inverse function (np.linalg.pinv)
inverse_mode = {'name':'pinv'}
# inverse_mode = {'name':'tiko','alpha':1e-5}

# Sub folder to save the ouput results 
SubDir = 'tutorial/tuto1'
os.makedirs(SubDir,exist_ok=True)
# output file
PathOutFile = SubDir + '/'  + 'S_%s_'%(basis_name) + os.path.basename(PathDataFile)[:-4] + \
                    '_%s_%s_Gwn_%d_Order_%d_%04df.pkl' % (drift_mode[0:6],inverse_mode['name'],stdGWN,Order,nfra)

#################################
##                             ##
#################################

# process sfi algo
cof1p, cof2p, D_average, psfi = multicore.sfi(list_data,dtframe,fun1p,fun2p,xboxlim,yboxlim,lcell,ncore,
                                                    drift_mode=drift_mode,inverse_mode=inverse_mode,histo_mode=True,verbose=True)

# the inferred coefficients cof1p, cof2p have same structures as fun1p, fun2p
# so the inferred active velocity is cof1p[2][0] and the coefficients for 
# the pair interaction are given by cof2p[0]
Dr = D_average[2,2]
print("inferred rotational diffusion, Dr=",Dr)
active_vel = cof1p[2][0]
print("inferred active velocity, U=",active_vel)

# In our case, we create a specific container 'Sabp' dict
# to collect all the relevant results of the test.
# The plotting function sfiabp/display/ works with this container.

Sabp = dict( cof1p = cof1p, cof2p = cof2p,
             active_vel = active_vel,
             D_average = D_average, psfi = psfi,
             data_file = os.path.basename(PathDataFile),
             frame_init = list_data[0],
             iid = np.array([istar,istar+nfra]),
             basis_name = basis_name, 
             Order = Order,  
             FuncRad = FuncRad,  
             VectorRad = VectorRad )

# save Sabp dict
with open( PathOutFile, 'wb') as outp:
    dill.dump( Sabp, outp )

#################################
##          plot               ##
#################################

# ## open S dict
# if 'Sabp' not in locals():
#     with open( PathOutFile, 'rb' ) as inp:    
#         Sabp = dill.load(inp)

## we now compare the inferred results with the exact pair interactions
# first get the analytical function
lfftheo = function1rN(2000,4)
# plot function
plt.ion()
# fig = sfidisp_sweep(Sabp, exact_fun = lfftheo, d = 3.17, rlim = [0,10], tjshift = -np.pi)
fig, _ = sfidisp_sweep(SubDir, exact_fun = lfftheo, d = 3.17,
                                rlim = [1,10], tishift = 0, tjshift = -np.pi, Prefix = 'S_' )
# FigManager = plt.get_current_fig_manager()
# FigManager.full_screen_toggle()
plt.show(block=True)
print('ok')


