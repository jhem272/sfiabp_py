#################################
##     tutorial3_flock.py      ##
#################################

import matplotlib.pyplot as plt
import numpy as np
import dill
import os

from sfiabp.base import base
from sfiabp.base import base2ptrigo
from sfiabp.general import addfunction
from sfiabp.vectorial import multicore
from sfiabp.display.sweep import sfidisp_sweep
from sfiabp.display.versus import sfidisp_versus
from sfiabp.vectorial.analytics import minimalmodel3

## general parameters
# The data to analyze is the dataset 'data/Ominimal_model3_small_5000f.pkl' 
# that simulates the dynamic of N=500 flocking particles. 
# data file
PathDataFile =  ('tutorial/data/Ominimal_model3_small_5000f.pkl')
# starting frame 
istar = 0
# number of frame 
nfra = 50
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
dtframe = data0['dtframe'] # 0.1
# box dimension (um) along x, where to perform the sfi 
xboxlim = data0['xframelim'] #  np.array([0,200])
# box dimension (um) along y, where to perform the sfi 
yboxlim = data0['yframelim'] #  np.array([0,200])
# cell length (um) for the sfi algo
# must be a common divisor of xboxlim and yboxlim
lcell = data0['lcell'] # 20
# core number
ncore = 5

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
    ## 2 particles force
    # Order, FuncRad, VectorRad
    Order = 2
    FuncRad = 'PolyExp'
    VectorRad = np.arange(0,16,2)
    lbase = base2ptrigo.polartrigo( Order, FuncRad, VectorRad )[0]
    fun2p = [ base.convcart( lbase ) ]

# drift_mode = 'Ito'
drift_mode = 'Stratonovich'
inverse_mode = {'name':'pinv'}
# inverse_mode = {'name':'tiko','alpha':1e-5}

# Sub folder to save the ouput results 
SubDir = 'tutorial/tuto3'
os.makedirs(SubDir,exist_ok=True)
# output file
PathOutFile = SubDir + '/'  + 'S_%s_'%(basis_name) + os.path.basename(PathDataFile)[:-4] + \
                    '_%s_%s_Gwn_%d_%04df.pkl' % (drift_mode[0:6],inverse_mode['name'],stdGWN,nfra)

#################################
##                             ##
#################################

# process sfi algo
cof1p, cof2p, D_average, psfi = multicore.sfi(list_data,dtframe,fun1p,fun2p,xboxlim,yboxlim,lcell,ncore,
                                                    drift_mode=drift_mode,inverse_mode=inverse_mode,histo_mode=True,verbose=True)

Dr = D_average[2,2]
print("inferred rotational diffusion, Dr=",Dr)
active_vel = cof1p[2][0]
print("inferred active velocity, U=",active_vel)

# create Sabp dict specific to our study
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
##                             ##
#################################

## we now compare the inferred results with the exact pair interactions
# first get the analytical function
lfftheo = minimalmodel3()
# plot tutorial
plt.ion()
fig, _ = sfidisp_sweep(SubDir, exact_fun = lfftheo, d = 3.17, rlim = [1,10],
                                   tishift=0, tjshift = -np.pi, Prefix = 'S_' )
# fig, _ = sfidisp_versus(SubDir, 'tutorial/tuto1', d = 3.17, rlim = [1,10], 
#                                      tishift=0, tjshift = -np.pi, Prefix = 'S_' )
# FigManager = plt.get_current_fig_manager()
# FigManager.full_screen_toggle()
plt.show(block=True)
print('ok')

