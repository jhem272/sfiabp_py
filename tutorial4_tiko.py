#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

""" very noisy environemnt"""
""" process multiple list of S file, frame number and alpha tikhonv """

import matplotlib.pyplot as plt 
import numpy as np
import dill
import copy
import os

from sfiabp.base import base
from sfiabp.base import base2ptrigo
from sfiabp.general import addfunction
from sfiabp.vectorial import multicore
from sfiabp.display.sweep import sfidisp_sweep
from sfiabp.vectorial.analytics import minimalmodel3
from sfiabp.vectorial.analytics import function1rN

#################################
##                             ##
#################################

## general parameters
# data file
PathDataFile = ('tutorial/data/Ominimal_model3_small_5000f.pkl')
# PathDataFile = ('tutorial/data/Sim_1r4_npar_82_u_6_k_2000_5000f.pkl')
# starting frame 
istar = 0
# number of frame 
nfra = 10
# get the data
with open( PathDataFile, 'rb' ) as inp:    
    data0 = dill.load(inp)
list_data = data0['X_Raw'][istar:istar+nfra]

# additional GWN on orientation 
# standard deviation (deg)
stdGWN = 10
# add noise 
addfunction.addGWnoise(list_data,stdGWN*(np.pi/180))

## parameters for sfi algo
# time between two frames (s)
dtframe = data0['dtframe']
# box dimension along x, where to perform the sfi 
xboxlim = data0['xframelim'] #
# box dimension along y, where to perform the sfi 
yboxlim = data0['yframelim'] # 
# cell length (um) for the sfi algo
lcell = data0['lcell']
# core number
ncore = 3

# choice of the drift basis functions
basis_name = 'Trigo'
if basis_name == 'Trigo': 
    # 1 particle function
    fun1p = [ base.stdfun1p() ] 
    # 2 particles function
    # Order, FuncRad, VectorRad
    Order = 2
    FuncRad = 'PolyExp'
    VectorRad = np.arange(0,16,2)
    lbase = base2ptrigo.polartrigo( Order, FuncRad, VectorRad )[0]
    fun2p = [ base.convcart( lbase ) ]

# drift mode : 'Ito', 'Stratonovich', 'Ito-shift', 'Stratonovich-old' 
# drift_mode = 'Ito'
drift_mode = 'Stratonovich'
# instead of pinv, we now use the tikonov inverse
# we must provide the regularization coefficient alpha 
# inverse_mode = {'name':'pinv'}
# inverse_mode = {'name':'tiko','alpha':1e-5}
# inverse_mode = {'name':'tiko','alpha': np.array([0,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10])}
inverse_mode = {'name':'tiko','alpha': np.array([0,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10]),'opt':'v1'}

# Sub folder to save the ouput results 
SubDir = 'tutorial/tuto4'
os.makedirs(SubDir,exist_ok=True)
# output file
PathOutFile = SubDir + '/'  + 'List_S_Trigo_' + os.path.basename(PathDataFile)[:-4] + \
                '_%s_%sv2_Gwn_%d_Order_%d_%df.pkl' % (drift_mode[0:6],inverse_mode['name'],stdGWN,Order,nfra)

#################################
##                             ##
#################################

# process sfi algo
# since alpha is a np.array, the sfi algo performs the inverse operation several times
# the inferred coefficients are provided in two lists, list_cof1p and list_cof2p  
list_cof1p, list_cof2p, D_average, psfi = multicore.sfi(list_data,dtframe,fun1p,fun2p,xboxlim,yboxlim,lcell,ncore,
                                                drift_mode=drift_mode,inverse_mode=inverse_mode,histo_mode=True,verbose=True)
    
list_Sabp = []   
for i, alpha in enumerate(psfi['inverse_mode']['alpha']):
    
    psfi_i = copy.deepcopy(psfi) 
    psfi_i['inverse_mode']['alpha'] = alpha
    Sabp_i = dict(  cof1p = list_cof1p[i], cof2p = list_cof2p[i],
                    D_average = D_average, psfi = psfi_i,
                    data_file = os.path.basename(PathDataFile),
                    frame_init = list_data[0],
                    iid = np.array([istar,istar+nfra]),
                    basis_name = basis_name, 
                    Order = Order,  
                    FuncRad = FuncRad,  
                    VectorRad = VectorRad )
    
    list_Sabp.append(Sabp_i)

# save list_Sabp
with open( PathOutFile, 'wb') as outp:
    dill.dump( list_Sabp, outp )

#################################
##                             ##
#################################

#### plot ####

# # get the analytical 3d function
# # lfftheo = function1rN(2000,4)
# lfftheo = minimalmodel3() 

# # fig = sfidisp_sweep(SubDir, exact_fun = lfftheo)
# fig = sfidisp_sweep(SubDir, exact_fun = lfftheo, d = 3.17,)
# # fig.canvas.manager.frame.Maximize()
# FigManager = plt.get_current_fig_manager()
# FigManager.full_screen_toggle()
# plt.show(block=True)

print('ok')

