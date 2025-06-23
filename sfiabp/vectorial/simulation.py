from numpy import floor_divide, mod, sqrt
from numpy import cos, sin, sqrt, pi
import numpy as np

# import matplotlib.pyplot as plt

import time
import os

from sfiabp.vectorial import cellprocess

# from Lib_Abp_Sim import Lib_Abp_Sim_Force as Lib_BenchForce

# import pandas as pd
# import trackpy as tp


def sim ( npar, nfra, dtframe, dtinc, xlim, ylim, lcell, fun1p, fun2p, fundiff, **kwargs ):

    # optional parameters
    isim = kwargs.get('isim', 0)
    prer = kwargs.get('prer', 0)
    verbose = kwargs.get('verbose', False)
    frame_init = kwargs.get('frame_init', [])

    # chronometer
    tps0 = time.time() # print('SFI, tps0 = ',tps0)
    # pid of the processus
    pid = os.getpid()
    # seed random generator
    np.random.seed(isim)

    if 'verbose':
        print('sim, pid : ', pid, ', isim : ', isim)

    ### init simulation

    # init frame
    if isinstance(frame_init,np.ndarray): 
        X = frame_init.copy() 
    else : 
        X  = InitFrame(npar,xlim,ylim)
    # init simulation 
    simu1 = SimuABP( X, dtinc, xlim, ylim, lcell, fun1p, fun2p, fundiff )
    
    ### main loop

    #tlist = np.linspace(0., tau, nfra)
    ovsm = int(dtframe / dtinc) 
    list_X = np.zeros(np.concatenate(([nfra],np.shape(X))))
    timetab = np.zeros(nfra)
    simu1.RunData(prer)

    j = 0
    while j < nfra:
        if verbose:
            print( 'isim : ', isim, ', frame : ',j+1,'/',nfra )
        # time
        timetab[j] = time.time()
        # save data
        list_X[j,:,:] = simu1.f
        # run simulation
        simu1.RunData(ovsm)
        j += 1

    #ABP_SupFunc.time_analysis(timetab)
    # chronometer
    tps1 = time.time(); tps = tps1-tps0; 
    simu1.timetab = timetab
    #simu1.corel = corel
    
    if verbose:
        print( 'isim : ', isim, ' done, tps = ', tps, 's' )
    
    ## save
    # construct a dict dedicated to parameters
    psim = dict( npar = npar, nfra = nfra, dtframe = dtframe, dtinc = dtinc,
                 lcell = lcell, xlim = xlim, ylim = ylim, isim = isim, prer = prer )
    # construct the output dict
    data = dict( X_Raw = list_X, dtframe = dtframe, xframelim = np.array([0,xlim]),
                 yframelim = np.array([0,ylim]), time = tps, psim = psim)

    return data


def create_listpair(list_r, list_ai,list_aj, type_count):
    
    nar = len(list_r)
    nai = len(list_ai)
    naj = len(list_aj)
    grid = np.zeros((nar,nai,naj,4)) # r, thetai, thetaj, prob 
    grid[:,:,:,0], grid[:,:,:,1], grid[:,:,:,2] = np.meshgrid(list_r,list_ai,list_aj,indexing='ij')
    
    if type_count['name'] == 'uniform':
        grid[:,:,:,3] = np.ones((nar,nai,naj))*type_count['iter']  
    
    npair = np.sum(grid[:,:,:,3])
    list_pair = np.zeros((int(npair),3))

    l = 0
    for i in range(nar):
        for j in range(nai):
            for k in range(naj):
                cpt = int(grid[i,j,k,3])
                if cpt > 0:
                    tabrep = np.tile( grid[i,j,k,0:3], (cpt,1) )
                    list_pair[l:l+cpt,:] = tabrep 
                    l += cpt

    return list_pair


def ConvPol2Cart(r,ai,aj):
    x = r*np.cos(-aj)
    y = r*np.sin(-aj)
    theta = -aj+ai
    return x, y, theta 


def InitFrame(npar,xlim,ylim):

    # Initial frame (dim = 3)
    X = np.zeros((npar,3))
    # random coordiantes
    X[:,0] = np.random.uniform(0,xlim,(1,npar)) # x
    X[:,1] = np.random.uniform(0,ylim,(1,npar)) # y 
    X[:,2] = np.random.uniform(0,2*np.pi,(1,npar)) # theta

    return X


def collision_merge(Xbuf, Xbuf_next):
    
    Xbuf_new = np.copy(Xbuf)
    npar = len(Xbuf_new)
    i = 0
    while(i < npar):
        if Xbuf[i] == -1 and Xbuf_next[i] != -1 :
            Xbuf_new[i] = Xbuf_next[i]
        i += 1 

    return Xbuf_new 


###############################################################
# Class SimuABP 
###############################################################


class SimuABP:

    ## Class Constructor

    def __init__(self, X, dtinc, xlim, ylim, lcell, forcelist_i, forcelist_ij, forcediffu, **kwargs):
        
        # number of particles / dimension 
        self.npar, self.dim = np.shape(X)
        # increment time 
        self.ddt = dtinc
        # box size 
        self.blsize = np.array([ xlim, ylim, 2*np.pi ])
        # cell length
        self.lc = lcell
        # Mode Run
        self.Mode_Run = kwargs.get('Mode_Run', { 'Mode' : 'Simple'})

        # force parameters
        self.nfi = len(forcelist_i) 
        self.nfij = len(forcelist_ij)
        self.forcelist_i = forcelist_i.copy()
        self.forcelist_ij = forcelist_ij.copy()
        self.forcediffu = forcediffu.copy()

        # iteration index
        self.i = 0
        # iteration time
        self.t = 0
        
        # cell list initialization
        self.bcsize = np.divide(self.blsize,self.lc)
        self.nc = int(self.bcsize[0]*self.bcsize[1])    
        self.head = -1*np.ones(self.nc, dtype=int)
        # self.head = self.head.astype(int)
        self.table = -1*np.ones(self.npar, dtype=int)
        # self.table = self.table.astype(int)

        # Initial frame
        self.fi = X

        # constraint particles in the box
        self.fi[:,0] = np.mod(self.fi[:,0],self.blsize[0])
        self.fi[:,1] = np.mod(self.fi[:,1],self.blsize[1])
        self.fi[:,2] = np.mod(self.fi[:,2],self.blsize[2]) 

        # Current frame
        self.f = np.copy(self.fi)
        self.fbuf = np.zeros(np.shape(self.fi))
        self.fbufpair = np.zeros(np.shape(self.fi))
        self.data_neig = []
        self.data_neig_bol = []

    ## General function

    def loadnpy(self, str): # to modify
        # copy the current frame 
        self.f = np.copy(self.f)
    
    def load(self, frame): 
        # replace the current frame by frame
        self.f = np.copy(frame)

    def RunData (self, ovsm):
        
        i = 0
        # for each sub-frame            
        while i < ovsm: 
            #print("frame = ",i)
            # self.BrownianDynamics_N1v2()
            self.BrownianDynamics_N1v3()
            # self.BrownianDynamics_N1v1()
            # self.BrownianDynamics_N2v1()
            i += 1

    def RefreshNeighbors(self):
        
        npar = self.npar; head = self.head
        table = self.table; f = self.f; lc = self.lc
        blsize = self.blsize; bcsize = self.bcsize 
        nc = self.nc

        head = -1*np.ones(nc, dtype = int)
        table = -1*np.ones(npar, dtype = int)
        self.data_neig = [ [] for i in range(npar) ]
        # refresh head table before calcul cell
        cellprocess.calcul_cell(head, table, f, lc, bcsize)
        cellprocess.calcul_neig( self.data_neig, head, table, f, lc, 
                                                bcsize, blsize, periodic_bound = True )
           
    def BrownianDynamics_N1v3 (self):
            
            ## define alias

            # data buffer 1p, 2p
            self.fbuf[:] = np.zeros(np.shape(self.f))
            self.fbufpair[:] = np.zeros(np.shape(self.f)) 
            dX = np.zeros(3)
            f, fbuf, fbufpair = self.f, self.fbuf, self.fbufpair

            blsize, bcsize = self.blsize, self.bcsize
            nfi, nfij = self.nfi, self.nfij
            forcelist_i, forcelist_ij = self.forcelist_i, self.forcelist_ij
            forcediffu = self.forcediffu
            npar, nc, lc = self.npar, self.nc, self.lc
            ddt = self.ddt
            
            ##  update the cell list
            ## calcul the neighbors            
            self.RefreshNeighbors()
            
            ## update the dynamic

            ipar = 0
            while ipar < npar:
                
                # interaction 1-particle
                for ifi in range ( nfi ):
                    fbuf[ipar,:] += ddt*forcelist_i[ifi](f[ipar,:])
                # diffusive motion  
                fbuf[ipar,:] += forcediffu[0](f[ipar,:],ddt)
                
                # interaction 2-particle
                for jpar in self.data_neig[ipar]:
                    
                    # calcul the dx vector with periodic condition
                    dX[:] = f[ipar,:] - f[jpar,:]
                    dX[0] = min([dX[0]-1*blsize[0],dX[0],dX[0]+1*blsize[0]],key=abs)
                    dX[1] = min([dX[1]-1*blsize[1],dX[1],dX[1]+1*blsize[1]],key=abs)
                    dX[2] = min([dX[2]-1*blsize[2],dX[2],dX[2]+1*blsize[2]],key=abs)
                    # calcul the 2-norms
                    r = sqrt(dX[0]**2+dX[1]**2)
                    
                    p = 0
                    while p < nfij:

                        # force jpar sur ipar
                        fbufpair[ipar,:] += ddt*forcelist_ij[p](dX,f[ipar,:],f[jpar,:],r)
                        # if fbufpair[ipar,0] > 1:
                        #     print('s')
                        #     forcelist_ij[p](dX,f[ipar,:],f[jpar,:],r)
                        p += 1

                ipar += 1
      
            f += fbuf + fbufpair
            # constraint particles in the box
            f[:,0] = mod(f[:,0],blsize[0])
            f[:,1] = mod(f[:,1],blsize[1])
            f[:,2] = mod(f[:,2],blsize[2])  



