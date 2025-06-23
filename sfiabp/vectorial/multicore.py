# import matplotlib.pyplot as plt
import multiprocessing as mp
from pathos.pools import _ProcessPool

import numpy as np
import time
import dill
import copy
import os
 
from sfiabp.vectorial import inference
from sfiabp.vectorial import trajectorydata

 
def sfi(list_data,dtframe,fun1p,fun2p,xboxlim,yboxlim,lcell,ncore,**kwargs):

    """ Description :Stochastic Force Inference algorithm
    - Process a list of data file (.pkl) 
    - The data is a dictionnary that must include the 
    following keys :
        'dtframe' : float, time interval between two frames
        'X_Raw' : numpy array, frame X n_particle X (x,y,theta,identity_particle)
    Libraries required : 
        conda 23.11.0; python 3.9.18, dill 0.3.7
        numpy 1.24.3; multiprocess 0.70.15 """

    #### creation of dict Param_SFI ####

    verbose = kwargs.get('verbose', False)
    Param_Sfi = dict( ncore = ncore, nproc = kwargs.get('nproc', ncore) )
    
    # box limits 
    boxsize = np.array([ xboxlim, yboxlim, [0,2*np.pi] ])
    blsize = np.diff(boxsize).flatten()
    
    if np.mod(blsize[0],lcell) != 0 or np.mod(blsize[1],lcell) != 0:
        raise KeyError('lcell not a divisor of the box spatial dimensions')
    
    ## optional parameters
    # dirft mode : 'Ito', 'Stratonovich', 'Ito-shift', 'Stratonovich-old', 
    Param_Sfi['drift_mode'] = kwargs.get('drift_mode', 'Stratonovich')
    # inverse_mode : 'pinv', 'tikhonov'
    Param_Sfi['inverse_mode'] = kwargs.get('inverse_mode', {'Name':'pinv'})
    # stro_diff_mode : 'Vestergaard', 'ABP_Vestergaard', 'ABP_CST'
    Param_Sfi['strato_diff_mode'] = kwargs.get('strato_diff_mode', 'ABP_Vestergaard')
    Param_Sfi['strato_diff_matrix'] = kwargs.get('strato_diff_matrix', np.array([[0,0,0],[0,0,0],[0,0,1]]))
    # Param_Sfi['strato_diff_mode'] = kwargs.get('strato_diff_mode', 'ABP_CST')
    # Param_Sfi['strato_diff_matrix'] = kwargs.get('strato_diff_matrix', np.array([[0,0,0],[0,0,0],[0,0,0.1]]) )
    # enable edge_filter
    Param_Sfi['edge_filter'] = kwargs.get('edge_filter', True)

    # B_mode : 'rectangle','trapeze','trapeze-shift'
    Param_Sfi['B_mode'] = 'rectangle'
    # diffusion_mode : 'Vestergaard', 'MSD', 'WeakNoise'
    Param_Sfi['diffusion_mode'] = 'Vestergaard'
    ## histogram parameters
    Param_Sfi['histo_mode'] = kwargs.get('histo_mode', False)
    Param_Sfi['histo_vecr'] = kwargs.get('histo_vecr', np.linspace(0,lcell,2*lcell+1,endpoint=True) )
    Param_Sfi['histo_veca'] = kwargs.get('histo_veca', (np.pi/180)*np.linspace(0,360,36+1,endpoint=True) )

    #### top statistics ####

    tps0 = time.time() # print('SFI, tps0 = ',tps0)  
    nproc = Param_Sfi['nproc']
    nfra = len(list_data)
    nfproc = np.floor_divide( (nfra+2*(nproc-1)), nproc ) # frame number per process
    
    if nfproc < 3:
        raise KeyError('the frame number per process is too low')
    
    iind_dif = np.ones(nproc)*nfproc   
    nerr = int(nfra - (nfproc*nproc-2*(nproc-1)))
    iind_dif[0:nerr] += 1
    iind = np.hstack( ([0], np.cumsum(np.ones(nproc)*(iind_dif-2))[:-1]) )
    
    if ncore == 1 :
        list_S = [ [] for i in range(nproc) ]
        for i in range(nproc):
            # process sfi  
            list_S[i] = SingleProc_S( list_data[int(iind[i]):int(iind[i])+int(iind_dif[i])],
                                            dtframe, fun1p, fun2p, boxsize, blsize, lcell, verbose, Param_Sfi, i, )

    elif ncore > 1 : # mode multi core
        # creat pool
        # pool = mp.Pool(processes=P['ncore'])
        pool = _ProcessPool(processes=Param_Sfi['ncore'])
        list_res = [ [] for i in range(nproc) ]
        # start pool process
        for i in range(nproc):
            # process sfi
            list_res[i] = pool.apply_async( SingleProc_S, args = ( list_data[int(iind[i]):int(iind[i])+int(iind_dif[i])],
                                                            dtframe, fun1p, fun2p, boxsize, blsize, lcell, verbose, Param_Sfi, i, ) )
            # res = pool.apply_async( f, args = ( 6, ) )
            # print(res.get(timeout=1))  
        # close pool process
        pool.close()
        pool.join()
        list_S = [ list_res[i].get() for i in range(nproc) ]

    S0 = copy.deepcopy(list_S[0])

    # if only S0 file -> no combination
    if nproc > 1 :
        iidproc = np.arange(1,nproc)
        for i in iidproc:
            # only S0 file (no combination)
            S0.additionS(list_S[i])
            if verbose:
                print('combination S',i)
    
    # end chronometer
    tps1 = time.time()
    tpstot = tps1-tps0 

    if verbose:
        print('processing time (s):',tpstot)
    # print information
    # S0.print_report()

    #### different outputs according to inverse_mode ####

    ## main parameters
    Param_Sfi['lcell'] = lcell # cell length (um) 
    Param_Sfi['xboxlim'] = xboxlim
    Param_Sfi['yboxlim'] = yboxlim
    Param_Sfi['boxsize'] = boxsize # box limits (um)
    Param_Sfi['blsize'] = blsize # box widths (um)
    Param_Sfi['dtframe'] = dtframe

    # results sfi
    Param_Sfi['time'] = tpstot
    Param_Sfi['frame_weight'] = S0.frame_weight
    Param_Sfi['stat_weight'] = S0.stat_weight
    Param_Sfi['pair_stat_weight'] = S0.pair_stat_weight
    Param_Sfi['Lambda'] = np.copy(S0.Lambda)
    Param_Sfi['histo'] = np.copy(S0.histo)

    ## process drift / different possible output format
    if 'opt' not in Param_Sfi['inverse_mode']:
        Param_Sfi['inverse_mode']['opt'] = 'v1' 
    inv_mode = Param_Sfi['inverse_mode'] 
    
    if inv_mode['name'] == 'pinv':
        
        # process the inverse
        inv_mode['conditional_number'] = S0.pinv()
        # add cof1p, cof2p
        cof1p, cof2p = cof1p2p(S0.phi_coefficients,fun1p,fun2p)
        # return the results
        return cof1p, cof2p, S0.D_average, Param_Sfi
    
    elif inv_mode['name'] == 'tiko' and isinstance(inv_mode['alpha'],(int,float)):
        
        # process the inverse
        inv_mode['nx'], inv_mode['na']  = S0.tiko(inv_mode['alpha'],inv_mode['opt'])
        # add cof1p, cof2p
        cof1p, cof2p = cof1p2p(S0.phi_coefficients,fun1p,fun2p)
        # return the results
        return cof1p, cof2p, S0.D_average, Param_Sfi
    
    elif inv_mode['name'] == 'tiko' and isinstance(inv_mode['alpha'],np.ndarray):
        
        nalpha = len(inv_mode['alpha'])
        inv_mode['list_nx'] = [ [] for i in range(nalpha)]
        inv_mode['list_na'] = [ [] for i in range(nalpha)]
        list_cof1p = [ [] for i in range(nalpha)]
        list_cof2p = [ [] for i in range(nalpha)]

        for i,alpha in enumerate(inv_mode['alpha']):
            # process the inverse
            inv_mode['list_nx'][i], inv_mode['list_na'][i]  = S0.tiko(alpha,inv_mode['opt'])
            # invmod.update(nx=nx,na=na,list_alpha=list_alpha,list_phicof=list_phicof)
            # add cof1p, cof2p
            list_cof1p[i], list_cof2p[i] = cof1p2p(S0.phi_coefficients,fun1p,fun2p)

        return list_cof1p, list_cof2p, S0.D_average, Param_Sfi
    
    else:

        KeyError('Wrong inverse_method argument, possible choice : -pinv-, -tiko-, -tiko_auto- ')
    

def SingleProc_S ( list_data, dtframe, fun1p, fun2p, boxsize, blsize, lcell, verbose, Param_Sfi, i ):
    
    #### preprocess ####

    pid = os.getpid()

    if verbose:
        print( 'pid:', pid, ', process:',i, ', pre process ...')

    # create data object
    list_data = trajectorydata.PreProcess( list_data, boxsize )
    # convert data to SFI format
    dn = 1 # interval to compute the frame derivative 
    Dn = 1 # interval between consecutive frame 
    # if dn, Dn = 1 then number of frames processed = nfra-2
    data = trajectorydata.StochasticTrajectoryData( list_data, dn, Dn, dtframe, boxsize, blsize, lcell )
    # set X_probe == 0 for particles close to the box edges
    if Param_Sfi['edge_filter']:
        data.Probe_RemoveEdge()
    
    #### process SFI #### 

    if verbose:
        print( 'process:',i, ', process sfi ...')

    S = inference.StochasticForceInference( fun1p, fun2p, Param_Sfi )

    # for each frame
    for j in range(data.N):
        
        # slice data (1 by 1 frame processed)
        data1by1 = data.SliceData(j,1) 
        # format data 
        data1by1.MidPoint()
        # neighbor process
        data1by1.CellProcess()
        # add statistics
        S.sfi_addstat(data1by1)
        # compute histogram (polar coordinate)
        # np.sum(histo) == pair_stat_weight
        S.histo_polar_addstat(data1by1)
        
        if verbose:
            print( 'process:',i, ', frame:', j+1, '/',data.N)
        
    delattr(S,'data')
    delattr(S,'bxt_i') 
    delattr(S,'bxt_s')
    delattr(S,'bxt_isom')
    delattr(S,'bxt_isop')

    S.i = i

    return S
    

def cof1p2p(phi_coefficients,fun1p,fun2p):

    cof1p = []
    cof2p = []
    ccur = 0
    for i in range(len(fun1p)):
        if fun1p[i](np.array([0,0,0])).ndim == 1: n1p = 1
        else: n1p = np.shape(fun1p[i](np.array([0,0,0])))[0]
        cof1p.append( phi_coefficients[ccur:ccur+n1p] )
        ccur += n1p
    for i in range(len(fun2p)):
        if fun2p[i](np.array([0,0,0]),np.array([0,0,0])).ndim == 1: n2p = 1
        else: n2p = np.shape(fun2p[i](np.array([0,0,0]),np.array([0,0,0])))[0]
        cof2p.append( phi_coefficients[ccur:ccur+n2p] )
        ccur += n2p
    
    return cof1p, cof2p