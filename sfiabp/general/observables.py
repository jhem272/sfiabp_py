
import multiprocessing as mp
import matplotlib.pyplot as plt

import numpy as np
import dill 
import copy
import os


def constraint_frame( list_X_ref, xboxlim, yboxlim ):
         
    ## pre process
    nfra = len(list_X_ref)
    # list_X = copy.deepcopy(list_X_ref)
    list_X = [ [] for i in range(nfra) ]
    # list_X = [ list_X_ref for i in range(N) ]
    dtheta = 2*np.pi

    for i in range(nfra):
        # remove particles found outside the box
        bolx = np.logical_and( xboxlim[0] <= list_X_ref[i][:,0], list_X_ref[i][:,0] < xboxlim[1] )
        boly = np.logical_and( yboxlim[0] <= list_X_ref[i][:,1], list_X_ref[i][:,1] < yboxlim[1] )
        iid = np.where( np.logical_and(bolx,boly) == False )[0]    
        list_X[i] = np.delete(list_X_ref[i],iid,axis=(0))
        # origin (0,0)
        list_X[i][:,0] = list_X[i][:,0] - xboxlim[0]
        list_X[i][:,1] = list_X[i][:,1] - yboxlim[0]
        # constraint orientation
        list_X[i][:,2] = np.mod(np.mod(list_X[i][:,2],dtheta),dtheta)
    
    return list_X, np.diff(xboxlim)[0], np.diff(yboxlim)[0]


def cleanframe(list_frame,tab):
            npar = len(tab)
            list_falsepart = []
            for i in range(npar):
                flag = False
                for j in range(int(tab[i,2])):
                    list_ipar = np.where( list_frame[ int(tab[i,1]) +j ] == tab[i,0] )[0]
                    if len(list_ipar)==0:
                        flag = True
                if flag:
                    list_falsepart.append(i)
            # delete false particle
            tab = np.delete(tab,list_falsepart,axis=0)
            return tab


def uniqueframe(list_frame):
            # init tab
            # 0- id, 1- index, 2-counts  
            tab = np.zeros((len(list_frame[0]),3))
            tab[:,0] = list_frame[0][:,3]
            tab[:,2] = np.ones(len(list_frame[0]))
            # for loop
            for i_frame, frame in enumerate(list_frame):
                if i_frame>0:
                    for id in frame[:,3]:
                        if id in tab[:,0]:
                            index = np.where(tab[:,0]==id)[0][0]
                            tab[int(index),2] += 1
                        else:
                            add_row = np.array([[id,i_frame,1]])
                            tab = np.append(tab,add_row,axis=0)
            
            # sort along 2-count column
            tab = tab[ tab[:,2].argsort() ]
            return tab 


def collect_traj_exp( list_frame, minthres ):

    # identity table for particles 
    # 0-id, 1-frist index occurence, 2-count
    tab = uniqueframe(list_frame)
    tab = cleanframe(list_frame,tab)
    tab = tab[ tab[:,2]>=minthres, : ]

    # collect trajectories
    npar = len(tab) 
    list_traj = [ [] for i in range(npar) ]
    
    for i in range(npar):
        # collect traj
        nframe_i = int(tab[i,2])
        traj = np.zeros((nframe_i,3))
        for j in range(nframe_i):
            frame = list_frame[int(tab[i,1])+j]
            ipar = np.where( frame[:,3] == tab[i,0] )[0][0]
            traj[j,:] = frame[ipar,:3]
        list_traj[i] = copy.deepcopy(traj)

    return list_traj


def find_divisor(xlim):
        list_divisor = []
        for i in np.linspace(1,xlim,xlim):
            if xlim % i == 0:
                list_divisor.append(i) 
        return np.array( list_divisor )


def find_common_divisor(xlim,ylim):
        list_divisor = []
        for i in np.linspace(1,np.min([xlim,ylim]),np.min([xlim,ylim])):
            if (xlim % i == 0) and (ylim % i == 0) :
                list_divisor.append(i) 
        return np.array( list_divisor )


def find_lcell(list_divisor,bd):
    if bd > list_divisor[-1]:
        raise KeyError('unable to find lcell, please change xboxlim, yboxlim')
    tab = list_divisor-bd  
    return list_divisor[np.where(tab>=0)[0][0]]


###########################################
        ## cell functions ##
###########################################


def create_cell(data, lc, blsize):
        # data orqanized as : frame, particle num, dim
        
        #bcsize = np.floor_divide(blsize,lc) # box size in cell unit
        bcsize = np.divide(blsize,lc) # box size in cell unit
        nc = int(bcsize[0]*bcsize[1]) # cell number
        nfra = len(data) # frame number, particle number
        Npar = [ len(data[i]) for i in range(nfra) ] 

        ## for each frame ##
        
        data_head = [ -1*np.ones(nc, dtype = int) for t in range(nfra) ]
        data_table = [ -1*np.ones(npar_t, dtype = int) for npar_t in Npar ]
        data_neig = [[ [] for i in range(Npar[t])] for t in range(nfra) ]
        # warning refresh head table beofre calcul cell
        
        # data_neig = [ -1*np.ones((npar,nmax), dtype = int) for i in range(nfra) ]

        for t in range (nfra):
            calcul_cell(data_head[t], data_table[t], data[t], lc, bcsize)
            calcul_neig(data_neig[t], data_head[t], data_table[t], data[t], lc, bcsize, blsize)

        # if isinstance(data,np.ndarray):
        #     data_head = -1*np.ones((nc,nfra), dtype = int)
        #     data_table = -1*np.ones((npar,nfra), dtype = int)

        #     for i in range (nfra):
        #         calcul_cell(data_head[:,i], data_table[:,i], data[i,:,:], lc, bcsize)

        return data_neig


def calcul_cell(head, table, frame2D, lc, bcsize):
        
        npar = len(table)
        i = 0
        while i < npar:

            # cell index
            # icl = int ( np.floor_divide(fbuf[i,1],lc)*bcsize[0] + np.floor_divide(fbuf[i,0],lc) )  
            icl = int ( np.floor_divide(frame2D[i,1],lc)*bcsize[0] + np.floor_divide(frame2D[i,0],lc) )

            if head[icl] == -1: 
                head[icl] = i
            else:
                table[i] = head[icl]
                head[icl] = i
            
            i += 1
        

def calcul_neig(table_neig, head, table, X, lc, bcsize, blsize, periodic_bound = False ):

        # X is a Nparticles x dim - shaped array.
        dX = np.zeros(3)
        #nc = len(head) # cell number
         # define the neighbour list
        neigtab = np.array([[0,0,0],[1,0,0],[1,-1,0],
                            [0,-1,0],[-1,-1,0],[-1,0,0],
                            [-1,1,0],[0,1,0],[1,1,0]])
        
        # periodic bound
        if periodic_bound:
            # for each cell 
            for icl, ipar in enumerate(head):    
                # if the cell is not empty
                if ipar != -1 :
                    yc = np.floor_divide(icl,bcsize[0]) # cell y-coordinate
                    xc = np.mod(icl,bcsize[0]) # cell x-coordinate
                    #ipar = head[icl] # header particle of icl-th cell
                    # for each particle in that cell
                    while(ipar != -1):
                        # create the neigbour list
                        list_inbc = []
                        for ilz in range(9):
                            inbc = int( np.mod(yc+neigtab[ilz,1],bcsize[1])*bcsize[0] + np.mod(xc+neigtab[ilz,0],bcsize[0]) )
                            list_inbc.append(inbc)
                        # for each neighbour cell index list 
                        for inbc in list_inbc:
                            jpar = head[inbc] # header particle of cell inbc-th
                            # for each particle in that neighbouring cell
                            while(jpar != -1):
                                if ipar != jpar: 
                                    # calcul the dx vector
                                    dX[:] = X[ipar,:3] - X[jpar,:3]
                                    dX[0] = min([dX[0]-1*blsize[0],dX[0],dX[0]+1*blsize[0]],key=abs)
                                    dX[1] = min([dX[1]-1*blsize[1],dX[1],dX[1]+1*blsize[1]],key=abs)
                                    dX[2] = min([dX[2]-1*blsize[2],dX[2],dX[2]+1*blsize[2]],key=abs)
                                    # calcul the 2-norms
                                    r = np.sqrt(dX[0]**2+dX[1]**2)
                                    # consider only particles closer than the cell length
                                    if r < lc:
                                        table_neig[ipar].append(jpar)
                                jpar = table[jpar]
                        ipar = table[ipar]
        
        # no periodic bound
        else :
            # for each cell 
            for icl, ipar in enumerate(head):    
                # if the cell is not empty
                if ipar != -1 :
                    yc = np.floor_divide(icl,bcsize[0]) # cell y-coordinate
                    xc = np.mod(icl,bcsize[0]) # cell x-coordinate
                    #ipar = head[icl] # header particle of icl-th cell
                    # for each particle in that cell
                    while(ipar != -1):
                        # create the neigbour list
                        list_inbc = []
                        for ilz in range(9):
                            ycneig = yc+neigtab[ilz,1]
                            xcneig = xc+neigtab[ilz,0]
                            # no periodic condition  
                            if (0 <= ycneig < bcsize[1]) and (0 <= xcneig < bcsize[0]):
                                list_inbc.append(int( ycneig*bcsize[0] + xcneig ))         
                        # for each neighbour cell index list 
                        for inbc in list_inbc:
                            jpar = head[inbc] # header particle of cell inbc-th
                            # for each particle in that neighbouring cell
                            while(jpar != -1):
                                if ipar != jpar: 
                                    # calcul the 2-norm
                                    dX[:] = X[jpar,:3] - X[ipar,:3]
                                    r = np.sqrt(dX[0]**2+dX[1]**2)
                                    # consider only particles closer than the cell length
                                    if r < lc:
                                        table_neig[ipar].append(jpar)
                                jpar = table[jpar]
                        ipar = table[ipar]


###########################################
      ## pair correlation function ##
###########################################


def compute_histo_cartesian_2D ( key_fun, list_X, list_X_neig, vecxy):
    
    hist_m = np.zeros(( 2, len(vecxy)-1, len(vecxy)-1 ))
    Nframe = len( list_X )
    
    for h in range(Nframe):

        X = list_X[h]
        X_neig = list_X_neig[h]

        # neighbour
        for i, lneigi in enumerate(X_neig):    
            for j in lneigi:
                
                ## get differential coordinates
                # orientation (rad) of the ith, jth particle
                Ai, Aj = X[i,2], X[j,2]
                Xji = X[j,:2]-X[i,:2] # Xj - Xi
                Aji = np.mod(np.mod(Aj-Ai,2*np.pi),2*np.pi) # Ai - AJ
                # rotation matrix
                MatRot = np.array([ [np.cos(-Ai), -np.sin(-Ai) ], 
                                    [np.sin(-Ai),  np.cos(-Ai) ] ])
                # relative vector in the i base
                Xjin = np.matmul(MatRot,Xji) 
                Ajin = Aji
                
                if vecxy[0] < Xjin[0] < vecxy[-1] and vecxy[0] < Xjin[1] < vecxy[-1] :
                    # index
                    ix = np.where( Xjin[0] - vecxy >= 0, Xjin[0] - vecxy, np.inf).argmin()
                    iy = np.where( Xjin[1]   - vecxy >= 0, Xjin[1]   - vecxy, np.inf).argmin()
                    # key function
                    hist_m[0][ix,iy] += key_fun(X[i,:], X[j,:])
                    # statistic weight
                    hist_m[1][ix,iy] += 1
    
    return hist_m


def Process_Corel ( key_name, list_X, blsize, lcell, vecxy, iproc = 0 ):

    # pid of the processus
    pid = os.getpid(); print('pid = ',pid)
    
    if key_name == 'Cpt':
        key_fun = lambda Xi, Xj : 1
    elif key_name == 'NjNi':
        key_fun = lambda Xi, Xj : np.dot( np.array([np.cos(Xi[2]),np.sin(Xi[2])]), np.array([np.cos(Xj[2]),np.sin(Xj[2])]) )

    hist_m = np.zeros(( 2, len(vecxy)-1, len(vecxy)-1 ))

    # main loop
    for j in range(len(list_X)):
        X = list_X[j:j+1]
        # process neighboors
        X_neig = create_cell( X, lcell, blsize)        
        # compute histogram
        hist_m += compute_histo_cartesian_2D( key_fun, X, X_neig, vecxy )
        # print
        print('iprocess = ',str(iproc),', frame = ',str(j) )

    return hist_m


def pair_corel (list_X_ref, key_name, xboxlim, yboxlim, vecxy, **kwargs):

    ## parameters
    if not( xboxlim.dtype == 'int64' and yboxlim.dtype == 'int64' ):
        raise KeyError('xframelim and yframelim must be of type int 64')
    
    # constraint particles
    list_X, xlim, ylim = constraint_frame( list_X_ref, xboxlim, yboxlim )
    
    # core number
    ncore = kwargs.get( 'ncore', 1 )
    # frame number
    nfra = len(list_X)
    # cell lenght (um)
    lcell = find_lcell(find_common_divisor(xlim,ylim),np.max(vecxy)*np.sqrt(2))

    ## algo
    blsize = np.array([ xlim, ylim, 2*np.pi ])
    vecm = np.meshgrid(vecxy[:-1],vecxy[:-1],indexing='ij')
    hist_m = np.zeros( np.concatenate(( [2], np.shape(vecm[0]) )) ) 

    iwid = int( np.ceil(nfra / ncore) )
    iind = np.arange(0,nfra,iwid)
    # create pool
    pool = mp.Pool(processes=ncore)
    async_result = []
    for ic, ival in enumerate(iind):
        # process pool
        # hist_m += Process_Corel ( key_name, list_X, blsize, lcell, vecxy, 0 )
        async_result.append( pool.apply_async( Process_Corel, args = (  key_name, list_X[ival:ival+iwid], blsize, lcell, vecxy, ic ) ) )    
    # close pool
    pool.close()
    pool.join()
    # collect results
    for i in range(ncore):
        hist_m += async_result[i].get()

    ## format output
    if key_name == 'Cpt':
        
        dx = np.diff(vecxy)[0]
        dy = np.diff(vecxy)[0]
        Lx = vecxy[-1]-vecxy[0]
        Ly = vecxy[-1]-vecxy[0]  
        hist_key = hist_m[0] / (np.sum(hist_m[0])*(dx*dy)/(Lx*Ly)) -1 # pair corel 
        hist_stat = hist_m[1]

    elif key_name == 'NiNj':

        hist_key = hist_m[0]/hist_m[1]
        hist_stat = hist_m[1]
    
    return hist_key, hist_stat


###########################################
            ## diffusion ## 
###########################################


def diffusion_exp(data,xframelim,yframelim):

    # collect trajectories (minthres = 0 ) 
    list_traj = collect_traj_exp(data, 1)
    # unwrap
    for i in range(len(list_traj)):
        # list_traj[i][:,0] = np.unwrap(list_traj[i][:,0],period=xframelim[1])
        # list_traj[i][:,1] = np.unwrap(list_traj[i][:,1],period=yframelim[1])
        list_traj[i][:,2] = np.unwrap(list_traj[i][:,2])  

    # frame number 
    npt = len(data)
    # vector diff
    it = np.arange(0,npt)

    # spatial dffusion
    diffrad = np.zeros(npt)
    for itdiff in it:
        a = [ (traji[itdiff:npt,0]-traji[:npt-itdiff,0])**2 + (traji[itdiff:npt,1]-traji[:npt-itdiff,1])**2 for traji in list_traj ]
        diffrad[itdiff] = np.mean(np.concatenate(a))
        # print(itdiff)

    # rotational diffusion
    diffpol = np.zeros(npt)
    for itdiff in it:
        a = [ (traji[itdiff:npt,2]-traji[:npt-itdiff,2])**2 for traji in list_traj ]
        diffpol[itdiff] = np.mean(np.concatenate(a))
        # print(itdiff)
    # plt.figure();plt.plot(np.unwrap(ltraj_a[0]))

    return diffrad, diffpol, it


def diffusion_simfast(data,xframelim,yframelim):

    npt = len(data)
    list_traj = np.transpose(data,[1,0,2])
    tabx = np.unwrap(list_traj[:,:,0],period=xframelim[1])
    taby = np.unwrap(list_traj[:,:,1],period=yframelim[1])
    tabp = np.unwrap(list_traj[:,:,2])
    
    it = np.arange(0,npt)

    # Dx,Dy
    diffrad = np.zeros(npt)
    for itdiff in it:
        diffrad[itdiff] = np.mean( (tabx[:,itdiff:npt]-tabx[:,:npt-itdiff])**2 + (taby[:,itdiff:npt]-taby[:,:npt-itdiff])**2 )
        # print(itdiff)
    # Dr
    diffpol = np.zeros(npt)
    for itdiff in it:
        diffpol[itdiff] = np.mean( (tabp[:,itdiff:npt]-tabp[:,:npt-itdiff])**2 )
        # print(itdiff)

    return diffrad, diffpol, it


def diffusion( list_frame, xframelim, yframelim ):

    ncol = np.shape(list_frame[0])[1]
    # type sim
    if ncol == 3:
        diffrad,diffpol,it = diffusion_simfast(list_frame,xframelim,yframelim)
    # type exp 
    elif ncol == 4:
        diffrad,diffpol,it = diffusion_exp(list_frame,xframelim,yframelim)
    # plt.figure();plt.plot(np.unwrap(ltraj_a[0]))

    return diffrad,diffpol,it


###########################################
  ## polarity autocorrelation function ##
###########################################


def npcor_sim(data):

    npt = len(data)
    list_traj = np.transpose(data,[1,0,2])
    # np.autocor
    tab = np.unwrap(list_traj[:,:,2])
    tabcos = np.cos(tab)
    tabsin = np.sin(tab)
    # loop
    it = np.arange(0,npt)
    npcor = np.zeros(npt)
    for itdiff in it:
        npcor[itdiff] = np.mean( tabcos[:,itdiff:npt]*tabcos[:,:npt-itdiff] + tabsin[:,itdiff:npt]*tabsin[:,:npt-itdiff] )
        # print(itdiff)
    
    return npcor, it


def npcor_exp(data,minthres):
    
    # collect trajectories (minthres = 0 ) 
    list_traj = collect_traj_exp(data, minthres)
    list_traj = [ np.column_stack( ( traji, np.cos(traji[:,2]), np.sin(traji[:,2]) ) ) for traji in list_traj ]
    # frame number 
    npt = len(list_traj[-1])
    # vector diff
    it = np.arange(0,npt)

    # loop
    it = np.arange(0,npt)
    npcor = np.zeros(npt)
    for itdiff in it:
        a = []
        for traji in list_traj:
            nfrai = len(traji)
            if itdiff < nfrai:
                a.append( traji[itdiff:nfrai,3]*traji[:nfrai-itdiff,3] + traji[itdiff:nfrai,4]*traji[:nfrai-itdiff,4] )
        npcor[itdiff] = np.mean(np.concatenate(a))
        # print(itdiff)

    return npcor, it


def npcorwin_sim(list_frame,nwin,ratio_recov=0.5,ncore=1):

    # warning
    if np.shape(list_frame[0])[1] != 3:
        raise KeyError('expected number of columns in list_frame equal to 3 (x,y,theta)')

    nfra = len(list_frame) # number of frame
    nrec = int(ratio_recov*nwin)
    n = int( np.floor( (nfra-nrec) / (nwin-nrec) ) ) # number of segment

    ind_start = np.arange(n)*nwin - np.arange(n)*nrec
    ind_middle = ind_start + int(nwin/2)

    list_results = [ [] for i in range(n) ]
    pool = mp.Pool(processes=ncore)
    for i in range(n):
        istart = ind_start[i]
        iend = ind_start[i]+nwin
        list_results[i] = pool.apply_async( func=npcor_sim, args = (list_frame[istart:iend],) )
    # close pool
    pool.close()
    pool.join()
    list_results = [ results.get() for results in list_results ]
    
    list_fres = [ results[0] for results in list_results ]
    list_fres = np.column_stack( list_fres )
    it = list_results[0][1]

    return list_fres, it, ind_middle


def npcorwin_exp(list_frame,nwin,ratio_recov=0.5,minthres=100,ncore=1):

    # warning
    if np.shape(list_frame[0])[1] != 4:
        raise KeyError('expected number of columns in list_frame equal to 4')
    
    nfra = len(list_frame) # number of frame
    nrec = int(ratio_recov*nwin)
    n = int( np.floor( (nfra-nrec) / (nwin-nrec) ) ) # number of segment

    ind_start = np.arange(n)*nwin - np.arange(n)*nrec
    ind_middle = ind_start + int(nwin/2)

    list_results = [ [] for i in range(n) ]
    for i in range(n):
        istart = ind_start[i]
        iend = ind_start[i]+nwin
        list_results[i] = npcor_exp(list_frame[istart:iend],minthres=minthres)

    # list_results = [ [] for i in range(n) ]
    # pool = mp.Pool(processes=ncore)
    # for i in range(n):
    #     istart = ind_start[i]
    #     iend = ind_start[i]+nwin
    #     kwds = dict( minthres=minthres ) # to update
    #     list_results[i] = pool.apply_async( func=npcor_exp, args = (list_frame[istart:iend],), kwds = kwds )
    # # close pool
    # pool.close()
    # pool.join()
    # list_results = [ results.get() for results in list_results ]
    
    list_fres = [ results[0] for results in list_results ]
    list_fres = np.column_stack( list_fres )
    it = list_results[0][1]

    return list_fres, it, ind_middle


# def npcor(list_frame,**kwargs):

#     ncol = np.shape(list_frame[0])[1]
#     minthres = kwargs.get('minthres', 0)
#     # print(minthres)

#     # type sim
#     if ncol == 3:
#         fres, it = npcor_simfast(list_frame)
#     # type exp 
#     if ncol == 4:
#         fres, it = npcor_exp(list_frame,minthres) 

#     return fres, it  


#########################################
    ## process polar order / gnf ##
#########################################


def processROI(list_frame,ox,oy,dx):
        
        nfra = len(list_frame)
        bdx = [ ox-dx/2,ox+dx/2]
        bdy = [ oy-dx/2,oy+dx/2]
        order = np.zeros(nfra,dtype=complex)
        count = np.zeros(nfra)
        # every frame
        for j, frame in enumerate(list_frame):
            a = np.logical_and( bdx[0]<=frame[:,0], frame[:,0]<bdx[1] )
            b = np.logical_and( bdy[0]<=frame[:,1], frame[:,1]<bdy[1] )
            c = np.logical_and(a,b) 
            # key function
            order[j] = np.sum(np.exp(1j*frame[c,2]))
            count[j] = np.sum(c) 
        
        return order, count


def polarorder( list_frame, xboxlim, yboxlim, kgrid=1 ):

    ## constrain particles in a square box
    # square box
    ox = (xboxlim[1]+xboxlim[0])/2
    oy = (yboxlim[1]+yboxlim[0])/2
    wid = np.min( [np.diff(xboxlim)[0], np.diff(yboxlim)[0]] )
    sqxboxlim = np.array([ ox-wid/2, ox+wid/2 ])
    sqyboxlim = np.array([ oy-wid/2, oy+wid/2 ])
    # constraint particles
    list_frame, xlim, _ = constraint_frame( list_frame, sqxboxlim, sqyboxlim)
    
    ## definition of the ROI
    # ox, oy, wid
    list_vcen = np.arange(0,kgrid)*(xlim/2)+xlim/(2*kgrid)
    ox,oy = np.meshgrid(list_vcen,list_vcen,indexing='ij')
    # npt_roi = len(list_vcen)**2
    ds0 = (xlim/kgrid)**2
    list_ds = np.logspace(2,np.log10(ds0),50)
    list_dx = np.sqrt(list_ds)
    npt_s = len(list_ds) # total points

    ## algo
    list_pop = np.zeros(npt_s)
    list_mun = np.zeros(npt_s)
    list_std = np.zeros(npt_s)
    nfra = len(list_frame)
    # for every surface
    for h, dx in enumerate(list_dx):
        
        pop = np.zeros([ nfra, np.shape(ox)[0], np.shape(ox)[1] ], dtype=complex)
        cpt = np.zeros([ nfra, np.shape(ox)[0], np.shape(ox)[1] ])

        for i in range(len(list_vcen)):
            for j in range(len(list_vcen)):
                pop[:,i,j] , cpt[:,i,j] = processROI(list_frame,ox[i,j],oy[i,j],dx)
        
        list_mun[h] = np.mean(cpt)
        list_std[h] = np.std(cpt) 

        a = np.ravel(cpt)
        a = a[np.nonzero(a)]
        b = np.ravel(pop)
        b = b[np.nonzero(b)]
        list_pop[h] = np.mean( np.abs( b/a ) )
        
    return list_pop, list_ds, list_mun, list_std

# def gnf_window( list_frame, xboxlim, yboxlim, nwin, ratio_recov=0.5):

#     nfra = len(list_frame)
#     nrec = int(ratio_recov*nwin)
#     # number of segment
#     n = int( np.floor( (nfra-nrec) / (nwin-nrec) ) )

#     ind_start = np.arange(n)*nwin - np.arange(n)*nrec
#     ind_middle = ind_start + int(nwin/2)

#     list_mean = [ [] for i in range(n) ]
#     list_std = [ [] for i in range(n) ]
#     list_s = [ [] for i in range(n) ]

#     for i in range(n):
#         istart = ind_start[i]
#         iend = ind_start[i]+nwin
#         list_mean[i], list_std[i], list_s[i]  = gnf( list_frame[istart:iend], xboxlim, yboxlim )
        
#     return np.column_stack(list_mean), np.column_stack(list_std), np.column_stack(list_s), ind_middle


#########################################
        ## polar histogram ##
#########################################


def histo1d(vecr,h):

    axr = vecr[:-1] + np.diff(vecr)/2
    lenr = np.shape(h)[0]
    h1d = np.array([ np.sum(h[i,:,:]) for i in range(lenr) ])
    h1dn = np.copy(h1d)/axr
    h1dn = h1dn/h1dn[-1]
    return axr, h1d, h1dn


def compute_histo_polar ( X, X_neig, vecr, veca ): 

    X = X[0]
    X_neig = X_neig[0]
    histpol = np.zeros(( len(vecr)-1, len(veca)-1, len(veca)-1 ))

    # neighbour
    for i, lneigi in enumerate(X_neig):    
        for j in lneigi: 
            # distance ij
            Xij = X[i,:2]-X[j,:2]
            d_ij = np.linalg.norm(Xij)
            # X_probe valid and d_ij between bound
            # angle (rad) of e_ij
            Ar = np.mod(np.arctan2(Xij[1],Xij[0]),2*np.pi)
            # angle (rad) of ith, jth particle
            Ai, Aj = X[i,2], X[j,2] 
            # angle (rad) of ith, jth particle
            ai, aj = np.mod(Ai-Ar,2*np.pi), np.mod(Aj-Ar,2*np.pi)
            # Increase Count
            if vecr[0] <= d_ij < vecr[-1]:
                ir = np.where(d_ij-vecr >= 0, d_ij-vecr, np.inf).argmin()
                iai = np.where(ai-veca >= 0, ai-veca, np.inf).argmin()
                iaj = np.where(aj-veca >= 0, aj-veca, np.inf).argmin()
                histpol[ir,iai,iaj] += 1

    return histpol

def Process_Histo ( list_X, blsize, lcell, vecr, veca, iproc = 0 ):

    # pid of the processus
    pid = os.getpid(); print('pid = ',pid)
    
    histpol = np.zeros(( len(vecr)-1, len(veca)-1, len(veca)-1 ))

    # main loop
    for j in range(len(list_X)):
        X = list_X[j:j+1]
        # process neighboors
        X_neig = create_cell( X, lcell, blsize)        
        # compute histogram
        histpol += compute_histo_polar( X, X_neig, vecr, veca )
        # print
        print('iprocess = ',str(iproc),', frame = ',str(j) )

    return histpol


def polar_histo (list_X_ref, xboxlim, yboxlim, vecr, veca, **kwargs ):

    ## parameters
    if not( xboxlim.dtype == 'int64' and yboxlim.dtype == 'int64' ):
        raise KeyError('xframelim and yframelim must be of type int 64')
    
    # constraint particles
    list_X, xlim, ylim = constraint_frame( list_X_ref, xboxlim, yboxlim )
    
    # core number
    ncore = kwargs.get( 'ncore', 1 )
    # frame number
    nfra = len(list_X)
    # cell lenght (um)
    lcell = find_lcell(find_common_divisor(xlim,ylim),np.max(vecr)*np.sqrt(2))

    ## algo
    blsize = np.array([ xlim, ylim, 2*np.pi ])
    histpol = np.zeros(( len(vecr)-1, len(veca)-1, len(veca)-1 ))

    iwid = int( np.ceil(nfra / ncore) )
    iind = np.arange(0,nfra,iwid)

    if ncore == 1:
        async_result = []
        for ic, ival in enumerate(iind):
            # process pool
            # hist_m += Process_Corel ( key_name, list_X, blsize, lcell, vecxy, 0 )
            histpol += Process_Histo(  list_X[ival:ival+iwid], blsize, lcell, vecr, veca, ic )  
        
    elif ncore > 1:
        # create pool
        pool = mp.Pool(processes=ncore)
        async_result = []
        for ic, ival in enumerate(iind):
            # process pool
            # hist_m += Process_Corel ( key_name, list_X, blsize, lcell, vecxy, 0 )
            async_result.append( pool.apply_async( Process_Histo, args = (  list_X[ival:ival+iwid], blsize, lcell, vecr, veca, ic ) ) )    
        # close pool
        pool.close()
        pool.join()
        # collect results
        for i in range(ncore):
            histpol += async_result[i].get()

    return histpol


