

import matplotlib.pyplot as plt
import numpy as np
import dill 
import os


##################################
        ## file manager ##
##################################

# return a list of file name, with a specific prefix and ending with .pkl 
def find_pkl (folderpath,prefix):
    ls = []
    for x in os.listdir(folderpath):
        if x.startswith(prefix):
            ls.append(x)
    list_fname = [folderpath + '/' + ls[i] for i in range(len(ls)) ]
    list_fname.sort()
    return list_fname

# get .pkl
def getpkl(pathfile):
    with open( pathfile, 'rb' ) as inp:    
        data = dill.load(inp)
    return data

# gte list of .pkl
def get_listpkl(pathfolder,keyword):
    list_DataFile = find_pkl(pathfolder,keyword)
    list_Data = [ [] for i in list_DataFile ]
    for i, DataFile in enumerate(list_DataFile):
        with open( DataFile, 'rb' ) as inp:    
            list_Data[i] = dill.load(inp)
    return list_Data, list_DataFile

def export_coeffs (dirname):
    """
    Exports the infered coefficients
    
    Exports the infered coefficients as one file per dimension. 
    Each file contains an array: the first index indicates the angular,
    dependence the second indicates the radial function.
    
    Parameters
    ----------
    
    dirname : str
        Index of the inference run.
    """
    fname = find_pkl (dirname)
            
    print(fname)
    
    with open(fname,'rb') as inputfile :
        S = dill.load(inputfile)

    for iid, basis in enumerate(S.drift_basis) :
        if basis['Name'] == 'Pair_Trigo_3':
            Pair_Base = basis
            
    Table_Cof = Pair_Base['Table_Cof'] 

    for dim in range(3):
        np.savetxt('coeffs_inf' + dirname + '_' + str(dim) + '.dat', Table_Cof[dim,:,:])
    
    return


###################################
      ## Print report Sabp ##
###################################


def print_report(Sabp, PathSFile=''):

    print('\n-----------------------------------------------------------')
    print('--- StochasticForceInference report for ABP particles ---')
    print('-----------------------------------------------------------')

    print('S File name : ',os.path.basename(PathSFile))
    
    print('Dimension xbox (um): ',Sabp['psfi']['xboxlim'])
    print('Dimension ybox (um) : ',Sabp['psfi']['yboxlim'])
    print('cell length (um) : ',Sabp['psfi']['lcell'])

    print('N frame processed : ',Sabp['psfi']['frame_weight'])
    print('stat weight:',Sabp['psfi']['stat_weight'])
    print('pair stat weight:',Sabp['psfi']['pair_stat_weight'])
    print('neigbor per particle : ',Sabp['psfi']['pair_stat_weight']/Sabp['psfi']['stat_weight'])

    print('Drift mode:',Sabp['psfi']['drift_mode'])
    if Sabp['psfi']['drift_mode'] == 'Stratonovich':
        print('Stratonovich difference mode:',Sabp['psfi']['strato_diff_mode'])
    print('Inverse mode : ',Sabp['psfi']['inverse_mode'])
    print('B mode : ',Sabp['psfi']['B_mode'])
    
    # print('Average diffusion tensor:')
    # print(Sabp['D_average'])
    # print('Measurement noise tensor:')
    # print(Sabp['Lambda'])

    print('# Abp results')
    print('Data file name : ',Sabp['data_file'])
    print('N particle (in first frame) : ',np.shape(Sabp['frame_init'])[0])
    print('Frame range : ',Sabp['iid'])
    print('Basis name:',Sabp['basis_name'])
    if Sabp['basis_name'] == 'Trigo':
        print('Active velocity (um/s) : ',Sabp['active_vel'])
        print('Rotational Diffusion (/s) : ',Sabp['D_average'][2,2])
        print('Order:',Sabp['Order'])
        print('FuncRad:',Sabp['FuncRad'])
        print('VectorRad:',Sabp['VectorRad'])


###################################
          ## Trackpy ##
###################################


def conv_dframe (dframe,mod_theta=False):
    
    # column 0,1,2,3 (4): x, y , theta, frame, particle
    
    if mod_theta:
        dframe[:,2] = np.mod(dframe[:,2],2*np.pi)
    
    idframe = np.unique(dframe[:,3])
    idframe.sort()
    list_frame=[]
    for i,idd in enumerate(idframe):
        ab = dframe[dframe[:,3] == idd]
        ab = np.delete(ab,3,axis=1) # remove frame column
        list_frame.append(ab)
        # sort if particle column exists
    
    ncol = list_frame[0].shape[1]
    nfra = len(list_frame)
    if ncol == 4:
        for i in range(nfra):
            ind = np.argsort(list_frame[i][:,-1])
            list_frame[i][:] = list_frame[i][ind,:] 

    return list_frame


def istrajok(d) :

    # get the frame column
    c0 = d[:,0]
    #un = np.unique(c0,0)

    nfra = int ( c0[-1] ) 
    si = np.zeros(nfra)

    # for each frame get the number of trajectories
    for i in range (nfra) :
        tabb = c0 == i 
        datar = d[tabb,:] # alias
        si[i] = datar.shape[0]

    bol = np.all( si == si[0] )
    return bol


########################################
## additional functions Trackpy / SFI ##
########################################


# Sfi_Vec_Lib_Multicore
def addGWnoise(data,br):
    
    nfra = len(data)
    for i in range(nfra):
        npart = len(data[i])
        theta_noise = np.random.normal(0,br,npart)
        data[i][:,2] = np.mod(data[i][:,2]+theta_noise,2*np.pi) 
    print('add noise')


# Sfi_Vec_Lib_Multicore
def addnoiseQ(data,c):
    # should be identity for experimental data    
    nfra = len(data)
    for i in range(nfra):
        # x
        data[i][:,0] = (np.floor_divide(data[i][:,0],c)+0.5)*c
        # y
        data[i][:,1] = (np.floor_divide(data[i][:,1],c)+0.5)*c
    print('add noise Q')


########################################
    ## Observables / deprecated ##
########################################

# Param = dict( 
#             list_fileData = list_fileData, ,
#                 MultiProc_Ok = MultiProc_Ok, blsize = blsize, blsize_sfi = blsize_sfi,
#                      )

def conv_list_array(tab, direction=0):

    nfra = len(tab) 
    if direction == 0:
        # list to arrray
        list_array= [] 
        for i in range(nfra):
            b = np.ones([len(tab[i]),len(max(tab[i],key = lambda x: len(x)))])*-1
            for j,k in enumerate(tab[i]):
                b[j][0:len(k)] = k
            list_array.append(b)
        return list_array
    
    else:
        # array to list
        npart = len(tab[0])
        list_list = [ [] for i in range(nfra) ]
        for i in range(nfra):
            for j in range(npart):
                lpar = []
                for k in range(len(tab[i][j])):
                    if tab[i][j,k] != -1:
                        lpar.append(int(tab[i][j,k]))  
                list_list[i].append(lpar)

        return list_list

#### functions speed ####

def Stat(FolderName, list_fileData, strData):

    nfile = 0
    npart = 0

    # get info
    for i,file in enumerate(list_fileData):
        with open( FolderName + '/' + file, 'rb') as inp:
            data = dill.load(inp)
        
        frame_nfile = len(data[strData])
        frame_dt = data['dtframe']
        nfile += frame_nfile
        npart += np.sum( [ len(data[strData][j]) for j in range(frame_nfile) ] ) 
        
    Stat = dict([ ('nfile',nfile), ('total_npart',npart), 
                        ('frame_nfile',frame_nfile), ('frame_dt',frame_dt ) ]) 

    return Stat

def slidevar(x,dn):

    npt = len(x)
    mu = 0 ; va = 0; cp = 0
    if dn<npt:
        for i in range(int(npt-dn)):
            mu += x[i+dn]-x[i]
            va += (x[i+dn]-x[i])**2
            cp += 1
    return np.array([mu,va,cp])

def varwalk(lx,dnvec):

    # increment vector
    tabvar = np.zeros((len(dnvec),4))
    for i,dn in enumerate(dnvec):
        for j in range(len(lx)):
            tabvar[i,0:3] += slidevar(lx[j],int(dn))
        print(str(i)+'/'+str(len(dnvec)))
    return tabvar

def slidevar2d(x,y,dn):

    npt = len(x)
    mu = 0 ; va = 0; cp = 0
    if dn<npt:
        for i in range(int(npt-dn)):
            d = np.sqrt( (x[i+dn]-x[i])**2 +  (y[i+dn]-y[i])**2 )
            mu += d
            va += d**2
            cp += 1
    return np.array([mu,va,cp])

def varwalk2d(lx,ly,dnvec):

    # increment vector
    tabvar = np.zeros((len(dnvec),4))
    for i,dn in enumerate(dnvec):
        for j in range(len(lx)):
            tabvar[i,0:3] += slidevar2d(lx[j],ly[j],int(dn))
        print(str(i)+'/'+str(len(dnvec)))
    return tabvar

def miniabs(lnp):

    nr,nc = np.shape(lnp[0])
    tab = np.zeros((nr,nc))
    tabin = np.array(lnp)
    for i in range(nr):
        for j in range(nc):
            tab[i,j] = min(tabin[:,i,j],key=abs)

    return tab

def findtraj(d,ipar,count,nfra):
    # count must be > 0 (the particle exists)
    tab = np.zeros((int(count),3))
    for i in range(int(count)):
        dd = d[int(nfra+i)]
        tab[i,:] = dd[np.where(dd[:,3] == ipar)][:,:3]
    return tab

def collect_data(d): 
    
    nfile = len(d)    
    for i in range(nfile):
        npart = len(d[i])
        if i == 0:
            tab = np.zeros([npart,3])
            tab[:,0] = d[i][:,3]
            tab[:,1] = np.ones(npart) 
        else:
            for j in range(npart): 
                ipar = int(d[i][j,3]) # particle number            
                if ipar in tab[:,0]: # if exist
                    ind = np.where(tab[:,0]==ipar)[0][0]
                    tab[ind,1] += 1
                else:
                    tab = np.vstack((tab,np.array([ ipar, 1, i])))
        print('file ',str(i))

    ltraj = []
    for i in range(len(tab)):
        ltraj.append(findtraj(d,tab[i,0],tab[i,1],tab[i,2])) 

    return ltraj

def ListTraj (FolderName, list_fileData, strData):

    # get info
    with open( FolderName + '/' + list_fileData[0], 'rb') as inp:
        data = dill.load(inp)
    Mode_Data = data['Mode_Data']
    blsize = data['blsize']

    if Mode_Data == 'SIM':

        ltraj = []
        for file in list_fileData:
            
            # reorganize data
            with open( FolderName + '/' + file, 'rb') as inp:
                data = dill.load(inp)
            ltraji = np.einsum('ijk -> jik',np.array(data[strData])) 

            # unwrap trajectories
            for j in range(len(ltraji)):
                ltraji[j][:,0] = np.unwrap(ltraji[j][:,0], period=blsize[0])
                ltraji[j][:,1] = np.unwrap(ltraji[j][:,1], period=blsize[1])
                # be careful angle may not be between 0 and 2*pi
                ltraji[j][:,2] = np.unwrap(ltraji[j][:,2]) 
            
            ltraj.append(ltraji)

    elif Mode_Data == 'EXP': 

        ltraj = []
        for file in list_fileData:
            
            # reorganize data
            with open( FolderName + '/' + file, 'rb') as inp:
                data = dill.load(inp)
            ltraji = collect_data(data[strData]) 

            # unwrap trajectories
            for j in range(len(ltraji)):
                ltraji[j][:,0] = np.unwrap(ltraji[j][:,0], period=blsize[0])
                ltraji[j][:,1] = np.unwrap(ltraji[j][:,1], period=blsize[1])
                # be careful angle may not be between 0 and 2*pi
                ltraji[j][:,2] = np.unwrap(ltraji[j][:,2]) 

            ltraj.append(ltraji)

    return ltraj

def CountSpeed(FolderName, list_fileData, ltraj, binu):

    # get info
    with open( FolderName + '/' + list_fileData[0], 'rb') as inp:
        data = dill.load(inp)
    dt = data['dtframe']
    
    histu = np.zeros(len(binu)-1)

    for i in range(len(ltraj)):
        
        # histo
        npart = len(ltraj[i])
        Dx = [ np.diff(ltraj[i][j][:,0]) for j in range(npart) ]
        Dy = [ np.diff(ltraj[i][j][:,1]) for j in range(npart) ]
        Du = [ np.sqrt(Dx[j]**2+Dy[j]**2)/dt for j in range(npart) ]

        for j in range(len(Du)):
            histu += np.histogram(Du[j],bins=binu)[0]
        # Dx = [ data[strData][j+1][:,:3] - data[strData][j][:,:3] for j in range(nfilu) ]
        # Npar = [ np.shape(Dx[j])[0] for j in range(nfilu) ] 
        # Bls = [ np.einsum('ij,k->ik', np.ones((Npar[j],1)), blsize) for j in range(nfilu) ]
        # Dx = [ miniabs( [Dx[j]-Bls[j], Dx[j], Dx[j]+Bls[j] ] ) for j in range(nfilu)]
        # Du = [ np.sqrt(Dx[j][:,0]**2+Dx[j][:,1]**2)/data['dtframe'] for j in range(nfilu) ]
        # histou += np.histogram(Du,bins=binu)[0]

    # save histo du
    # plt.figure(); plt.plot(binu[:-1],histou)
    HistoSpeed = dict([ ('histu',histu),('binu',binu) ])

    return HistoSpeed

def CalcDiffu (FolderName, list_fileData, ltraj):

    # get info
    with open( FolderName + '/' + list_fileData[0], 'rb') as inp:
        data = dill.load(inp)
    dt = data['dtframe']

    # init vector dnvec based on the trajectory length (mean or max)
    nfile = len(ltraj)
    totn = np.max( [ len(ltraj[0][i]) for i in range(len(ltraj[0])) ] )
    dnvec = np.arange(1,totn)

    # Mode_Data = data['Mode_Data']
    # if Mode_Data == 'SIM':
    #     dnvec = np.arange(1,totn)    
    # elif Mode_Data == 'EXP': 
    #     # dnvec = np.arange(1,int(10**np.floor(np.log10(nfra))))
    #     dnvec = np.arange(1,totn)

    tab_r = np.zeros((len(dnvec),4))
    tab_a = np.zeros((len(dnvec),4)) 

    for i in range(nfile):

        npart = len(ltraj[i])
        ltraj_x = [ ltraj[i][j][:,0] for j in range(npart) ]
        ltraj_y = [ ltraj[i][j][:,1] for j in range(npart) ]
        ltraj_a = [ ltraj[i][j][:,2] for j in range(npart) ]
        # variance
        tab_r += varwalk2d(ltraj_x,ltraj_y,dnvec)
        tab_a += varwalk(ltraj_a,dnvec)
        # plt.figure();plt.plot(np.unwrap(ltraj_a[0]))

    # calculate and save the variance
    # 0: mean dr ,1: va dr**2 , 2: count
    tab_r[:,3] = tab_r[:,1]/tab_r[:,2] - (tab_r[:,0]/tab_r[:,2])**2
    tab_a[:,3] = tab_a[:,1]/tab_a[:,2] - (tab_a[:,0]/tab_a[:,2])**2
    Diffu = dict([ ('varr',tab_r),('vara',tab_a),('dt',dnvec*dt ) ]) 

    return Diffu





