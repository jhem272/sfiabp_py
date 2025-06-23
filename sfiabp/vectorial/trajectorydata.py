
import numpy as np
import copy

from sfiabp.vectorial import cellprocess

def PreProcess( list_data, boxsize ):
    
    N = len(list_data)
    list_data_new = [ [] for i in range(N) ]
    dtheta = 2*np.pi

    for i in range(N):
        # remove particles found outside the box
        bolx = np.logical_and( boxsize[0,0] <= list_data[i][:,0], list_data[i][:,0] < boxsize[0,1] )
        boly = np.logical_and( boxsize[1,0] <= list_data[i][:,1], list_data[i][:,1] < boxsize[1,1] )
        iid = np.where( np.logical_and(bolx,boly) == False )[0]    
        list_data_new[i] = np.delete(list_data[i],iid,axis=(0))
        # origin (0,0)
        list_data_new[i][:,0] = list_data_new[i][:,0] - boxsize[0,0]
        list_data_new[i][:,1] = list_data_new[i][:,1] - boxsize[1,0]
        # constraint orientation
        list_data_new[i][:,2] = np.mod(np.mod(list_data_new[i][:,2],dtheta),dtheta)

    return list_data_new        

def bound_mat(mat,blsize,mode='all'):

    if mode == 'all':
        dim = len(blsize)
        for d in range (dim):
            mat[:,d] = np.mod(np.mod(mat[:,d],blsize[d]),blsize[d])
    
    elif mode == 'angle':
        mat[:,2] = np.mod(np.mod(mat[:,2],blsize[2]),blsize[2])

    return mat

def bound_mat_v2(list_frame,blsize):

    flag = False
    for X in list_frame:
        npar = len(X)
        for i in range(npar):
            if not( 0 <= X[i,0] < blsize[0]):
               flag = True
            if not( 0 <= X[i,1] < blsize[1]):
               flag = True
            if not( 0 <= X[i,2] < blsize[2]):
               flag = True  
        
        bound_mat(X,blsize)

    return flag


class StochasticTrajectoryData(object): 
    
    """This class is a formatter and wrapper class for stochastic
    trajectories. It performs basic operations (discrete derivative,
    mid-point "Stratonovich" positions.
    """ 

    def __init__(self, list_frame, dn, Dn, dtframe, boxsize, blsize, lcell):
        
        """ list_frame is a list of np.array of shape : frame X n_particle X (x,y,theta) 
        or frame X n_particle X (x,y,theta,idparticle) """ 

        # create X_minus, X_ito, X_plus, X_id, X_Probe

        def delete_particles(data,list):
            for npar in list:
                mask = data[:,-1] == npar
                data = data[np.logical_not(mask),:]
            return data 

        imax = len(list_frame)-1
        N = int( np.floor((imax-2*dn)/Dn) + 1 ) # number of frame 

        self.boxsize = boxsize
        self.blsize = blsize
        self.lcell = lcell

        ## gather X_minus, X_ito, X_plus ##
        
        i_star = 0
        ind_minus = range(i_star,i_star+N*Dn,Dn) 
        ind_ito = range(i_star+dn,i_star+dn+N*Dn,Dn)
        ind_plus = range(i_star+2*dn,i_star+2*dn+N*Dn,Dn) 
        
        if ind_plus[-1] > imax:
            raise KeyError("Invalid number of frame: ",N) 
        
        Nparticles_old = [ np.shape(list_frame[ind_ito[i]])[0] for i in range (N)]

        if len(list_frame)>2 and np.shape(list_frame[0])[1]==3:
            # simulated data
            # dimension of list_frame must be nframe x nparticle x 3 coordinates (x,y,theta) ')
            
            X_minus = [ list_frame[ind_minus[i]] for i in range(N) ]
            X_ito = [ list_frame[ind_ito[i]] for i in range(N) ]
            X_plus = [ list_frame[ind_plus[i]] for i in range(N) ]        

            # filter particles found close to the edge of the box 
            # (only for simulated data)
            for i in range(N):
                x = np.column_stack(( X_minus[i][:,0], X_ito[i][:,0], X_plus[i][:,0] ))
                dx = np.diff(x)
                y = np.column_stack(( X_minus[i][:,1], X_ito[i][:,1], X_plus[i][:,1] ))
                dy = np.diff(y)
                bx = np.any( np.abs(dx) >= self.blsize[0]/2 ,axis=1)
                by = np.any( np.abs(dy) >= self.blsize[1]/2 ,axis=1)
                b = np.logical_not( np.logical_or(bx,by) )
                # if np.all(b):
                #     print('ok')
                # delete particles
                X_minus[i] = X_minus[i][b,:]
                X_ito[i] = X_ito[i][b,:]
                X_plus[i] = X_plus[i][b,:]

            # identity / probe
            self.X_id = [ np.arange(np.shape(X_ito[i])[0]) for i in range (N) ]
            self.X_probe = [ np.ones(np.shape(X_ito[i])[0]) for i in range (N) ]

        ## standard treatment
        elif len(list_frame)>2 and np.shape(list_frame[0])[1]==4:
            # experimental data
            # dimension of list_frame must be nframe x nparticle x 4 coordinates (x,y,theta,id number) ')

            X_minus = [ list_frame[ind_minus[i]] for i in range(N) ]
            X_ito = [ list_frame[ind_ito[i]] for i in range(N) ]
            X_plus = [ list_frame[ind_plus[i]] for i in range(N) ]        
            
            id_minus = [ set(list_frame[ind_minus[i]][:,3]) for i in range(N) ]
            id_ito = [ set(list_frame[ind_ito[i]][:,3]) for i in range(N) ]
            id_plus = [ set(list_frame[ind_plus[i]][:,3]) for i in range(N) ]

            # filter particles
            for i in range(N):
            
                sint = id_minus[i] & id_ito[i] & id_plus[i]
                
                ## delete particles
                X_minus[i] = delete_particles(X_minus[i],list(id_minus[i]-sint)) 
                X_ito[i] = delete_particles(X_ito[i],list(id_ito[i]-sint)) 
                X_plus[i] = delete_particles(X_plus[i],list(id_plus[i]-sint)) 
                
                b = np.array_equal(X_minus[i][:,3],X_ito[i][:,3])
                bb = np.array_equal(X_ito[i][:,3],X_plus[i][:,3])
                if not(b) or not(bb):
                    raise KeyError("Invalid array_equal") 

            # particle numero
            self.X_id = [ X_ito[i][:,3] for i in range(N) ]
            # particle probe
            self.X_probe = [ np.ones(np.shape(X_ito[i])[0]) for i in range (N) ]
        
        else:
            raise KeyError("Invalid data, please check") 
        
        # X_minus ( keep x,y,theta and delete other columns)
        self.X_minus = [ X_minus[i][:,:3] for i in range(N) ]
        # X_plus
        self.X_plus = [ X_plus[i][:,:3] for i in range(N) ]
        # X_ito
        self.X_ito = [ X_ito[i][:,:3] for i in range(N) ]
        # dimension
        self.d = np.shape(self.X_ito[0])[1]
        # dtime vector
        self.dt = dtframe*np.ones(N)
        # time vector 
        self.t = np.linspace(0., dtframe*N, N,endpoint=False)
        # particle number
        self.Nparticles = [ np.shape(self.X_ito[i])[0] for i in range (N)]
        # frame number 
        self.N = len(self.X_ito)
        
        # if DelRaw:        
        #     del data['dtframe']
        # particle type
        # self.X_type = [ np.zeros(np.shape(X_ito[i])[0]) for i in range (N) ]


    def Probe_RemoveEdge(self):
        
        """ set X_probe == 0 for particles close to the box edges """ 

        Xbound = np.array([ 0+self.lcell,self.blsize[0]-self.lcell ])
        Ybound = np.array([ 0+self.lcell,self.blsize[1]-self.lcell ])
        
        N = len(self.X_probe)
        for i in range(N):
            # bolean arrays
            bolx = np.logical_and( Xbound[0] < self.X_ito[i][:,0], self.X_ito[i][:,0] < Xbound[1] )
            boly = np.logical_and( Ybound[0] < self.X_ito[i][:,1], self.X_ito[i][:,1] < Ybound[1] )
            vbol = np.logical_and(bolx,boly)
            # update Xprobe
            self.X_probe[i][:] = np.logical_and(self.X_probe[i],vbol)

   
    def MidPoint(self):
        
        # particle number corrected by X_probe
        # self.Nparticles_PR = data['Nparticles'] # particle number list
        self.Nparticles = [ int(np.sum(self.X_probe[i])) for i in range(self.N) ]
        # self.Nparticles = data['Nparticles'] # particle number list
        # tauN
        self.tauN = np.einsum('t,t->',self.dt,self.Nparticles) # particle number x dt quantities
        self.iN = np.einsum('t->',self.Nparticles)

        ## create dX_plus dX_minus ##

        self.dX_plus = [ [] for i in range (self.N)] 
        self.dX_minus = [ [] for i in range (self.N)]
        self.Xdot = [ [] for i in range (self.N)]
        
        for i in range(self.N):
            
            x = np.column_stack(( self.X_minus[i][:,0],self.X_ito[i][:,0],self.X_plus[i][:,0] ))
            dx = np.diff(np.unwrap(x,period=self.blsize[0]))
            y = np.column_stack(( self.X_minus[i][:,1],self.X_ito[i][:,1],self.X_plus[i][:,1] ))
            dy = np.diff(np.unwrap(y,period=self.blsize[1]))
            t = np.column_stack(( self.X_minus[i][:,2],self.X_ito[i][:,2],self.X_plus[i][:,2] ))
            dt = np.diff(np.unwrap(t))

            self.dX_minus[i] = np.column_stack(( dx[:,0], dy[:,0], dt[:,0] ))
            self.dX_plus[i] = np.column_stack(( dx[:,1], dy[:,1], dt[:,1] )) 
            self.Xdot[i] = self.dX_plus[i]/self.dt[i]

        ## create X_strat, X_smooth ##
        # better using dX_plus, dX_minus rather than X_plus, X_minus
        # calcul X_strat and apply periodic boound to Xstrat
        self.X_strat = [self.X_ito[t] + 0.5 * self.dX_plus[t] for t in range(self.N)]
        self.X_isop = [ self.X_ito[t] + self.dX_plus[t] for t in range(self.N) ]
        self.X_isom = [ self.X_ito[t] - self.dX_minus[t] for t in range(self.N) ]
        # self.X_smooth = [ self.X_ito[t] + (self.dX_plus[t]-self.dX_minus[t])/3. for t in range(self.N) ]

        self.X_strat = [bound_mat(self.X_strat[t],self.blsize,mode='angle') for t in range (self.N)]
        self.X_isop = [bound_mat(self.X_isop[t],self.blsize,mode='angle') for t in range (self.N)]
        self.X_isom = [bound_mat(self.X_isom[t],self.blsize,mode='angle') for t in range (self.N)]
        # self.X_smooth = [bound_mat(self.X_smooth[t],blsize) for t in range (self.N)]

        if bound_mat_v2( self.X_strat,self.blsize) :
            raise KeyError('particles found outside of the box, perhaps the blsize is not correct')
        if bound_mat_v2( self.X_isop,self.blsize) :
            raise KeyError('particles found outside of the box, perhaps the blsize is not correct')
        if bound_mat_v2( self.X_isom,self.blsize) :
            raise KeyError('particles found outside of the box, perhaps the blsize is not correct')
    
    
    def CellProcess(self):

        ## cell mode ##
        # compute neig list for each data X_ito, X_strat, X_isom, X_isop (very important)         
        # process neighboors
        
        self.X_ito_neig = cellprocess.create_cell( self.X_ito, self.lcell, self.blsize ) 
        self.X_ito = [ [self.X_ito[t], self.X_ito_neig[t], self.X_probe[t] ] for t in range (self.N)]
        self.X_strat_neig = cellprocess.create_cell( self.X_strat, self.lcell, self.blsize )
        self.X_strat = [ [self.X_strat[t], self.X_strat_neig[t], self.X_probe[t] ] for t in range (self.N)]
        self.X_isom_neig = cellprocess.create_cell( self.X_isom, self.lcell, self.blsize )
        self.X_isom = [ [self.X_isom[t], self.X_isom_neig[t], self.X_probe[t] ] for t in range (self.N)]
        self.X_isop_neig = cellprocess.create_cell( self.X_isop, self.lcell, self.blsize )
        self.X_isop = [ [self.X_isop[t], self.X_isop_neig[t], self.X_probe[t] ] for t in range (self.N)]

        # wrong procedure
        # elif option == 'vold':
        #     self.X_ito_neig = Sfi_Vec_Lib_SupFunc.create_cell( self.X_ito, lcell, blsize ) 
        #     self.X_ito = [ [self.X_ito[t], self.X_ito_neig[t], self.X_type[t], self.X_probe[t] ] for t in range (self.N)]
        #     self.X_strat = [ [self.X_strat[t], self.X_ito_neig[t], self.X_type[t], self.X_probe[t] ] for t in range (self.N)]
        #     self.X_isom = [ [self.X_isom[t], self.X_ito_neig[t], self.X_type[t], self.X_probe[t] ] for t in range (self.N)]
        #     self.X_isop = [ [self.X_isop[t], self.X_ito_neig[t], self.X_type[t], self.X_probe[t] ] for t in range (self.N)]

        # self.X_neig = Sfi_Vec_Lib_SupFunc.conv_list_array(data['X_neig'],1)
        # self.X_smooth = [ [self.X_smooth[t], neig[t]] for t in range (self.N)]


    def SliceData ( self, i , iwid ) :

        data_new = copy.deepcopy(self)
        data_new.N = iwid

        data_new.t = self.t[i:(i+iwid)]
        data_new.dt = self.dt[i:(i+iwid)]
        data_new.Nparticles = self.Nparticles[i:(i+iwid)]
        data_new.X_ito = self.X_ito[i:(i+iwid)]
        data_new.X_plus = self.X_plus[i:(i+iwid)]
        data_new.X_minus = self.X_minus[i:(i+iwid)]
        data_new.X_id = self.X_id[i:(i+iwid)]
        data_new.X_probe = self.X_probe[i:(i+iwid)]
        
        return data_new



