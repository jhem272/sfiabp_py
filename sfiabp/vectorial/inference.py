#
# StochasticForceInference is a package developed by Pierre Ronceray
# and Anna Frishman aimed at reconstructing force fields and computing
# entropy production in Brownian processes by projecting forces onto a
# finite dimensional functional space. Compatible Python 3.6 / 2.7.
#
# Reference: Anna Frishman and Pierre Ronceray, Learning force fields
#     from stochastic trajectories, arXiv:1809.09650, 2018.
#
import scipy
import numpy as np
from sfiabp.vectorial import cellprocess

## S object ## 

class StochasticForceInference(object): 
    
    ## Definition class StochasticForceInference
    def __init__( self, fun1p, fun2p, P = '' ):

        if isinstance(P,dict):
            # def sfi_init( self, drift_basis, drift_basis_option, diffusion_basis, diffusion_basis_option ):
            "drift_basis : list of dict"
            "drift_basis_option : list of dict"
            
            self.B_mode = P['B_mode']
            self.drift_mode = P['drift_mode']
            self.strato_diff_mode = P['strato_diff_mode']
            self.matdif = P['strato_diff_matrix']
            self.diffusion_mode = P['diffusion_mode']
            self.drift_b, self.drift_grad_b  = cellprocess.basis_selector(fun1p,fun2p)
            
            # current statistical weight
            self.stat_weight = 0
            # current number of frame
            self.frame_weight = 0
            # current pair statistical weight
            self.pair_stat_weight = 0

            # mesh / histogram
            self.histo_vecr = P['histo_vecr']
            self.histo_veca = P['histo_veca']
            self.histo_mesh = np.meshgrid( self.histo_vecr[:-1], self.histo_veca[:-1], self.histo_veca[:-1], indexing='ij' )
            self.histo = np.zeros(( 1, len(self.histo_vecr)-1, len(self.histo_veca)-1, len(self.histo_veca)-1 ))

    def sfi_addstat(self,data):
        
        # Throughout, indices m,n... denote spatial indices; indices
        # i,j.. denote particle indices; indices a,b,... are used for
        # the fit functions.

        self.data = data

        sta_weight = self.stat_weight # current weight
        cur_weight = data.iN    # weight of the next data 
        new_weight = sta_weight+cur_weight # new weight 
        
        #### Drift mode ####
        
        ### acc_mode
        self.bxt_i = self.drift_b(data.X_ito[0]) 
        self.bxt_s = self.drift_b(data.X_strat[0])
        self.bxt_isom = self.drift_b(data.X_isom[0])
        self.bxt_isop = self.drift_b(data.X_isop[0])
        
        # vall = np.max(np.abs(data.X_ito[0][0]-data.X_strat[0][0]))
        # if vall>100:
        #     print('ok')

        ### updata probe
        self.data.Xdot[0] = self.data.Xdot[0][self.data.X_probe[0].astype(bool),:]
        self.data.dX_plus[0] = self.data.dX_plus[0][self.data.X_probe[0].astype(bool),:]
        self.data.dX_minus[0] = self.data.dX_minus[0][self.data.X_probe[0].astype(bool),:]

        ### Initialization
        if sta_weight == 0: 
            
            Nfuncs = np.shape(self.bxt_i)[1]
            self.drift_B = np.zeros((Nfuncs,Nfuncs))
            
            self.v_scalarb = np.zeros(Nfuncs)
            self.phi_scalarb = np.zeros(Nfuncs)
            self.int_D_grad_b = np.zeros(Nfuncs)
            self.w_scalarb = np.zeros(Nfuncs)

            # self.v_coefficients = np.zeros((3,Nfuncs))
            # self.phi_coefficients = np.zeros((3,Nfuncs))
            # self.w_coefficients = np.zeros((3,Nfuncs))

            # data_drift = { 'weight': [], 'v_coefficients' : [], 'phi_coefficients' : [], 'w_coefficients' : [], 'B' : [] }
            # self.SFI_DicoDat = { 'data_drift' : data_drift }

        ### New estimates / B_mode 

        if self.B_mode == 'rectangle': 
            drift_B = np.einsum('iam,ibm->ab',self.bxt_i,self.bxt_i)
            self.drift_B = (sta_weight*self.drift_B + cur_weight*drift_B)/new_weight
        
        elif self.B_mode == 'trapeze': 
            # self.bxt_isop = self.drift_b(data.X_isop[0])
            # b_inst_s = 0.5*(b_inst_i+self.drift_b(self.data.X_ito[t]+self.data.dX_plus[t]))
            self.bxt_trapeze = 0.5*(self.bxt_i+self.bxt_isop)
            # return np.einsum('iam,ibm->ab',b_inst_s,b_inst_i)
            drift_B = np.einsum('iam,ibm->ab',self.bxt_trapeze,self.bxt_i)
            self.drift_B = (sta_weight*self.drift_B + cur_weight*drift_B)/new_weight

        elif self.B_mode == 'trapeze-shift':
            # b_inst_i = self.drift_b(self.data.X_ito[t]-self.data.dX_minus[t])
            # self.bxt_isom = self.drift_b(data.X_isom[0])
            # self.bxt_isop = self.drift_b(data.X_isop[0])
            # b_inst_s = 0.5*(b_inst_i+self.drift_b(self.data.X_ito[t]+self.data.dX_plus[t]))
            self.bxt_trapeze_shift = 0.5*(self.bxt_isom+self.bxt_isop)    
            # return np.einsum('iam,ibm->ab',b_inst_s,b_inst_i)
            drift_B = np.einsum('iam,ibm->ab',self.bxt_trapeze_shift,self.bxt_isom)
            self.drift_B = (sta_weight*self.drift_B + cur_weight*drift_B)/new_weight

        else:
            raise KeyError("Wrong B_mode argument")

        # # The velocity fit coefficients are given by Stratonovich
        # v_scalarb = np.einsum('iam,im->a',self.bxt_s,self.data.Xdot[0])
        # self.v_scalarb = (sta_weight*self.v_scalarb + cur_weight*v_scalarb)/new_weight

        ### D_local (useful in case of stratonovich drift mode) 
        # diffusion_mode = 'Vestergaard'
        if self.strato_diff_mode == 'Vestergaard': 
            D_local = self.__D_Vestergaard__
        elif self.strato_diff_mode == 'ABP_Vestergaard':
            D_local = self.__D_ABP_Vestergaard__
        elif self.strato_diff_mode == 'ABP_CST':
            D_local = self.__D_ABP_CST__
        else:
            raise KeyError("Invalid diffusion_mode parameter: ",self.strato_diff_mode)
        
        ### Drift_mode

        if self.drift_mode == 'Ito':

            # The velocity fit coefficients are given by Stratonovich
            v_scalarb = np.einsum('iam,im->a',self.bxt_s,self.data.Xdot[0])
            self.v_scalarb = (sta_weight*self.v_scalarb + cur_weight*v_scalarb)/new_weight
            # The drift fit coefficients are given directly by Ito
            # integration of Binv x_dot(t) b(x(t)).
            #return np.einsum('iam,im->a',self.drift_b(self.data.X_ito[t]),self.data.dX_plus[t])/self.data.dt[t]
            phi_scalarb = np.einsum('iam,im->a',self.bxt_i,self.data.Xdot[0])
            self.phi_scalarb = (sta_weight*self.phi_scalarb + cur_weight*phi_scalarb)/new_weight


        elif self.drift_mode == 'Ito-shift':
                
            # The velocity fit coefficients are given by Stratonovich
            v_scalarb = np.einsum('iam,im->a',self.bxt_s,self.data.Xdot[0])
            self.v_scalarb = (sta_weight*self.v_scalarb + cur_weight*v_scalarb)/new_weight
            # return np.einsum('iam,im->a',self.drift_b(self.data.X_ito[t]-self.data.dX_minus[t]),self.data.dX_plus[t])/self.data.dt[t]
            phi_scalarb = np.einsum('iam,im->a',self.bxt_isom,self.data.Xdot[0])
            self.phi_scalarb = (sta_weight*self.phi_scalarb + cur_weight*phi_scalarb)/new_weight


        elif self.drift_mode == 'Stratonovich-old':
            
            # The velocity fit coefficients are given by Stratonovich
            v_scalarb = np.einsum('iam,im->a',self.bxt_s,self.data.Xdot[0])
            self.v_scalarb = (sta_weight*self.v_scalarb + cur_weight*v_scalarb)/new_weight
            self.gradbxt_i = self.drift_grad_b(data.X_ito[0])
            # if self.drift_projectors.is_crossdiffusing:
            # #int_D_grad_b = data.trajectory_integral(lambda t : np.einsum('imn,inia->ma',D_local(t),self.drift_projectors.grad_b(X[t])))
            # raise NotImplementedError("Cross-diffusing multi-particles inference not implemented yet - write to me if you need it.")
            int_D_grad_b = np.einsum('imn,inam->a',D_local(0),self.gradbxt_i)
            #if diffusion_mode == 'Vestergaard' or diffusion_mode == 'ABP_Vestergaard': # only Veestergaard
            self.int_D_grad_b = (sta_weight*self.int_D_grad_b + cur_weight*int_D_grad_b)/new_weight           
            #self.phi_projections = (statweight*self.phi_projections + newweight*phi_projections)/(statweight+newweight)


        elif self.drift_mode == 'Stratonovich':
            
            # The velocity fit coefficients are more complicated
            # def bv(t): return np.einsum('iam,im->a',0.5*(self.drift_b(self.data.X_ito[t])+self.drift_b(self.data.X_ito[t]+self.data.dX_plus[t])),self.data.dX_plus[t])/self.data.dt[t]
            # v_scalarb = np.einsum('iam,im->a',0.5*(self.bxt_i+self.bxt_isop),self.data.Xdot[0])
            v_scalarb = np.einsum('iam,im->a',0.5*(self.bxt_i+self.bxt_isop),self.data.Xdot[0])
            self.v_scalarb = (sta_weight*self.v_scalarb + cur_weight*v_scalarb)/new_weight
            self.gradbxt_i = self.drift_grad_b(data.X_ito[0])
            # if self.drift_projectors.is_crossdiffusing:
            # #int_D_grad_b = data.trajectory_integral(lambda t : np.einsum('imn,inia->ma',D_local(t),self.drift_projectors.grad_b(X[t])))
            # raise NotImplementedError("Cross-diffusing multi-particles inference not implemented yet - write to me if you need it.")
            #  int_D_grad_b = self.data.trajectory_average(lambda t : np.einsum('imn,inam->a',D_local(t),self.drift_grad_b(X[t])))
            int_D_grad_b = np.einsum('imn,inam->a',D_local(0),self.gradbxt_i)
            #if diffusion_mode == 'Vestergaard' or diffusion_mode == 'ABP_Vestergaard': # only Veestergaard
            self.int_D_grad_b = (sta_weight*self.int_D_grad_b + cur_weight*int_D_grad_b)/new_weight           
            #self.phi_projections = (statweight*self.phi_projections + newweight*phi_projections)/(statweight+newweight)
    
        ##### diffusion #####

        ### Initialization
        if sta_weight == 0:

            self.D_average = np.zeros((data.d,data.d))
            self.Lambda = np.zeros((data.d,data.d))
            self.D_average_inv = np.zeros((data.d,data.d))

            # data_diffusion = { 'weight': [], 'D_average' : [], 'Lambda' : [] }
            # self.SFI_DicoDat['data_diffusion'] = data_diffusion

        diffusion_method = self.diffusion_mode
        # Select the (noisy) local diffusion matrix estimator:
        
        if diffusion_method == 'MSD':
            D_local = self.__D_MSD__
            #X = lambda t : self.data.X_strat[t]
            #self.diffusion_error_factor = 1
        
        elif diffusion_method == 'Vestergaard':
            D_local = self.__D_Vestergaard__
            # "Smooth" integration style (TODO: try to remember why
            # this is good...)
            #X = lambda t : self.data.X_ito[t] #+ (self.data.dX_plus[t]-self.data.dX_minus[t])/3. 
            #self.diffusion_error_factor = 4
        
        elif diffusion_method == 'WeakNoise': 
            D_local = self.__D_WeakNoise__
            #X = lambda t : self.data.X_ito[t]
            #self.diffusion_error_factor = 2
       
        else:
            raise KeyError("Wrong diffusion_method argument:",diffusion_method)

        D_average = np.einsum('t,tmn->mn',self.data.dt,np.array([ np.einsum('imn->mn', D_local(t) ) for t in range(len(self.data.t)) ]))/self.data.tauN
        self.D_average = (sta_weight*self.D_average + cur_weight*D_average) / new_weight
        
        Lambda = np.einsum('t,tmn->mn',self.data.dt,np.array([ np.einsum('imn->mn', self.__Lambda__(t) ) for t in range(len(self.data.t)) ]))/self.data.tauN
        self.Lambda = (sta_weight*self.Lambda +  cur_weight*Lambda) / new_weight

        #### Update quantities ####

        self.stat_weight = new_weight # new weight 
        self.frame_weight += data.N

        # compute pair stat weight
        cpt = 0
        for i in range(len(data.X_ito_neig[0])):
            if data.X_probe[0][i]:
                cpt += len(data.X_ito_neig[0][i]) 
        self.pair_stat_weight += cpt

        # # self.SFI_DicoDat['data_drift']['weight'].append(self.drift_stat_weight)
        # #self.SFI_DicoDat['data_drift']['v_coefficients'].append(self.v_coefficients)
        # #self.SFI_DicoDat['data_drift']['phi_coefficients'].append(self.phi_coefficients[:2,-2:])
        # #self.SFI_DicoDat['data_drift']['w_coefficients'].append(self.w_coefficients) 
        # #self.SFI_DicoDat['data_drift']['B'].append(self.drift_projectors.B)

    def pinv(self):

        ##### process drift with pinv method ####

        self.drift_B_inv = np.linalg.pinv(self.drift_B)
        # integration of B_inv x_dot(t) b(x(t)).
        self.v_coefficients = np.einsum('a,ab->b',self.v_scalarb, self.drift_B_inv ) 
        self.v_ansatz = lambda x : np.einsum('iam,a->im',self.drift_b(x), self.v_coefficients) 

        if self.drift_mode == 'Ito' or self.drift_mode == 'Ito-shift' :
            
            # The drift fit coefficients are given directly by Ito
            # integration of Binv x_dot(t) b(x(t)).
            self.phi_coefficients = np.einsum('a,ab->b',self.phi_scalarb, self.drift_B_inv ) 
            self.phi_ansatz = lambda x : np.einsum('iam,a->im',self.drift_b(x), self.phi_coefficients) 
            # We then estimate w by difference between phi and v
            # (generally not very useful, but comes for free).
            self.w_coefficients = self.phi_coefficients - self.v_coefficients
            self.w_ansatz = lambda x : np.einsum('iam,a->im',self.drift_b(x), self.w_coefficients)

        elif self.drift_mode == 'Stratonovich-old' or self.drift_mode == 'Stratonovich' :
            
            self.w_coefficients = - np.einsum('a,ab->b', self.int_D_grad_b,self.drift_B_inv)
            self.w_ansatz = lambda x : np.einsum('iam,a->im',self.drift_b(x), self.w_coefficients) 
            # Reconstruct the drift phi_mu = v_mu + w_mu
            self.phi_coefficients = self.w_coefficients + self.v_coefficients 
            self.phi_ansatz = lambda x : np.einsum('iam,a->im',self.drift_b(x), self.phi_coefficients)
        
        # return conditional number 
        return np.linalg.cond(self.drift_B)
    
    def tiko(self,alpha,opt):

        # method = 'tikov1'
        # method = 'tikov2'

        if opt == 'v1':
            ## process drift v1
            ndim = np.shape(self.drift_B)[0]
            self.drift_B_inv = np.linalg.pinv( self.drift_B+alpha*np.diag(np.ones(ndim)) )
            self.v_coefficients = self.v_scalarb @ self.drift_B_inv  
            if self.drift_mode == 'Ito' or self.drift_mode == 'Ito-shift' :
                self.phi_coefficients = self.phi_scalarb @ self.drift_B_inv  
            elif self.drift_mode == 'Stratonovich-old' or self.drift_mode == 'Stratonovich' :
                self.phi_coefficients = (- self.int_D_grad_b + self.v_scalarb) @ self.drift_B_inv 

        elif opt == 'v2':
            ## process drift v2
            if self.drift_mode == 'Ito' or self.drift_mode == 'Ito-shift' :
                self.phi_coefficients = scipy.sparse.linalg.lsqr(self.drift_B.T,self.phi_scalarb,damp=alpha)[0]
            elif self.drift_mode == 'Stratonovich-old' or self.drift_mode == 'Stratonovich' :
                self.phi_coefficients = scipy.sparse.linalg.lsqr(self.drift_B.T,self.v_scalarb-self.int_D_grad_b,damp=alpha)[0] 

        # nx, na, norm, list_alpha, list_phicof
        nx = np.linalg.norm(self.phi_coefficients)
        na = np.linalg.norm(self.phi_coefficients@self.drift_B-self.phi_scalarb)
        
        #  nx = [ np.linalg.norm(phicof) for phicof in list_phicof ]
        #  na = [ np.linalg.norm(phicof@self.drift_B-self.phi_scalarb) for phicof in list_phicof ]
        #  # select alpha_0
        #  invmod['alpha'] = 0
        #  self.tikov2(invmod)
        #  # import matplotlib.pyplot as plt; plt.figure(); plt.loglog(na,nx,'o')
        #  # plt.figure()
        #  # plt.loglog(na,nx,'o')
        #  # plt.figure()
        #  # plt.loglog(alpha,na,'o')
        #  # plt.loglog(alpha,nx,'o')

        return nx, na

    def additionS(self,Si):

        ## add two S objects
        # refresh stat weight for S0
        sw = self.stat_weight 
        cw = Si.stat_weight 
        nw = cw + sw  
        # frame weight for S0
        fsw = self.frame_weight 
        fcw = Si.frame_weight 
        fnw = fcw + fsw    
        # pair new weight for S0
        psw = self.pair_stat_weight 
        pcw = Si.pair_stat_weight
        pnw = pcw + psw   # new weight for S0 

        self.drift_B = ( sw*self.drift_B + cw*Si.drift_B ) / nw
        self.v_scalarb = ( sw*self.v_scalarb + cw*Si.v_scalarb ) / nw
        if self.drift_mode == 'Ito':
            self.phi_scalarb = ( sw*self.phi_scalarb + cw*Si.phi_scalarb ) / nw
        elif self.drift_mode == 'Stratonovich':
            self.int_D_grad_b = ( sw*self.int_D_grad_b + cw*Si.int_D_grad_b ) / nw           
        self.D_average = (sw*self.D_average + cw*Si.D_average) / nw
        self.Lambda = (sw*self.Lambda +  cw*Si.Lambda) / nw
        
        self.stat_weight = nw
        self.frame_weight = fnw
        self.pair_stat_weight = pnw
        # histogram
        self.histo += Si.histo 

    def print_report(self):
        """ Tell us a bit about yourself.
        """
        print("             ")
        print("  --- StochasticForceInference report --- ")
        print("Average diffusion tensor:\n",self.D_average)
        print("Measurement noise tensor:\n",self.Lambda)
        
        # if hasattr(self,'DeltaS'):
        #     print("Entropy production: inferred/bootstrapped error",self.DeltaS,self.error_DeltaS)
        # if hasattr(self,'drift_projections_self_consistent_error'):
        #     print("Drift information: inferred/bootstrapped error",self.drift_information,self.error_drift_information)
        #     print("Drift: squared typical error on projections:",self.drift_projections_self_consistent_error)
        #     print("  - due to trajectory length:",self.drift_trajectory_length_error)
        #     print("  - due to discretization:",self.drift_discretization_error_bias)
        # if hasattr(self,'diffusion_projections_self_consistent_error'):
        #     print("Diffusion: squared typical error on projections:",self.diffusion_projections_self_consistent_error)
        #     print("  - due to trajectory length:",self.diffusion_trajectory_length_error)
        #     print("  - due to discretization:",self.diffusion_discretization_error_bias)
        #     if hasattr(self,'diffusion_drift_error'):
        #         print("  - due to drift:",self.diffusion_drift_error)


    # Local diffusion estimators. All these are local-in-time noisy
    # estimates of the diffusion tensor (noise is O(1)). Choose it
    # adapted to the problem at hand.
    def __D_MSD__(self,t):
        return np.einsum('im,in->imn',self.data.dX_plus[t],self.data.dX_plus[t])/(2*self.data.dt[t])
    
    def __D_Vestergaard__(self,t):
        # Local estimator inspired by "Vestergaard CL, Blainey PC,
        # Flyvbjerg H (2014). Optimal estimation of diffusion
        # coefficients from single-particle trajectories. Physical
        # Review E 89(2):022726.".
        #
        # It is unbiased with respect to measurement noise, at the
        # cost of a 4x slower convergence. Use this estimator if
        # measurement noise is the limiting factor on inferring
        # D. Note that the error is minimized when symmetrizing the
        # correction term and integrating in Ito, i.e. evaluating the
        # projector at the initial point of the interval.
        return (np.einsum('im,in->imn',self.data.dX_plus[t]+self.data.dX_minus[t],self.data.dX_plus[t]+self.data.dX_minus[t])
            +   np.einsum('im,in->imn',self.data.dX_plus[t],self.data.dX_minus[t])
            +   np.einsum('im,in->imn',self.data.dX_minus[t],self.data.dX_plus[t]))  /(4*self.data.dt[t])

    def __D_ABP_Vestergaard__(self,t):
        a = (np.einsum('im,in->imn',self.data.dX_plus[t]+self.data.dX_minus[t],self.data.dX_plus[t]+self.data.dX_minus[t])
            +   np.einsum('im,in->imn',self.data.dX_plus[t],self.data.dX_minus[t])
            +   np.einsum('im,in->imn',self.data.dX_minus[t],self.data.dX_plus[t]))/(4*self.data.dt[t])
        b = np.tensordot(np.ones(self.data.Nparticles[t]),self.matdif,axes=0)
        #b = np.tensordot(np.ones(self.data.Nparticles[t]),np.array([[0,0,0],[0,0,0],[0,0,0]]),axes=0)
        return a*b

    def __D_ABP_CST__(self,t):
        b = np.tensordot(np.ones(self.data.Nparticles[t]),self.matdif,axes=0)
        #b = np.tensordot(np.ones(self.data.Nparticles[t]),np.array([[0,0,0],[0,0,0],[0,0,0]]),axes=0)
        return b

    def __Lambda__(self,t,use_v=False):
        # Lambda term is a local estimator for the measurement
        # error. It is valid only in the weak drift limit;
        # specifically, if eta is the random localization error, then
        #
        # <Lambda_munu> = <eta_mu eta_nu> - dt^2 <F_mu F_nu>
        #
        # i.e. it results in an underestimate (and can even be
        # negative) if dt is large.
        L = - (np.einsum('im,in->imn',self.data.dX_plus[t],self.data.dX_minus[t]) + np.einsum('im,in->imn',self.data.dX_minus[t],self.data.dX_plus[t]))/2
        if use_v:
            v = self.v_ansatz(self.data.X_ito[t])
            L += self.data.dt[t]**2 * np.einsum('im,in->imn',v,v)
        return  L

    def histo_polar_addstat( self, data ): 

        list_X = data.X_ito
        Nframe = len( list_X )
        
        for h in range(Nframe):

            X = list_X[h][0]
            X_neig = list_X[h][1]
            X_probe = list_X[h][2]

            # neighbour
            for i, lneigi in enumerate(X_neig):    
                if X_probe[i]:
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
                        if self.histo_vecr[0] <= d_ij < self.histo_vecr[-1]:
                            ir = np.where(d_ij-self.histo_vecr >= 0, d_ij-self.histo_vecr, np.inf).argmin()
                            iai = np.where(ai-self.histo_veca >= 0, ai-self.histo_veca, np.inf).argmin()
                            iaj = np.where(aj-self.histo_veca >= 0, aj-self.histo_veca, np.inf).argmin()
                            self.histo[0][ir,iai,iaj] += 1

