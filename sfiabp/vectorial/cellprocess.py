import numpy as np

def basis_selector(lb1,lb2):
    
    """ convert the drift functions into an appropriate format for the sfi algo """
    
    # create the function bsingle
    bsingle = []
    if lb1:
        # bsingle = lambda X :  np.concatenate( [ i(X) for i in lb1 ] )
        bsingle = lambda X :  np.vstack( [ i(X) for i in lb1 ] )
    # create the function bpair
    bpair = []
    if lb2:
        bpair = lambda Xi,Xj :  np.vstack( [ i(Xi,Xj) for i in lb2 ] )
    # return funcs, grad
    if lb1 and not lb2 :
        funcs,grad = noninteracting_basis_cell( bsingle )
    elif lb1 and lb2 :
        funcs,grad = pair_interaction_basis_cell( bsingle, bpair )
    else:
        raise KeyError("Unknown basis type.")
    
    return funcs,grad

def pair_interaction_basis_cell(bsingle,bpair):

    # bpair is the pair interaction basis, function of Xi and Xj
    # bsingle is an optional single-particle field basis

    def C(XL,i):
        
        # implement cell mode
        X = XL[0] # XL[0] = frame
        Neig = XL[1] # XL[1] = table neighboorhood
        Probe = XL[2] # XL[2] = probe 

        single = bsingle(X[i]) 
        pair = [ bpair( X[i], X[i] ) ] # equivalent to bpairNull
        for j in Neig[i]:
            pair.append( bpair( X[i], X[j]) )
        pair = np.sum(pair, axis = 0 )

        return np.row_stack((single, pair))

    return interacting_particles_nocrossdiffusion_basis_cell(C)

### creation of the functions and their gradients

def interacting_particles_nocrossdiffusion_basis_cell(C,grad_C_i = None,epsilon = 1e-6):
    
    """ A generic basis for interacting particles, with C taking two
    arguments: C(xi,{xj}_j_neq_i,i), adapted to infer the force on
    particle i with form Fi = Falpha C(xi, {xj}). C can be composed
    of one-body, two-body or higher interactions. In practice C(X,i)
    receives the whole X array and the index i too so that no copy
    is needed. C returns an array of n scalars (each corresponding
    to a fitting function).
    
    Note that this is only for particles systems with no
    crossdiffusion, ie no D_ij_munu terms with i != j (allows for a
    massive simplification of the gradient evaluation).
    
    grad1_C is the gradient of C wrt xi, if unspecified it will be
    computed through finite differences with step epsilon. It
    returns a d x n array. """

    if grad_C_i is None:
        def grad_C_i(XL,i):
            Nparticles,dim = XL[0].shape
            # dx_vals = [ np.array([[ epsilon if j==i and mu==nu else 0 for mu in range(dim)] for j in range(Nparticles) ]) for nu in range(dim) ]
            # return np.array([ (C( [ XL[0]+dx,XL[1],XL[2],XL[3] ],i) - C( [ XL[0]-dx,XL[1],XL[2],XL[3] ] ,i))/(2*epsilon) for dx in dx_vals ])
            ## ABP_Vestergaard ansatz
            dx = np.zeros((Nparticles,dim)) 
            dx[i,2] = epsilon 
            l2 = (C( [XL[0]+dx,XL[1],XL[2]],i) - C( [XL[0]-dx,XL[1],XL[2]],i))/(2*epsilon)
            nb = l2.shape[0]
            return np.array([np.zeros((nb,3)),np.zeros((nb,3)),l2])

    def interacting_basis_function_cell(XL):
        # X is a Nparticles x dim - shaped array.
        # XL[0] : data, XL[1] : neig, XL[2] : probe  
        # Nparticles = XL[0].shape[0]
        # Nparticles = np.arange(len(XL[0]))
        # probe on (filtering option)
        Nparticles = np.arange(len(XL[0]))[XL[2] == 1]
        # Output has shape Nparticles x n
        return np.array([ C(XL,i) for i in Nparticles ])

    def interacting_basis_function_gradient_cell(XL):
        # Nparticles = XL[0].shape[0]
        # filtering option
        # ind = np.arange(Nparticles)
        # ind = ind[XL[2] == 1]
        Nparticles = np.arange(len(XL[0]))[XL[2] == 1]
        # Output has shape Nparticles x d x n
        return np.array([ grad_C_i(XL,i) for i in Nparticles ])
        # return np.array([ grad_C_i(XL,i) for i in ind ])

    return interacting_basis_function_cell, interacting_basis_function_gradient_cell

def noninteracting_basis_cell(C,grad_C = None,epsilon = 1e-6):
    
    """ Format a generic single-particle basis for use by SFI. The input
    is a function C(X) with X a d-dimensional array, and returns a
    n-dimensional array (the values of the n fit functions at X). If
    grad_C is provided it should return a d x n array, otherwise
    finite differences are used. """
    
    # The output is formatted to take 
    if grad_C is None:
        def grad_C(X):
            d = len(X)
            dx_vals = epsilon * np.identity(d) 
            return np.array([ (C(X+dx) - C(X-dx))/(2*epsilon) for dx in dx_vals ])

    def basis_function_cell(XL):
        X=XL[0]
        # X is a Nparticles x dim - shaped array.
        # Output has shape Nparticles x n
        return np.array([ C(x) for x in X ])

    def basis_function_gradient_cell(XL):
        X=XL[0]
        # Output has shape Nparticles x d x n
        return np.array([ grad_C(x) for x in X ])

    return basis_function_cell, basis_function_gradient_cell

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
        
# create_cell
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


