import numpy as np
from sfiabp.base import base2ptrigo


def minimalmodel3(Arep = 50, Adip = 4, R = 1.5, V = 10, C = -20 ):
    
    def vr(r,ai,aj):
        v00 = Arep/r**2
        v01 = V*R**3*np.cos(aj)/(r**3)
        vp1p1 = -(9/2)*Adip*np.cos(ai+aj)/(r**4)
        vp1m1 = (3/2)*Adip*np.cos(ai-aj)/(r**4)
        return v00+v01+vp1p1+vp1m1

    def vtheta(r,ai,aj):
        v01 = -(1/2)*V*R**3*np.sin(aj)/(r**3)
        return v01 + 0*ai
    
    def vomega(r,ai,aj):
        v10 = C*np.sin(ai)/r**2 + 0*aj
        vp1p1 = 6*Adip*(R**2)*np.sin(ai+aj)/(r**3)
        vp1m1 = -2*Adip*(R**2)*np.sin(ai-aj)/(r**3)
        return v10+vp1p1+vp1m1
    
    return [ vr, vtheta, vomega ]


def function1rN(k,pow):
    
    def vr(r,ai,aj):
        v = k/r**pow
        return v + 0*ai + 0*aj
    def vtheta(r,ai,aj):
        return 0*r + 0*ai + 0*aj
    def vomega(r,ai,aj):
        return 0*r + 0*ai + 0*aj
    
    return [ vr, vtheta, vomega ]


def functionNull():
    def vr(r,ai,aj):
        return 0*r + 0*ai + 0*aj    
    def vtheta(r,ai,aj):
        return 0*r + 0*ai + 0*aj
    def vomega(r,ai,aj):        
        return 0*r + 0*ai + 0*aj
    return [vr, vtheta, vomega]


#######################
## sfi projection 1d ##
#######################


def scalarprod(f,g,x):
    s = sum(f(x)*g(x))
    return s


def Gram(Base,x):
    
    nb = len(Base)
    m = np.zeros((nb,nb))

    for i in range(nb):
        for j in range(nb):
            m[i,j] = scalarprod(Base[i],Base[j],x)
    
    return m


def ResFuncDef(vec,Base) :
    nb = len(Base)
    def ResFunc(x):
        r = 0
        for i in range(nb):
            r += vec[i]*Base[i](x)
        return r
    return ResFunc


def sfiprod1d(fun,lbase,xsamp):

    nb = len(lbase)
    vscalb = np.zeros(nb)
    for i in range(nb):
        vscalb[i] = scalarprod(fun,lbase[i],xsamp)

    G = Gram(lbase,xsamp)
    G_inv = np.linalg.pinv(G)
    vcof = np.einsum('a,ab->b',vscalb, G_inv ) 
    
    resfun = ResFuncDef(vcof,lbase)
    return resfun, vcof


def sfiprod3d( lfun, lbase, vecr, vecai, vecaj, Mode_ACCL = False ): 

    ndim = len(lbase) # dimension (must be 3)
    nfun = len(lbase[0]) # number of basis function per dimension
    grid_pol = mesh_create(vecr,vecai,vecaj) # mesh creation
    # grid_car = mesh_pol2car(grid_pol)
    mask = np.ones(np.shape(grid_pol)[1:]) # mask 

    if Mode_ACCL:
        
        vscalb_acc = np.zeros((ndim,nfun))
        gmat_acc = np.zeros((ndim,nfun,nfun))
        vcof = np.zeros((ndim,nfun))

        # preprocess (consume a lot of memory)
        matfun = mesh_force(lfun,grid_pol)
        print('Preprocess function')
        
        matbase = [ mesh_force(lbase[i],grid_pol) for i in range(ndim) ]
        print('Preprocess base function')
        
        dd = np.prod(np.shape(matbase))
        print( 'memory = ' + str(dd) )

        # for each dimension
        for i in range(ndim):
            
            vscalb_acc[i,:] = prods3D_funbase_acc(matfun[i],matbase[i]*mask)
            gmat_acc[i,:,:] = prods3D_gram_acc(matbase[i]*mask)
            G_inv = np.linalg.pinv(gmat_acc[i,:,:])
            vcof[i,:] = np.einsum('a,ab->b',vscalb_acc[i,:], G_inv ) 
            print('Component : ' + str(i) + ', done')

        del matfun, matbase, vscalb_acc, gmat_acc, G_inv

    else:

        vscalb = np.zeros((ndim,nfun))
        gmat = np.zeros((ndim,nfun,nfun))
        vcof = np.zeros((ndim,nfun))

        # for each dimension
        for i in range(ndim):
            
            vscalb[i,:] = prods3D_funbase(lfun[i],lbase[i],grid_pol,mask)
            gmat[i,:,:] = prods3D_gram(lbase[i],grid_pol,mask)
            G_inv = np.linalg.pinv(gmat[i,:,:])
            vcof[i,:] = np.einsum('a,ab->b',vscalb[i,:], G_inv ) 

            print( 'Component : ' + str(i) + ' processd' )
        
        del vscalb, gmat, G_inv

    # # reshape coefficient based on type_drift_basis 
    # if type_drift_basis == 'Trigo':
    #     vcofres = np.reshape(vcof,np.shape(vcof)[0]*np.shape(vcof)[1])
    #     tablecof = Sfi_Vec_Lib_Base_Trigo.TocofCat(Pair_Trigo,vcofres)

    return vcof


############################################
## Single Functions 
############################################


def f_pol2car(r,ai,aj):
    veccar = np.zeros(3)
    ar = np.mod(2*np.pi-aj,2*np.pi)
    veccar[0] = r*np.cos(ar) # x (um)
    veccar[1] = r*np.sin(ar) # y (um)
    veccar[2] = np.mod(ar+ai,2*np.pi)# orientation thetai / ex (rad)    
    return veccar

# Sfi_Vec_Script_ScalarProduct_2D
def mesh_create(rv,aiv,ajv):
    
    # dim of r / dim of ai / dim of av / r,ai,aj
    tab = np.zeros((3,len(rv),len(aiv),len(ajv)))

    for ir in range(len(rv)):
        for iai in range(len(aiv)):
            for iaj in range(len(ajv)):
                tab[0,ir,iai,iaj] = rv[ir] # distance (mum)
                tab[1,ir,iai,iaj] = aiv[iai] # angle ai (rad)
                tab[2,ir,iai,iaj] = ajv[iaj] # angle aj (rad)

    return tab

def mesh_pol2car(grid_pol):
    
    # assumption : nj colinear with ex
    grid_car = np.zeros(np.shape(grid_pol))

    for i in range(np.shape(grid_pol)[0]):
        for j in range(np.shape(grid_pol)[1]):
            for k in range(np.shape(grid_pol)[2]):
                grid_car[i,j,k,:] = f_pol2car(grid_pol[i,j,k,0],grid_pol[i,j,k,1],grid_pol[i,j,k,2])
    
    return grid_car

def mesh_force(f,grid):

    # f must be a list of function
    nf = len(f)
    gforce = np.zeros(( nf, np.shape(grid)[1], np.shape(grid)[2], np.shape(grid)[3] ))
    
    for h in range(nf):
        for i in range(np.shape(grid)[1]):
            for j in range(np.shape(grid)[2]):
                for k in range(np.shape(grid)[3]):
                    gforce[h,i,j,k] = f[h](grid[0][i,j,k],grid[1][i,j,k],grid[2][i,j,k])
                
    return gforce


############################################
## Granick base function 
############################################


def KDiel(f):
    # 1 st order relaxation
    epsinf = -0.5
    eps0 = 0.25
    fc = 3e4 # cut off frequency (Hz)
    v = epsinf + (eps0-epsinf)/(1-1j*(f/fc))
    return v


def KMett(f):
    # 1 st order relaxation
    epsinf = 0.9
    eps0 = -0.5
    fc = 1e4 # cut off frequency (Hz)
    v = epsinf + (eps0-epsinf)/(1+1j*(f/fc))
    return v

# Sfi_Vec_Script_ScalarProduct_2D
def Granick_Hz(p):

    # return the Granick function in cartesian coordinates

    # permittivity constant (C2/(Nm2))
    eps = 6.95*1e-10
    # Amplitude Voltage (Alexis : 6,8,10 /2)
    v = p['vpp']/2
    # lambda width between electrodes (m)
    tick = 1e-6*p['spacer']
    # Amplitude Electric field (V/um)
    E0 = v/tick
    # radius particle (m)
    R = 1e-6*p['d']/2
    # hemisphere (um)
    l = 1e6*3*R/8
    # translational drag (Pa.s.um)
    ksit = 25*1e-3
    # rotational drag (Pa.s.um3)
    ksir = 75*1e-3
    # constant 
    C = 2*np.pi*eps*R**3*E0 
    CD = 3*1e36/(4*np.pi*eps*ksit)

    # coefficient 
    if p['Diel_head']:
        Kh = KDiel(p['f'])
        Kt = KMett(p['f'])
        dh2 = CD*(C*np.abs(Kh))**2
        dt2 = CD*(C*np.abs(Kt))**2
        dhdt = CD*C**2*np.real(np.conjugate(Kh)*Kt)
    else:
        Kh = KMett(p['f'])
        Kt = KDiel(p['f'])
        dh2 = CD*(C*np.abs(Kh))**2
        dt2 = CD*(C*np.abs(Kt))**2
        dhdt = CD*C**2*np.real(np.conjugate(Kh)*Kt)

    def f(x,y,ti,tj=0):

        # orientation vector j and i
        angle_j = tj
        vTheta_j = np.array([np.cos(angle_j),np.sin(angle_j)])
        angle_i = ti
        vTheta_i = np.array([np.cos(angle_i),np.sin(angle_i)])

        # position vectors
        Xij = np.array([x,y]) # j -> i
        Rhh = -l*vTheta_j + Xij + l*vTheta_i
        Rht = -l*vTheta_j + Xij - l*vTheta_i
        Rth =  l*vTheta_j + Xij + l*vTheta_i
        Rtt =  l*vTheta_j + Xij - l*vTheta_i

        # Force ́& Torque flow
        Fhh = dh2 * Rhh/np.linalg.norm(Rhh)**(5)
        Fht = dhdt * Rht/np.linalg.norm(Rht)**(5)
        Fth = dhdt * Rth/np.linalg.norm(Rth)**(5)
        Ftt = dt2 * Rtt/np.linalg.norm(Rtt)**(5)
        
        # (Pa.um2) / (Pa.s.um) = um / s
        F = Fhh + Fht + Fth + Ftt  
        # (Pa.um3) / (Pa.s.um3)
        T = (ksit/ksir) * np.cross( l*vTheta_i, Fhh + Fth ) + np.cross( -l*vTheta_i, Ftt + Fht ) 

        gforce = np.concatenate((F,[T]))
        
        return gforce
    
    return f

# Sfi_Vec_Script_ScalarProduct_2D
def Granick_Func(gparam):

    # return the Granick function in cartesian coordinates
    # (same function as Granick_Hz except gparam includes directly the Granick coef)
    
    l = 3*gparam['r']/8
    dh = gparam['dh']
    dt = gparam['dt']
    kto = gparam['kto']
    dh2 = dh**2
    dt2 = dt**2
    dhdt = gparam['dhdt']

    def f(x,y,ti,tj=0):

        # orientation vector j and i
        angle_j = tj
        vTheta_j = np.array([np.cos(angle_j),np.sin(angle_j)])
        angle_i = ti
        vTheta_i = np.array([np.cos(angle_i),np.sin(angle_i)])

        # position vectors
        Xij = np.array([x,y]) # j -> i
        Rhh = -l*vTheta_j + Xij + l*vTheta_i
        Rht = -l*vTheta_j + Xij - l*vTheta_i
        Rth =  l*vTheta_j + Xij + l*vTheta_i
        Rtt =  l*vTheta_j + Xij - l*vTheta_i

        # Force ́& Torque flow
        Fhh = dh2 * Rhh/np.linalg.norm(Rhh)**(5)
        Fht = dhdt * Rht/np.linalg.norm(Rht)**(5)
        Fth = dhdt * Rth/np.linalg.norm(Rth)**(5)
        Ftt = dt2 * Rtt/np.linalg.norm(Rtt)**(5)
        
        F = Fhh + Fht + Fth + Ftt  
        T = kto * np.cross( l*vTheta_i, Fhh + Fth ) + np.cross( -l*vTheta_i, Ftt + Fht ) 

        gforce = np.concatenate((F,[T]))
        
        return gforce
    
    return f

# Sfi_Vec_Script_ScalarProduct_2D
def Convert_to_polar(lgra):

    MatRot = lambda A : np.array([ [np.cos(A),-np.sin(A)],[np.sin(A),np.cos(A)] ]) # rotation matrix

    # def fvec(r,ai,aj):    
    #     # apply Granick function 
    #     vcar = f_pol2car(r,ai,aj)
    #     vforce_car = lgra(vcar[0],vcar[1],vcar[2])
    #     # convert to er, etheta, torque frame
    #     vforce_pol = np.zeros(np.shape(vforce_car))        
    #     ar = np.mod(2*np.pi-aj,2*np.pi)
    #     vforce_pol[:2] = np.matmul(MatRot(-ar),vforce_car[:2]) 
    #     vforce_pol[2] = vforce_car[2] 
    #     return vforce_pol
    
    def fer(r,ai,aj):
        
        # apply Granick function 
        vcar = f_pol2car(r,ai,aj)
        vforce_car = lgra(vcar[0],vcar[1],vcar[2])
        ar = np.mod(2*np.pi-aj,2*np.pi)

        return np.matmul(MatRot(-ar),vforce_car[:2])[0]
    
    def fet(r,ai,aj):
        
        # apply Granick function 
        vcar = f_pol2car(r,ai,aj)
        vforce_car = lgra(vcar[0],vcar[1],vcar[2])       
        ar = np.mod(2*np.pi-aj,2*np.pi)
       
        return np.matmul(MatRot(-ar),vforce_car[:2])[1]
    
    def fto(r,ai,aj):
        
        # apply Granick function 
        vcar = f_pol2car(r,ai,aj)
        vforce_car = lgra(vcar[0],vcar[1],vcar[2])

        return vforce_car[2]

    return [fer, fet, fto]

def create_bench_fun():
    def fer(d_ij,ai,aj):
            return np.cos(ai)*(20/d_ij**2)
    def fet(d_ij,ai,aj):
            return 0
    def fto(d_ij,ai,aj):
            return 0
    return [fer,fet,fto]

def create_granick_trigo(gparam):

    l = 3*gparam['r']/8
    dh = gparam['dh']
    dt = gparam['dt']
    kto = gparam['kto']
    dh2 = dh**2
    dt2 = dt**2
    dhdt = gparam['dhdt']

    # radial force
    ler = [ [] for i in range(7) ]
    ler[0] = lambda r : (dh+dt)**2/r**4 # 00 ai,aj
    ler[1] = lambda r : 4*l*(dt2-dh2)/r**5 # 10 ai,aj
    ler[2] = lambda r : 4*l*(dh2-dt2)/r**5 # 01 ai,aj
    ler[3] = lambda r : -(15/2)*l**2*(dh-dt)**2/r**6 # 1-1 ai,aj
    ler[4] = lambda r : (25/4)*l**2*(dh+dt)**2/r**6 # 20 ai,aj
    ler[5] = lambda r : -(25/2)*l**2*(dh-dt)**2/r**6 # 11 ai,aj
    ler[6] = lambda r : (25/4)*l**2*(dh+dt)**2/r**6 # 02 ai,aj

    # ortho radial force
    let = [ [] for i in range(7) ]
    let[0] = lambda r : 0 # 00 ai,aj
    let[1] = lambda r : l*(dh2-dt2)/r**5 # 10 ai,aj
    let[2] = lambda r : l*(dt2-dh2)/r**5 # 01 ai,aj
    let[3] = lambda r : 0 # 1-1 ai,aj
    let[4] = lambda r : -(5/2)*l**2*(dh+dt)**2/r**6 # 20 ai,aj
    let[5] = lambda r : 5*l**2*(dh-dt)**2/r**6 # 11 ai,aj
    let[6] = lambda r : -(5/2)*l**2*(dh+dt)**2/r**6 # 02 ai,aj
    
    # torque
    lto = [ [] for i in range(7) ]
    lto[0] = lambda r : 0 # 00 ai,aj
    lto[1] = lambda r : kto*l*(dt2-dh2)/r**4 # 10 ai,aj
    lto[2] = lambda r : 0 # 01 ai,aj
    lto[3] = lambda r : -(3/2)*kto*l**2*(dh-dt)**2/r**5 # 1-1 ai,aj
    lto[4] = lambda r : (5/2)*kto*l**2*(dh+dt)**2/r**5 # 20 ai,aj
    lto[5] = lambda r : -(5/2)*kto*l**2*(dh-dt)**2/r**5 # 11 ai,aj
    lto[6] = lambda r : 0 # 02 ai,aj

    return [ler,let,lto]

#######################
## sfi projection 2d ##
#######################

# Sfi_Vec_Lib_Analytics
def prods3D_scalar(f,g,x,mask):

    cpt = 0
    for i in range(np.shape(x)[1]):
        for j in range(np.shape(x)[2]):
            for k in range(np.shape(x)[3]):
                cpt += (f(x[0,i,j,k],x[1,i,j,k],x[2,i,j,k])
                            *g(x[0,i,j,k],x[1,i,j,k],x[2,i,j,k])
                                *mask[int(x[0,i,j,k]),int(x[1,i,j,k]),int(x[2,i,j,k])])
                 
    return cpt

# Sfi_Vec_Script_ScalarProduct_2D
def prods3D_funbase(lfun,lbase,grid,mask):

    nbase = len(lbase)
    vscalb = np.zeros(nbase)
    for i in range(nbase):
        vscalb[i] = prods3D_scalar(lfun,lbase[i],grid,mask)
        print(str(i))
    
    return vscalb

# Sfi_Vec_Script_ScalarProduct_2D
def prods3D_funbase_acc(matfun,matbase):
    vscalb = np.einsum('ijk,mijk->m',matfun,matbase)
    return vscalb

# Sfi_Vec_Script_ScalarProduct_2D
def prods3D_gram(lb,x,mask):
    
    nb = len(lb)
    m = np.zeros((nb,nb))

    for i in range(nb):
        for j in range(nb):
            m[i,j] = prods3D_scalar(lb[i],lb[j],x,mask)
            print( '( '+ str(i) + ' ' +str(j) + ' )' )
    
    return m

# Sfi_Vec_Script_ScalarProduct_2D
def prods3D_gram_acc(matbase):
    g = np.einsum('aijk,bijk->ab',matbase,matbase)    
    return g

def scalar_product_trigo_2D ( lfun, grid_pol, mask, Mode_ACCL, Base_Rad, Order, InfName):
            
    # r_vec = np.linspace(3,16,26,endpoint=False) # um
    # ai_vec = (np.pi/180)*np.linspace(0,360,36,endpoint=False) # rad
    # aj_vec = (np.pi/180)*np.linspace(0,360,36,endpoint=False) # rad
    # # Mesh creation
    # grid_pol = Sfi_Vec_Lib_Analytics.mesh_create(r_vec,ai_vec,aj_vec)

    ## Trigo basis
    Pair_Trigo = base2ptrigo.Rad2Trigo_FullCut( Base_Rad, Order = Order )
    lbase = Pair_Trigo['Base']
    ## SFI Algo
    ndim = len(lbase)
    nfun = len(lbase[0])

    if Mode_ACCL:
        
        vscalb_acc = np.zeros((ndim,nfun))
        gmat_acc = np.zeros((ndim,nfun,nfun))
        vcof = np.zeros((ndim,nfun))
        # preprocess (consume a lot of memory)
        matfun = mesh_force(lfun,grid_pol)
        print('Preprocess function')  
        matbase = [ mesh_force(lbase[i],grid_pol) for i in range(ndim) ]
        print('Preprocess base function') 
        dd = np.prod(np.shape(matbase))
        print( 'memory = ' + str(dd) )

        # for each dimension
        for i in range(ndim):
            
            vscalb_acc[i,:] = prods3D_funbase_acc(matfun[i],matbase[i]*mask)
            gmat_acc[i,:,:] = prods3D_gram_acc(matbase[i]*mask)
            G_inv = np.linalg.pinv(gmat_acc[i,:,:])
            vcof[i,:] = np.einsum('a,ab->b',vscalb_acc[i,:], G_inv ) 
            print('Component : ' + str(i) + ', done')

        del matfun, matbase, vscalb_acc, gmat_acc, G_inv

    else:

        vscalb = np.zeros((ndim,nfun))
        gmat = np.zeros((ndim,nfun,nfun))
        vcof = np.zeros((ndim,nfun))

        # for each dimension
        for i in range(ndim):
            
            vscalb[i,:] = prods3D_funbase(lfun[i],lbase[i],grid_pol,mask)
            gmat[i,:,:] = prods3D_gram(lbase[i],grid_pol,mask)
            G_inv = np.linalg.pinv(gmat[i,:,:])
            vcof[i,:] = np.einsum('a,ab->b',vscalb[i,:], G_inv ) 

            print( 'Component : ' + str(i) + ' processd' )
        
        del vscalb, gmat, G_inv

    # reshape coefficient based on type_drift_basis 
    vcofres = np.reshape(vcof,np.shape(vcof)[0]*np.shape(vcof)[1])
    tablecof = base2ptrigo.TocofCat(Pair_Trigo,vcofres)

    ## save results directly in a dico

    lp = {}
    lp['si'] = InfName
    lp['trial'] = 0
    lp['Base_Dim'] = Pair_Trigo['Base_Dim']
    lp['Nbase_Dim'] = Pair_Trigo['Nbase_Dim']
    lp['Base_Cat'] = Pair_Trigo['Base_Cat']
    # specific to 'Trigo'
    lp['Order']    = Pair_Trigo['Order']
    lp['Base_Rad'] = Pair_Trigo['Base_Rad']
    lp['list_term']    = Pair_Trigo['list_term']
    lp['list_term_num']    = Pair_Trigo['list_term_num']

    lco = tablecof
    # radial function for each trigo mode
    lfr = base2ptrigo.Init_ListFuncRad(Pair_Trigo['Base_Rad'],tablecof)
    # full 2D functions for each dimension
    lff = base2ptrigo.Init_ListFuncFull(Pair_Trigo['Base_Cat'],tablecof)

    Dic = { 'lp' : lp, 'lco' : lco, 'lfr' : lfr, 'lff' : lff } 
    return Dic



