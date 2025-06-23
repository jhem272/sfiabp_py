
""" A library of projection bases for Stochastic Force Inference. """

from numpy import sqrt, arctan2, cos, sin, abs
import numpy as np

from scipy.special import factorial
# warning problem with math.factorial with large number (>21, dtype = object error)
# import math
# use PolyExpBig instead

##############################
##     Basis Function      ##
##############################

# radial base functions

def Gauss(b,v) :
    return lambda x : np.exp(-0.5*((x-b)/v)**2)

def Step(c,w): # work only with drift_mode = Ito
    xf = lambda x : 1*(x>=(c-w/2))*(x<(c+w/2))
    return xf

def PolyExp(k,r0) :
    return lambda x: (x/r0)**k*np.exp(-x/r0)/factorial(k)

def Inv(k) :
    return lambda x: 1/x**k

def PolyExpBig(k,r0) :

    def logfact(k):
        r, c = 0, 1
        while c <= k :
            r += np.log(c)
            c += 1 
        return r

    tab = [ logfact(i) for i in range(200) ]
    
    if k >= 200:
        return lambda x: np.exp(  k*np.log(x/r0) -x/r0 - logfact(k) )
    else:
        return lambda x: np.exp(  k*np.log(x/r0) -x/r0 - tab[k] )
        
# Sfi_Vec_Lib_Bases
def polartrigo(Order,FuncRad,VectorRad):

    if FuncRad == 'PolyExp':
        # Base_Rad = [ PolyExp(k,1) for k in np.arange(0,22,1) ]
        Base_Rad = [ PolyExp(k,1) for k in VectorRad ]
        # Base_Rad = [ PolyExp(k,1) for k in [0,1,2,3,4,5,6,7,8,10,12,14,16] ]
        # Base_Rad = [ PolyExp(k,1) for k in [0,2,4,6,8,10,12] ]
        # Base_Rad = [ PolyExp(k,1) for k in [0,2,4,6,8] ]

    elif FuncRad == 'Gauss' : 
        VectorDiff = np.diff(VectorRad)
        VectorDiff = np.hstack((VectorDiff,[VectorDiff[-1]]))
        Base_Rad = [ Gauss(VectorRad[i],2*VectorDiff[i]) for i in range(len(VectorRad)) ]
            
    elif FuncRad == 'Inv' :
        Base_Rad = [ Inv(k) for k in VectorRad ]
    
    elif FuncRad == 'Step' :
        VectorDiff = np.diff(VectorRad) 
        VectorCent = VectorRad[:-1] + VectorDiff/2
        Base_Rad = [ Step(VectorCent[i],VectorDiff[i]) for i in range(len(VectorDiff)) ]

    elif FuncRad == 'PolyExpBig':
        Wid = VectorRad[0]
        Center = VectorRad[1]
        VectorK = Center / Wid 
        Base_Rad = [ PolyExpBig(k,Wid) for k in VectorK.astype(int) ]

    # elif FuncRad == 'PolyExpWid':
    #     Wid = VectorRad[0]
    #     Center = VectorRad[1]
    #     VectorK = Center / Wid 
    #     Base_Rad = [ PolyExpBig(k,Wid) for k in VectorK.astype(int) ]

    def costrig(k,l,Base_Rad,m):
        return lambda d_ij,ai,aj : cos(k*aj+(l-abs(k))*ai)*Base_Rad[m](d_ij)

    def sintrig(k,l,Base_Rad,m):
        return lambda d_ij,ai,aj : sin(k*aj+(l-abs(k))*ai)*Base_Rad[m](d_ij)

    # number of radial base function
    nbrad = len(Base_Rad)
    # sequence order
    nl = np.arange(0,Order+1)
    # count
    cpt = 0

    # buil list of cos and sine functions 
    ncofsgl = np.sum([ 2*n for n in range(Order+1) ]) + 1
    # list 3 dim (x,y,t)
    lbase_cos = [ [] for j in range(ncofsgl) ]
    lbase_sin = [ [] for j in range(ncofsgl) ]
    # list of trigo name 
    term_num = np.zeros((ncofsgl,2)) 
    term_string = [ [] for j in range(ncofsgl) ]

    for l in nl:

            if l == 0:        
                
                for m in range(nbrad):
                    # er, etheta, torque
                    lbase_cos[cpt].append(costrig(0,0,Base_Rad,m))
                    lbase_sin[cpt].append(sintrig(0,0,Base_Rad,m)) 
                    # list of trigo name
                    name = str(0) + 'ai + ' + str(0) + 'aj' 
                    term_num[cpt,:] = [0,0]
                    term_string[cpt] = name

                cpt += 1

            else:

                for kk in range(2*l):

                    k = kk-l+1

                    for m in range(nbrad):              
                        # e_r, etheta, torque
                        lbase_cos[cpt].append(costrig(k,l,Base_Rad,m)) 
                        lbase_sin[cpt].append(sintrig(k,l,Base_Rad,m))
                        # list of trigo name
                        name = str(l-abs(k)) + 'ai + ' + str(k) + 'aj' 
                        term_num[cpt,:] = [l-abs(k),k]
                        term_string[cpt] = name

                    cpt += 1

    # er, etheta, etorque 
    lbasecat = [ lbase_cos, lbase_sin, lbase_sin ]
    
    # flat base list
    lbaseflat = [ [],[],[] ] 
    for i in range(len(lbasecat)):
        for j in range(len(lbasecat[0])):
            for k in range(len(lbasecat[0][0])):
                lbaseflat[i].append(lbasecat[i][j][k])

    return lbaseflat, lbasecat, Base_Rad, term_string, term_num


def Fij_Trigo_FullCut(S,R_bound,dr,da,SeqOrd=-1):

    # get info for the pair functions
    Order = S['Order']
    FuncRad = S['FuncRad']
    VectorRad = S['VectorRad']
    lbasecat = polartrigo( Order, FuncRad, VectorRad )[1]
    tablecof = TocofCat( np.shape(lbasecat), S['cof2p'][0] )
    
    ## choose specific coefficients 
    if SeqOrd == -1:
        # full order 
        lff = Init_ListFuncFull(lbasecat,tablecof) # get the 3D function
    elif SeqOrd == -2:
        # Dass / Alert 2 order
        tablecof_seqo = np.zeros( np.shape(tablecof) )
        tablecof_seqo[0,0,:] = tablecof[0,0,:] # vr, zero order
        tablecof_seqo[2,:,:] = tablecof[2,:,:] # omega, all second order
        lff = Init_ListFuncFull(lbasecat,tablecof_seqo) # get the 3D function
    elif SeqOrd == -3: 
        # Daas / Alert 1 order
        tablecof_seqo = np.zeros( np.shape(tablecof) )
        tablecof_seqo[0,0,:] = tablecof[0,0,:] # vr, zero order
        tablecof_seqo[2,1,:] = tablecof[2,1,:] # omega, only coef a10
        lff = Init_ListFuncFull(lbasecat,tablecof_seqo) # get the 3D function
    elif SeqOrd == -4:
        # Only alignement interaction
        tablecof_seqo = np.zeros( np.shape(tablecof) )
        tablecof_seqo[2,:,:] = tablecof[2,:,:] # omega, all second order
        lff = Init_ListFuncFull(lbasecat,tablecof_seqo) # get the 3D function
    else:
        # partial order
        notri = givetri(np.arange(0,SeqOrd+1))
        tablecof_seqo = tablecof[:,notri,:]
        lbasecat_seqo = polartrigo( SeqOrd, FuncRad, VectorRad )[1]
        lff = Init_ListFuncFull(lbasecat_seqo,tablecof_seqo)

    # grid
    # S.P['lcell']
    nr = int( np.round( ( (R_bound[1]-R_bound[0])/dr ) ) )
    na = int( np.round( ( 2*np.pi/da ) ) )
    vr = np.linspace(R_bound[0],R_bound[1],nr,endpoint=False)
    va = np.linspace(0,2*np.pi,na,endpoint=False)
    vm = np.meshgrid(vr,va,va,indexing='ij')
    
    meshforce = np.zeros(np.shape(vm))
    for i in range(3):
        meshforce[i] = lff[i](vm[0],vm[1],vm[2])
    
    def Func(Xij, Xi, Xj, d_ij):

        # Final vector 
        vecF = np.zeros(3)
        # angle (rad) of e_ij
        Ar = np.mod(np.arctan2(Xij[1],Xij[0]),2*np.pi)
        # angle (rad) of ith, jth particle
        Ai, Aj = Xi[2], Xj[2] 
        # angle (rad) of ith, jth particle
        ai, aj = np.mod(Ai-Ar,2*np.pi), np.mod(Aj-Ar,2*np.pi)

        # unit radial vector e_ij
        vRad = np.array([np.cos(Ar),np.sin(Ar),0])
        # unit orthoradial vector e_ij
        vOrtho = np.array([-np.sin(Ar),np.cos(Ar),0])
        # unit torque
        vTorque = np.array([0,0,1])
        
        if R_bound[0] <= d_ij < R_bound[1]:
            # get index 
            ir = int( np.floor_divide( d_ij-R_bound[0], dr ) )
            iai = int( np.floor_divide( ai, da ) )
            iaj = int( np.floor_divide( aj, da ) )
            # interaction (trigo functions)
            vecF += meshforce[0][ir,iai,iaj]*vRad
            vecF += meshforce[1][ir,iai,iaj]*vOrtho
            vecF += meshforce[2][ir,iai,iaj]*vTorque
            # vecF += meshforce[2][ir,iai,iaj]*vTorque
            # vecF += meshforce[0][ir,iai,iaj]*vRad
            # vecF += meshforce[1][ir,iai,iaj]*vOrtho
            # vecF += meshforce[1][ir,iai,iaj]*vOrtho
            # vecF += meshforce[2][ir,iai,iaj]*vTorque
            
        return vecF
    
    return Func


###################################
##     Additionnal Function      ##
###################################


# Sfi_Vec_Lib_Trigo
def TocofCat(Base_Dim,phi_cof):

    ndim = Base_Dim[0]
    ntri = Base_Dim[1]
    nrad = Base_Dim[2]
    mat = np.zeros((ndim,ntri,nrad))
    
    for i in range(len(phi_cof)):
        idim = np.floor_divide(i,ntri*nrad)
        itri = np.floor_divide(i-idim*ntri*nrad,nrad)
        irad = i-idim*ntri*nrad-itri*nrad
        mat[idim,itri,irad] = phi_cof[i]

    return mat


# Sfi_Vec_Plot_Trigo
def Init_ListFuncRad(ListBase_Radial,Table_Cof):
        
        ndim, ntri, nrad  = np.shape(Table_Cof) 

        def fij(i,j):            
            def f(x):
                # r = np.zeros(np.shape(x))
                # for k in range(nrad):
                #     # r += Table_Cof[i,j,k]*np.array(ListBase_Radial[k](x),dtype=float)
                tab = np.array([ Table_Cof[i,j,k]*ListBase_Radial[k](x) for k in range(nrad) ])
                return np.sum(tab,axis=0)
            return f

        ll = [[ [] for j in range(ntri) ] for i in range(ndim)] 

        for i in range(ndim):
            for j in range(ntri):
                ll[i][j] = fij(i,j)  
        
        return ll


def givetri(List_Ord):
        def lord(n):
            if n == 0: l = [0]
            elif n == 1: l = [1,2]
            elif n == 2: l = [3,4,5,6]
            elif n == 3: l = [7,8,9,10,11,12]
            elif n == 4: l = [13,14,15,16,17,18,19,20]
            return l
        list_tri = []
        for i in List_Ord:
            list_tri.extend(lord(i))
        return list_tri


def Init_ListFuncFull( ListBase_Cat, Table_Cof, List_Ord='all' ):
    
    ndim, ntri, nrad  = np.shape(Table_Cof) 
    iradi=np.arange(nrad)

    if isinstance(List_Ord,str):
        itrig=np.arange(ntri)
    elif isinstance(List_Ord,int):
        # itrig=np.array([List_Tri])
        itrig= givetri(List_Ord)
    else: # a list or np.ndarray
        # itrig=np.array(List_Tri)
        itrig = givetri(List_Ord)

    def fij(i):            
        def f(d_ij,ai,aj):
            r = 0
            for j in itrig:
                for k in iradi:
                    r += Table_Cof[i,j,k]*ListBase_Cat[i][j][k](d_ij,ai,aj)
            return r
        return f

    ll = [ [] for i in range(ndim)] 

    for i in range(ndim):
            ll[i] = fij(i)  
    
    return ll


def InitListFunc( Order, FuncRad, VectorRad, phicof ):

    _, lbasecat, lbaserad, trig_string, trig_num = polartrigo(Order,FuncRad,VectorRad)
    tabcof = TocofCat( np.shape(lbasecat), phicof)

    lffrad = Init_ListFuncRad( lbaserad, tabcof )
    lffull = Init_ListFuncFull( lbasecat, tabcof )

    return lffull, lffrad, tabcof



def InitListFuncSym( Order, FuncRad, VectorRad, phicof):

    def SinTrigo(k,kai,kaj):
        def func(ai,aj):
            return k*np.sin(kai*ai+kaj*aj)
        return func

    def CosTrigo(k,kai,kaj):
        def func(ai,aj):
            return k*np.cos(kai*ai+kaj*aj)
        return func
    
    def zerof():
        return lambda ai, aj: ai*0 + aj*0
        
    def PolyExp(k, r0):
        return lambda x: (1/factorial(k))*(x/r0)**k*np.exp(-x/r0)

    def PolyExpDiff(k, r0):
        return lambda x: (k/x-1/r0)*(1/factorial(k))*(x/r0)**k*np.exp(-x/r0)

    def PolyExpInt(k,r0):
        def func(x):
            rr = np.array([ ((-1)*factorial(k)/factorial(k-i))*x**(k-i)*np.exp(-x) for i in range(k+1) ])        
            return np.sum(rr,axis=0)
        def funcc(x):
            return (1/factorial(k))*r0*(func(6/r0) -func(x/r0)) 
        return funcc

    def createfunc(lbrad, tabcof):
        def fun(itrig):
            def cof1(r):
                rr = np.array([ tabcof[itrig,k]*lbrad[k](r) for k in range(tabcof.shape[1]) ])
                return np.sum(rr,axis=0)
            return cof1
        listfun = [ fun(i) for i in range(tabcof.shape[0]) ]
        return listfun
    
    def convcofsym(tabcof):
        tabcofsym = np.copy(tabcof)
        tabcofsym[0,:] = 0*tabcofsym[0,:] 
        tabcofsym[2,:] = -tabcofsym[1,:] 
        tabcofsym[6,:] = tabcofsym[4,:]
        return tabcofsym
    
    # def convcof2(tabcof):
    #     tabcof2 = np.copy(tabcof[2])
    #     tabcof2[]
    #     return tabcof2

    def concofvr(tabcof):
        tabcofvrpot = np.copy(tabcof)*-1    
        return tabcofvrpot

    lpoly = [ PolyExp(kk, 1) for kk in VectorRad ]
    lpolydif = [ PolyExpDiff(kk, 1) for kk in VectorRad ]
    lpolyint = [ PolyExpInt(kk, 1) for kk in VectorRad ]

    lsin = [ zerof(), SinTrigo(1,1,0), SinTrigo(1,0,1), SinTrigo(1,1,-1), SinTrigo(1,2,0), SinTrigo(1,1,1), SinTrigo(1,0,2) ]
    lcos = [ CosTrigo(1,0,0), CosTrigo(1,1,0), CosTrigo(1,0,1), CosTrigo(1,1,-1), CosTrigo(1,2,0), CosTrigo(1,1,1), CosTrigo(1,0,2) ]
    lcosdt1 = [ zerof(), SinTrigo(-1,1,0), zerof(), SinTrigo(-1,1,-1), SinTrigo(-2,2,0), SinTrigo(-1,1,1), zerof() ]
    lsinom = [ zerof(), SinTrigo(-1,1,0), zerof(), SinTrigo(-1,1,-1), SinTrigo(-2,2,0), SinTrigo(-1,1,1), zerof() ]
    lsin2 = [ zerof(), SinTrigo(1,1,0), zerof(), SinTrigo(1,1,-1), SinTrigo(1,2,0), SinTrigo(1,1,1), zerof() ]
    lupot = [ zerof(), CosTrigo(1,1,0), CosTrigo(1,0,1), CosTrigo(1,1,-1), CosTrigo(1/2,2,0), CosTrigo(1,1,1), CosTrigo(1,0,2) ]

    # def PolyExpDiff(k, r0) :
    #     if k==0 :
    #         f = lambda x: -np.exp(-x/r0)/r0
    #     else :
    #         f = lambda x: (k*(x/r0)**(k-1)/r0 - (x/r0)**k/r0)*np.exp(-x/r0)/math.factorial(k)
    #     return f

    #### main ####
    # lbaseradif =
    lbasecat, lbaserad = polartrigo(Order,FuncRad,VectorRad)[1:3]
    tabcof = TocofCat( np.shape(lbasecat), phicof)
    
    vr = createfunc(lpoly,tabcof[0])
    up = createfunc(lpolyint,tabcof[0])
    omp = createfunc(lpolyint,-tabcof[0])
    om = createfunc(lpoly,tabcof[2])
    domdvr = createfunc(lpolydif,tabcof[2])

    return vr, om, up, omp, domdvr


def InitListFuncTrigo(Order,opt):

    def Sin(kai,kaj):
        def func(ai,aj):
            return np.sin(kai*ai+kaj*aj)
        return func 
    
    def Cos(kai,kaj):
        def func(ai,aj):
            return np.cos(kai*ai+kaj*aj)
        return func
    
    def CosDev(kai,kaj):
        def func(ai,aj):
            return -kai*np.sin(kai*ai+kaj*aj)
        return func
    
    if opt == 'Sin': ftrig = Sin
    elif opt == 'Cos': ftrig = Cos
    elif opt == 'CosDev': ftrig = CosDev

    listfunctrigo = [] 
    for n in range(Order+1):
        if -n+1 < n+1:
            for k in np.arange(-n+1,n+1):
                listfunctrigo.append( ftrig(n-np.abs(k),k) )
        else:
            listfunctrigo.append( ftrig(0,0) )
         
    return listfunctrigo


def cf3d(listfun,listfuntrigo, listitrigo=[]):
    if listitrigo == []:
        listitrigo = np.arange(len(listfun))

    def fun(r,ai,aj):
        rr = np.array([ listfun[k](r)*listfuntrigo[k](ai,aj) for k in listitrigo ])
        return np.sum(rr,axis=0) 
    
    return fun




