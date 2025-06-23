# import matplotlib.pyplot as plt
# from matplotlib.patches import Circle

from numpy import sqrt, arctan2, cos, sin, abs
import numpy as np
import copy
# import matplotlib.pyplot as plt 
import dill


def stepr(rmin,rmax):
        return lambda r: 1*(rmin<=r)*(r<rmax)
    
def stepa(aimin,aimax,ajmin,ajmax):
    return lambda ai,aj: 1*(aimin<=ai%(2*np.pi))*(ai%(2*np.pi)<aimax)*(ajmin<=aj%(2*np.pi))*(aj%(2*np.pi)<ajmax)

def stepmesh(rmin,rmax,aimin,aimax,ajmin,ajmax):
    return lambda r,ai,aj : stepr(rmin,rmax)(r)*stepa(aimin,aimax,ajmin,ajmax)(ai,aj)


## main functions 


def flatindex(ir,iai,iaj,nr,nai,naj):
    ind = ir*nai*naj + iai*naj + iaj 
    return int(ind)


def flatindexinv(ind,nr,nai,naj):
    ir = np.floor_divide(ind,nai*naj)
    iai = np.floor_divide( ind-ir*nai*naj, naj )
    iaj = ind-ir*nai*naj-iai*naj
    return int(ir),int(iai),int(iaj)


def polarmesh( vecr, vecai, vecaj, FuncKern ):
    
    nr = len(vecr[:-1])
    nai = len(vecai[:-1])
    naj = len(vecaj[:-1])

    # build cat base 
    lcat1 = [[[ [] for iaj in range(naj) ] for iai in range(nai) ] for ir in range(nr) ] 
    
    # problem for loop with lambda functions
    for ir in range(nr):
        for iai in range(nai):
            for iaj in range(naj):
                if FuncKern == 'Step':
                    lcat1[ir][iai][iaj] = stepmesh(vecr[ir],vecr[ir+1],vecai[iai],vecai[iai+1],vecaj[iaj],vecaj[iaj+1])
                
    # build flat base
    lflat1 = [ [] for i in range(nr*nai*naj) ] 
    
    for ir in range(nr):
        for iai in range(nai):
            for iaj in range(naj):
                ind = flatindex(ir,iai,iaj,nr,nai,naj)
                lflat1[ind] = lcat1[ir][iai][iaj]
    
    # build list of radial base 
    if FuncKern == 'Step':
        lbaserad = [ stepr(vecr[ir],vecr[ir+1]) for ir in range(nr) ] 

    lbasecat = [ lcat1 for i in range(3) ]
    lbaseflat = [ lflat1 for i in range(3) ]

    # r = np.linspace(0,10,500)
    # plt.figure(); plt.plot(r,stepr(0,1)(r))

    return lbaseflat, lbasecat, lbaserad


def TocofCat( vecr, vecai, vecaj, array1d ):
    nr = len(vecr[:-1])
    nai = len(vecai[:-1])
    naj = len(vecaj[:-1])
    mesh1d = np.reshape( array1d,(3,nr,nai,naj))
    return mesh1d 


def Init_ListFuncFull( mesh1d, lcat ):
    
    ndim, nr ,nai, naj = np.shape(lcat)
    
    def fun(i):
        def f(r,ai,aj):
            val = 0
            for ir in range(nr):
                for iai in range(nai):
                    for iaj in range(naj):
                        val += mesh1d[i][ir,iai,iaj]*lcat[i][ir][iai][iaj](r,ai,aj)
            return val
        return f 

    ll = [ [] for i in range(ndim)] 
    for i in range(ndim):
            ll[i] = fun(i)  

    return ll


def FijMesh( FileName ):

    # get info
    with open( FileName, 'rb' ) as inp:    
        data = dill.load(inp)
    tableco = data['tableco']
    vecr = data['vecr']
    veca = data['veca'] 

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

        if vecr[0] <= d_ij < vecr[-1]:
            # get grid coordinate 
            ir = np.where(d_ij-vecr >= 0, d_ij-vecr, np.inf).argmin()
            iai = np.where(ai-veca >= 0, ai-veca, np.inf).argmin()
            iaj = np.where(aj-veca >= 0, aj-veca, np.inf).argmin()
            vecF += tableco[0][ir,iai,iaj]*vRad 
            vecF += tableco[1][ir,iai,iaj]*vOrtho 
            vecF += tableco[2][ir,iai,iaj]*vTorque 

        return vecF   
    
    return Func










                    






