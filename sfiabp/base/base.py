from numpy import floor_divide, mod, sqrt
from numpy import cos, sin, sqrt, pi
import numpy as np


# 1 particle force 

# constant force x
def cstx(k = 1): # constant force
    return lambda X :  np.array([ k*1, 0, 0 ]) 
# constant force y
def csty(k = 1):
    return lambda X :  np.array([ 0, k*1, 0 ])
# active force
def active(U = 1):
    return lambda X : np.array([ U * np.cos(X[2]), U * np.sin(X[2]), 0 ])
# active force, perpendicular
def active_perp(U = 1):
    return lambda X : np.array([ U * np.sin(X[2]), -U * np.cos(X[2]),0])

def stdfun1p():
    return lambda X : np.array([ [1,0,0], [0,1,0], [np.cos(X[2]),np.sin(X[2]),0], [np.sin(X[2]),-np.cos(X[2]),0] ])


# Diffusion
def Diffu(D, Dr):
    def Func(X,ddt):
        matD = np.array(([[sqrt(2*D),0,0],[0,sqrt(2*D),0],[0,0,sqrt(2*Dr)]]))
        vdif = np.concatenate((np.random.normal(0,1,2),np.random.normal(0,1,1)))
        vecD = sqrt(ddt)*np.matmul(matD,vdif) 
        return vecD
    return Func


# soft sphere interaction
def FijSSP(d,eps):

    ssp = lambda r: (eps/d)*(1-r/d) 

    def Func(Xij, Xi, Xj, dij):

        # Final vector 
        vecF = np.zeros(3)
        # angle (rad) of e_ij
        Ar = np.mod(np.arctan2(Xij[1],Xij[0]),2*np.pi)
        # unit radial vector e_ij
        vRad = np.array([np.cos(Ar),np.sin(Ar),0])

        if dij < d:
            vecF += ssp(dij)*vRad 
        
        return vecF
    
    return Func


# radial interaction of form k/r**p 
def FijRad1rN(p,k):

    fr = lambda r: k/r**p
        
    def Func(Xij, Xi, Xj, dij):

        # Final vector 
        vecF = np.zeros(3)
        # angle (rad) of e_ij
        Ar = np.mod(np.arctan2(Xij[1],Xij[0]),2*np.pi)
        # unit radial vector e_ij
        vRad = np.array([np.cos(Ar),np.sin(Ar),0])

        vecF += fr(dij)*vRad 
        
        return vecF
    
    return Func


def convcart( flatbase ):
    
    nbdim = len(flatbase[0])

    def pair(Xi,Xj):
        
        # calcul the force at ith position
        tab = np.zeros((3*nbdim,3)) # 3 cof        
        # Xi-Xj
        Xij = Xi[:2]-Xj[:2]
        # distance ij
        d_ij = np.linalg.norm(Xij)

        # escape sequence (error detection to be moved)
        if np.all(Xi == Xj): 
            return tab
        # if Xj[3] not in Particle_Type:  
        #     return tab

        # angle (rad) of e_ij
        Ar = np.mod(np.arctan2(Xij[1],Xij[0]),2*np.pi)
        # angle (rad) of ith, jth particle
        Ai, Aj = Xi[2], Xj[2] 
        # angle (rad) of ith, jth particle
        ai, aj = np.mod(Ai-Ar,2*np.pi), np.mod(Aj-Ar,2*np.pi)

        # unit radial vector e_ij
        vRad = np.array([cos(Ar),sin(Ar),0])
        # unit orthoradial vector e_ij
        vOrtho = np.array([-sin(Ar),cos(Ar),0])
        # unit torque
        vTorque = np.array([0,0,1])
        
        i=0
        while(i<nbdim):
            tab[i,:] = flatbase[0][i](d_ij,ai,aj)*vRad # er 
            tab[i+nbdim,:] = flatbase[1][i](d_ij,ai,aj)*vOrtho # etheta 
            tab[i+2*nbdim,:] = flatbase[2][i](d_ij,ai,aj)*vTorque # torque 
            i+=1
            
        return np.nan_to_num(tab)
    
    return pair


