# sfiabp.vectorial.multicore.sfi

__sfiabp.vectorial.multicore.sfi__ ( list_data, dtframe, fun1p, fun2p, xboxlim, yboxlim, lcell, ncore, verbose = False,  drift_mode = ’Stratonovich’, inverse_mode = ‘pinv’, edge_filter = True,  strato_diff_mode = 'ABP_Vestergaard’ , strato_diff_matrix = np.array([[0,0,0],[0,0,0],[0,0,1]]),  histo_mode = False, histo_vecr = np.linspace(0,lcell,2*lcell+1,endpoint=True) , histo_veca = (np.pi/180)*np.linspace(0,360,36+1,endpoint=True) )

The function implements the stochastic force inference algorithm applied to the inference of the forces and the diffusion of a set of identical brownian particles. The stochastic trajectories of N particles, regularly spaced by the time interval dtframe, are stored in list_data. The positions are constrained by the size of the box, xboxlim and yboxlim. To infer the 1,2 particle forces, the basis functions are specified respectively by fun1p, fun2p and once the process done, the function returns the corresponding inferred coefficients cof1p, cof2p. In addition, the function returns the constant diffusion matrix of shape 3x3 with the diagonal elements Dxx, Dyy and the rotational diffusion Dr. The estimation is done by the Vestergaard estimator. A main advantage of this function is the significant acceleration of the computation process powered by the cell method, for which the 2-particles force is taken into account only if the neighbor distance between the two particles is within a cell of size lcell. The pairwise interaction with the neighbors found outside the cell are therefore neglected. To increase the speed even more, the process is parallelized with the standard multiprocess library of python. The core number is provided by ncore.   

__PARAMETERS :__

__list_data : array_like__ 

List of frames saved at regularly time interval. Each frame ‘i’  has the shape Npar_i x Ncor with Npar_i the particle number and Ncor = 3 the coordinates of the particle : x_pos (um), y_pos (um) and the orientation $\theta$ (rad). Optionally, the algorithm treats also the case Ncor = 4, the fourth column being the label of the particle (list of int) which is useful for experimental data, where the particle number Npar_i changes over time. 

__dtframe : float__

Time interval between each frame.

__fun1p :  list of fun__

List of basis function for the 1-particle drift. For each function, the input is the coordinates of the particle : X = np.array([x,y,theta]); the output is the drift components : F = np.array([Fx,Fy,theta_dot]). For example, the unitary active force that moves forward the particle along its orientation is written as : Fact = lambda X : np.array([np.cos(X[2]), np.sin(X[2]),0])

__fun2p : list of fun__

List of basis function for the 2 particle drift. For each function, the input is the coordinate of the two interacted particles : Xi and Xj ; the output is the the force F_j->i acting on i from j. Several types of basis function are implemented in the sub-module sfiabp.base. Please see the tutorials for more information. 

__xboxlim, yboxlim : sequence of ints__

Boundaries of the box along the x or y axis respectively (xboxlim = np.array([lim_inf,lim_sup])). The values can be equal to the boundaries of the frame, but it is also possible to use smaller values. In this case the SFI algorithm is applied only to a  restricted portion of the frame.

__lcell :  int__

Size dimension of the cell, the number must a divisor of both the widths of the box, along the x and y directions.

__Ncore : int__

Number of core to use for the analysis. 

__verbose : bol, optional__

Print information related to the progress of the computation

__drift_mode : { 'Stratonovich', 'Ito' }, optional__

Internal method to compute the drift.

__inverse_mode : {'name':'pinv'}__

By default, the Gram matrix is inverted by the pseudo-inverse method (‘pinv’). 

__edge_filter : bol, optional__

The filter discards the particles closed to the box edge form the statistics while taking them into account as neighbors. Indeed, these particles may be influenced by neighbors located outside the box which may biased the force inference. The distance from the edge for which the filter applied is the length of lcell. The density histogram is non longer symmetric if the filter is enabled. 

__strato_diff_mode : {'Vestergaard', 'ABP_Vestergaard', 'ABP_CST'}, optional__
__strato_diff_matrix : ndarray, optional__

Additional parameters to provide in case of the 'Stratonovich' estimator. In case of ‘Vestergaard’, the full diffusion matrix is estimated by the Vestergaard estimator. In case of ‘ABP_Vestergaard’, the diffusion matrix is weighted by a boolean mask of same shape, given by strato_diff_matrix. In particular, this is useful in our study, since the the spatial diffusion can be neglected (Dxx = Dyy = 0)  but not the rotational diffusion Dr, so that the mask is np.array([[0,0,0],[0,0,0],[0,0,1]]). In the last option ‘ABP_CST’, the already known diffusion matrix is directly provided to strato_diff_matrix which accelerates the computation.

__histo_mode :  boolean, optional__
__histo_vecr : ndarray, optional__
__histo_veca : ndarray, optional__

If histo_mode enabled, compute the pair histogram in polar coordinates (r, thata_i, theta_j) in the same time of the SFI process. The radial bins (um) are delimited by the vector histo_vecr (by default : np.linspace( 0,lcell, 2*lcell+1, endpoint=True)) and the angular bins (rad) by histo_veca (by default : (np.pi/180)*np.linspace(0,360,36+1,endpoint=True) ) 

__RETURNS :__ 

__cof1p, cof2p :  array like__

Inferred coefficient related to the 1,2 particle forces, same order of appearance as for fun1p, fun2p . 

__D_average : ndarray__

Diffusion matrix of shape 3x3 computed by the Vestergaard estimator.

__psfi : dict__

Output dictionary that collects the parameters used for the SFI process and several information related to the process. The main keys are the followings : 
‘histo’ : ndarray
Pair histogram in polar coordinates. The radial bins (um) are stored in psfi[‘histo_vecr’] and the angular bins (rad) in psfi[‘histo_veca’].  
‘stat_weight’ : int
Effective number of particles computed by the SFI process 
‘pair_stat_weight’ : int 
Effective number of neighbors computed by the SFI process, must be equal to np.sum(histo) 
‘time’ : float
Elapsed time (s) to execute the function.  

# sfiabp.vectorial.simulation.sim

__sfiabp.vectorial.simulation.sim__ ( npar, nfra, dtframe, dtinc, xlim, ylim , lcell, fun1p, fun2p, fundiff, isim = 0, verbose = False,  prer = prer, frame_init = frame_init )

The function process the temporal evolution of npar brownian particles, that moves within a box of size xlim and ylim. To increase the speed of the simulation, the function use the cell list algorithm  with the cell dimension given by lcell. At time t = 0, the initial frame is set to frame_init or by default composed of uniformly distributed particles (x position (um), y position (um) and the orientation theta (rad)), then for each time increment dtinc, the simulation computes the next frame. A total of nfra number of frame spaced from dtframe are saved. The 1 particle function as well as the  pairwise interaction and the tensor diffusion are provided by the list of function : fun1p, fun2p and fundiff. Finally the trajectories and different parameters of the simulation are collected and return in a output dictionary.   

Parameters :

npar : int 
Number of particle which is constant during the whole simulation.  
nfra : int 
Number of frame to save. 
dtframe : int
Time interval (s) between each frame.
dtinc : int
Time integration (s) of the simulation with the condition dtinc < dtframe. 
xlim : int
ylim : int
The algorithm implements the periodic boundary conditions with the spatial boundaries 0 <= x < xlim, and 0 <= y < ylim along respectively the x and y component. 
lcell : int
Cell dimension which must be a divisor of both xlim and ylim.    
fun1p : list of fun
fun2p : list of fun 
List of 1 and 2 particle forces. 
fundiff : list of fun
List of tensor diffusion.

Optional Parameters :

isim : int 
A number that label the simulation and fix the seed of the random generator. 
verbose : int 
Print information related to the progress of the computation
prer : int
Number of simulation step increment before saving the first frame.

Returns :

SimData : dict
Output dictionary that collects the results of the simulation inference. 
See below the meaning of the different keys.
	‘X_Raw’ : list of ndarray
List of frames that collect the particle trajectories over time.
‘dtframe’ : List of ndarray
Time interval (s) between two consecutive frames. 
‘xframelim’ : ndarray
‘yframelim’ : ndrarray
Spatial boundaries (um) of the simulation. xframelim = np.array([0,xlim]) and yframelim = np.array([0,ylim]). 
‘P’ : dict
Parameters of the simulation.
‘time’ : float
Elapsed time (s) to execute the function.  

    1) References
1. J. Iwasawa, D. Nishigushi and M. Sano. Physical Review Research, 3:043104, 2021
2. A. Frishman and P. Ronceray Physical Review X, 10(2):021009, 2020
3. A. Poncet et al. Physical Review E, 103(1):012605, 2021
