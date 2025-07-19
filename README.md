# ﻿sfiabp_py  

The repository _sfiabp_py_ proposes to adapt the stochastic force inference (SFI) algorithm presented in [_Frishman et al. Phys. Rev. X, 10(2):021009, 2020_](url) to the analysis of 2D active polar particles. 
It includes the main python module _sfiabp_ that contains all the required functions to perform the analysis presented in XXX as well as several tutorial scripts for an easy handling of these functions. 

This document summarizes the main features of the python module _sfiabp_. Part 1 lists all the dependencies for the good operation of the module. Part 2 lists the sub-modules and the most important functions. More details on these functions are provided in DOC.Md. Part 3 briefly describes the tutorials. For example, they illustrate the inference process with different type of basis function, the data analysis with the measure of the relevant observables and the process of new simulations to validate the inference. The tutorials are accompanied with specific data sets, that are described in the last part. 

_sfiabp_py_ is distributed under the GNU Library General Public License v2.

# List of dependencies

Please check the installation of the following packages : 
- numpy
- scipy
- matplotlib
- os
- multiprocess
- dill
- pathos

# List of sub-modules

**sfiabp.vectorials** : This sub-module contains the most important functions of the program. _multicore.sfi_ to perform the SFI algorithm and _simulation.sim_, a cell list type algorithm that allows the dynamics evolution of N particles with tunable interactions. 
The other routines ensure the good working of the two functions.

**sfiabp.base** : A sub-module managing the different basis functions, useful for the inference or the simulation processes. 
The important sub-module is _base2ptrigo_ and the main functions _base2ptrigo.polartrigo_,  _base2ptrigo.InitListFunc_ to generate in a easy way the trigonometric basis functions presented in the article. 

**sfiabp.display** : A sub-module providing different routines to plot and manage efficiently the inference results. The main functions _sfidisp_sweep_ and _sfidisp_versus_ allow to plot the results in an interactive way. 

**sfiabp.general** : A sub module providing additional functions useful for the data analysis. The sub-module _observables_ contains the functions to measure the observables discussed in the article. 

# List of the tutorials

**tutorial1_trigo.py** : In this tutorial, we show a basic illustration of the SFI (stochastic force inference) method and its related function _vectorial.multicore.sfi_, applied to a synthetic data set, composed of 82 self propelled particles with a simple isotropic radial pair interaction. 
To run the function, one important argument to provide is the basis functions. In our case,  the basis for the 1 particle interaction is given simply by the unitary active velocity  $(V_x, V_y) = (cos(\theta), sin(\theta))$, with $\theta$ being the angular orientation of the particle. 
The basis for the 2 particle interaction is the set of trigonometric functions described in the article, which are provided by the function _base2ptrigo.polar_trigo_. The three parameters to enter are : (1) the type of radial functions ( by default FuncRad = ‘PolyExp’ for  Polynomial-exponential function ), (2) the span of the radial functions VectorRad indicating the center of each functions and (3) the trigonometric order n=0,1,2….
An optional argument is the choice of the internal method for processing the drift.  
The recommended and default method, used in this article, is the ‘Stratonovich’ mode which is the unbiased method, robust against diffusion and error noises. 
It reduces significantly the number of frame required, but at the same time increases, the time to process a single frame. 
However, it is possible to use the classical method, denoted here the ‘Ito’ mode. 
The method processes more rapidly a single frame but needs a higher number of frames to converge and the convergence is biased by the different source of noises. 
Note that although the SFI algo implements the ‘cell’ method to accelerate the process, the overall duration may be quite long and depends largely on several parameters such as the cell dimension, closely related to the number of particles to analyze, the number of frame to process and the number of basis functions. 
At the end, the inferred results are compared to the theoretical function with the plot function _display.sfidisp_sweep_.

**tutorial2_mesh.py** : The second tutorial proposes to analyze the same data set as in tutorial1_trigo.py but with another set of basis functions.
Sometimes, it is difficult to predict the form of the pair interaction so that it may be useful to process a first crude estimation that can guide later the choice of a more precise set of basis functions.
For this, the set of basis functions of type ‘mesh’ implemented in the library _base.base2pmesh_ can be of practical interest. 
We resume the same dataset and now use the function _base2pmesh.polarmesh_ to generate the basis functions of type mesh. 
For this we provide the grid in polar coordinates $(r, \theta_1, \theta_2)$ and the function used for each bin, here a simple ‘Step’ function. 
After the SFI process is done, we compare the inferred results with the theoretical prediction. 
Unfortunately, the method is mainly limited by the maximum number of basis functions that the computer can handle (about 1000-1500).

**tutorial3_flock.py, tutorial4_obs.py** : In these tutorials, we focus on a specific data set that emulates the flock of particles and try to reproduce the observed dynamics. 
The interactions to find are a mix between hydrodynamic interaction ( neutral squirmer ) and dipole-dipole interaction and in that respect, may look similar to the interactions found in experiments. 
The synthetic data contains 5000 frames showing the dynamics of 500 particles within a periodic box of size $200 \times 200\\ \mu m^2$.
Through this example, we illustrate different functions of the programs that can be useful for a full analysis, for example, the measurement of different observables that describe the dynamics, the inference of the forces and the comparison between the inferred trigonometric terms and the predicted ones.
At the end, we process of a new simulation with the inferred forces and compare the properties between the new simulation and the original data (autocorrelation of polarity, video analysis).
As a result, we show that the inferred simulation is very similar to the original one showing that the interactions have been correctly captured.

**script_scalarprod1d.py, script_scalrprod3d.py** : Two addtional scripts to show how to expand analytically an arbitrary function on a set of basis functions. 

# Bench mark data

We provide two synthetic data sets to run the tutorials (/tutorial/data) whose main features are described below. 
Both data are saved in a python dictionary as .pkl file. 
A format that can be managed with the standard module ‘dill’.
The different dictionary keys (dict.keys()) are the following : 
-  ’X_Raw’ : The data to infer. The format must be a list of ndarray of dimension Nfra x N x Dim,  with Nfra the frame number, N the particle number, Dim the dimension of the coordinates equal to Dim=3 $(x, y, \theta)$. In case of experimental data set, the dimension can be Dim=4 , the fourth column being the label (int) of the particle given by the tracking method. If Dim=4, the particle number N can vary over time. 
- ‘dtframe’ : the time interval between two frames.
 - ‘xframlim’ and ‘yframelim’ : the size of the box.
- ‘lcell’ : cell dimension used in case of data provided by a simulation.

**Sim_1r4_npar_82_u_6_k_2000_5000f.pkl** : Simulated data presenting the dynamic of 82 self propelled particles interacting with a simple isotropic radial pair interaction of form $F(r) = 2000/r^4$. Active velocity  U = 6 um/s, rotational diffusion Dr  = 0.1  /s^2 and frame number Nfra = 5000 ( ‘dtframe’ = 0.1 s, ‘xframelim’ = ‘yframelim’ = [0,320] um, lcell = 20 um).

**Ominimal_model3_small_5000f.pkl** : Simulated data set presenting the dynamic of 500 flocking particles, interacting with a complex pair interaction. The analytical form is given in the notebook (tutorial3).  Active velocity U = 6 um/s, rotational diffusion Dr  = 0.1 /s^2 and frame number Nfra = 5000  ( ‘dtframe’ = 0.1 s, ‘xframelim’ = ‘yframelim’ = [0,200] um, lcell = 20 um ).

**20180927ordered_df_pv_m.pkl** : In addition we provide the first 100 frames of the experimental data set A ( [_Iwasawa et al. Phys. Rev. Research, 3:043104, 2021_](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.043104) ) discussed in the article. Active velocity U = 9.5 um/s, rotational diffusion Dr = 0.08 rad/s^2,  (‘dtframe’ = 1/15 s,  ‘xframelim’ = [0,420] um,  ‘yframelim’ = [0,336] um, recommended cell dimension ‘lcell’ = 21 um) 
