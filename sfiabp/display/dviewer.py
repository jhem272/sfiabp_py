
from matplotlib.widgets import Slider
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as lines
import ipywidgets as widgets

import numpy as np
import dill
import os
from sfiabp.base  import base2ptrigo
from sfiabp.base  import base2pmesh


"""--------------------------------------------------------------------
                interactive display of sfi results 
--------------------------------------------------------------------"""


def sfidisp_dviewer(Path, PathRef, **kwargs):

    #### function events ####

    # # mouse click
    # def handleEvent (event):
    #     global thetai, thetaj, idic
        
    #     if idic == 0:
    #         if event.inaxes in fig.axes[3:7]:
    #             thetai = event.xdata
    #             thetaj = event.ydata
    #             refresh_figure()
    #     elif idic == 1:
    #         if event.inaxes in fig.axes[0:9]:
    #             thetai = event.xdata
    #             thetaj = event.ydata
    #             refresh_figure()
    #     elif idic == 2: # mesh lff theory
    #         if event.inaxes in fig.axes[0:9]:
    #             thetai = event.xdata
    #             thetaj = event.ydata
    #             refresh_figure()

    #     print(event)

    # key press function
    def on_press(event):
        global iRfile, nRfile
        print(event.key)
        if event.key in ['a','e']:
            if event.key == 'a': 
                iRfile = (iRfile-1)%nRfile
            elif event.key == 'e': 
                iRfile = (iRfile+1)%nRfile
            refresh_figure()

    def refresh_figure():
        fig.clf()
        create_figure(fig,list_Rabp[iRfile],RabpRef)
        
    #### main ####

    # get R data
    list_Rabp = interpreter(Path)
    # get R data ref
    with open( PathRef, 'rb' ) as inp:    
        RabpRef = dill.load(inp) 
        RabpRef['filename'] = os.path.basename(PathRef)
        
    # #  optional parameters
    # d = kwargs.get('d', 0) # particle diameter
    # dijinc = kwargs.get('dijinc', 0.3) # radial increment for key press event
    # tishift = kwargs.get('tishift', 0) # radial increment for key press event
    # tjshift = kwargs.get('tjshift', 0) # radial increment for key press event 
    # lff = kwargs.get('exact_fun', [])
    # mask_thres = kwargs.get('mask_thres', 0)

    # # global parameters
    global iRfile, nRfile

    iRfile = 0 # index R file
    nRfile = len(list_Rabp)
        
    # Mode_View = ['spl','norm']
    # idic = 0
    # nidic = 2
    # if lff != [] : 
    #     Mode_View.append('vs')
    #     nidic = 3

    # Sabp = list_Sabp[iSfile0][iSfile1] # init S file
    # vecr = Sabp['psfi']['histo_vecr']
    # rbound = [vecr[0],vecr[-1]]  # boundaries r (um)

    fig = plt.figure(figsize=(10,4),constrained_layout=True)
    create_figure(fig,list_Rabp[iRfile],RabpRef)

    # fig.canvas.mpl_connect("button_press_event",handleEvent)
    fig.canvas.mpl_connect("key_press_event",on_press)
    
    return fig
    

###########################
## create main figure ##
###########################


def create_figure(fig,Rabp,RabpRef):

    global iRfile, nRfile

    # Sabp = list_Sabp[iSfile0][iSfile1] # init S file

    ax0 = fig.add_subplot( 1, 2, 1)
    ax1 = fig.add_subplot( 1, 2, 2)

    # pcorel 
    plot_pair_corel ( fig, ax0, RabpRef['CorelCpt'], typeplot = 'log')
    plot_pair_corel ( fig, ax1, Rabp['CorelCpt'], typeplot = 'log')

    ftz = 9
    ax0.set_title('%s'%(RabpRef['filename']),fontsize=ftz)
    ax1.set_title('%s'%(Rabp['filename']),fontsize=ftz)

    # npt_angle = 32+1
    # vi = np.linspace(0, 2*np.pi, num=npt_angle, endpoint=True) + tishift
    # vj = np.linspace(0, 2*np.pi, num=npt_angle, endpoint=True) + tjshift
    # tig_eval, tjg_eval = np.meshgrid(vi[:-1], vj[:-1], indexing = 'ij')
    # tig, tjg = np.meshgrid(vi, vj, indexing = 'ij')
    # vecr = Sabp['psfi']['histo_vecr']
    # veca = Sabp['psfi']['histo_veca']
    # dvecr = vecr[1]-vecr[0]
    # dveca = veca[1]-veca[0]

    # # fig.text( 0.5, 0.5, "iSfile0 %d, iSfile1 %d"%(iSfile0, iSfile1), fontsize=10, linespacing=1.5)
    
    # #### structure ####

    # gs = fig.add_gridspec(3,3,width_ratios=[0.3,0.9,0.8],height_ratios=[0.15,1,0.03])
    
    # gs10 = gs[1,0].subgridspec(3,1)
    # for i in range(3):
    #     fig.add_subplot(gs10[i])
    # gs11 = gs[1,1].subgridspec(2,2)
    # for i in range(2):
    #     for j in range(2):
    #         fig.add_subplot(gs11[i,j])
    # gs12 = gs[1,2].subgridspec(3,2)
    # for i in range(3):
    #     for j in range(2):
    #         fig.add_subplot(gs12[i,j])

    # # format_axes(fig)

    # ftz_mesh_title = 10 # fontsize mesh title
    # ftz_title = 9 # fontsize title
    # ftz_tick = 9
    # ftz_label = 10
    # ftz_txt = 9.3 # fontsize text

    # #### plot ####

    # ## gs11 / Mesh
    # # histogram
    # # index histo
    # ir = np.where(dij-vecr >= 0, dij-vecr, np.inf).argmin()
    # histo = Sabp['psfi']['histo'][0][ir]
    # # tig_histo, tjg_histo = np.meshgrid(veca[:-1], veca[:-1], indexing = 'ij')
    # tig_histo, tjg_histo, histo = reshape_histo(veca,veca,vi,vj,histo)
    # pcm0 = fig.axes[3].pcolormesh( tig_histo, tjg_histo, histo,
    #                                 cmap='viridis',norm=colors.LogNorm(vmin=1,vmax=1e4,clip=True))  
    # fig.colorbar(pcm0)
    # fig.axes[3].set_title('histogram $d\\theta$=%.1f°, dr=%.1f'%(dveca*180/np.pi,dvecr),fontsize=ftz_title)
    # fig.axes[3].set_xlabel('$\\theta_i$')
    # fig.axes[3].set_ylabel('$\\theta_j$')
    # fig.axes[3].set_aspect('equal')
    
    # # mesh velocities vr, vtheta, omega 
    # list_titles = ["$v_r$","$v_\\theta$","$\\omega$"]
    # list_vmx = [ [-2,2], [-1,1], [-1,1] ]
    # for i,axi in enumerate( [fig.axes[ix] for ix in [4,5,6]] ):
    #     pcm0 = axi.pcolormesh( tig, tjg, Sabp['lff'][i](dij, tig_eval, tjg_eval), cmap='RdBu_r', vmin=list_vmx[i][0], vmax=list_vmx[i][1] )
    #     fig.colorbar(pcm0)
    #     axi.set_title(list_titles[i],fontsize=ftz_mesh_title)        
    #     axi.set_xlabel('$\\theta_i$')
    #     axi.set_ylabel('$\\theta_j$')
    #     axi.set_aspect('equal')
    
    # ## gs12 / function with r
    # r = np.linspace(rbound[0],rbound[1],100)
    # list_ylim_rad = [ (-10,10), (-1,1), (-1,1) ]
    # list_ylabel = ['$v_r$ $(um.s^{-1})$','$v_\\theta$ $(um.s^{-1})$',"$\\omega$ $(s^{-1})$"] 
    # vr,h1,h1n=histo1d(vecr,Sabp['psfi']['histo'][0])
    # kfach1d = np.abs( np.array([ list_ylim_rad[i][0]*0.5 for i in range(3) ]) )

    # for i,axi in enumerate( [fig.axes[ix] for ix in [7,9,11]] ):
        
    #     axi.plot(r,Sabp['lff'][i](r, thetai, thetaj),label='sfi')
    #     if lff != []:
    #         axi.plot(r,lff[i](r,thetai,thetaj),'-',label='exact')

    #     axi.set_ylabel(list_ylabel[i],fontsize=ftz_label)
    #     axi.set_ylim(list_ylim_rad[i])
    #     axi.set_xlim(rbound[0],rbound[1])
    #     # vertical line, current distance r
    #     axi.axvline(dij,ls='--',c='red',lw=1)
    #     # histogram 1d
    #     axi.plot(vr,h1n*kfach1d[i]+list_ylim_rad[i][0],ls='-',label='pair cor.',c='0.8')
    #     # first shaded area
    #     axi.axvspan(0,d,alpha=0.3,color='grey')
    #     # second shaded area (collision effect)
    #     z = 2*Sabp['active_vel']*Sabp['psfi']['dtframe']
    #     axi.axvspan(d,d+z,alpha=0.1,color='grey')
    #     # fig.axes[9].axvspan(d,d+z,alpha=0.1,color='grey')
    #     axi.tick_params(axis='both',which='major',labelsize=ftz_tick)

    # # fig.axes[7].legend( ncol=1, loc='upper right', frameon=False, fontsize=ftz_gs12 )
    # fig.axes[7].legend( ncol=3, bbox_to_anchor=[0,1], loc='lower left', frameon=False, fontsize=ftz_tick )
    # fig.axes[11].set_xlabel("$r$ $(um)$",fontsize=ftz_label)
        
    # fig.axes[7].text( 0.69, 0.72, "$r$ = %.1f $um$ "%(dij) + "\n" + "$\\theta_1$ = %.2f $rad$ "%(thetai) + "\n" + \
    #                                 "$\\theta_2$ = %.2f $rad$"%(thetaj), fontsize=8.5, transform = fig.axes[7].transAxes)
    
    # ## gs12 / function with theta
    # vtheta = np.linspace(0,2*np.pi,100)
    # list_ylim_theta = [ (-10,10), (-1,1), (-1,1) ]

    # for i,axi in enumerate( [fig.axes[ix] for ix in [8,10,12]] ):

    #     axi.plot(vtheta,Sabp['lff'][i](dij, thetai, vtheta),label="$\\theta_1$ = %.2f, sfi"%(thetai),ls='-',color='tab:blue' )
    #     axi.plot(vtheta,Sabp['lff'][i](dij, vtheta, thetaj),label="$\\theta_2$ = %.2f, sfi"%(thetaj),ls='-',color='tab:orange')

    #     if lff != []:
    #         axi.plot(vtheta,lff[i](dij, thetai, vtheta),label="exact",ls='--',color='tab:blue')
    #         axi.plot(vtheta,lff[i](dij, vtheta, thetaj),label="exact",ls='--',color='tab:orange')

    #     axi.set_xlim(vtheta[0],vtheta[-1])    
    #     axi.set_ylim(list_ylim_theta[i])
    #     axi.set_ylabel(list_ylabel[i],fontsize=ftz_label)
    #     axi.tick_params(axis='both',which='major',labelsize=ftz_tick)
    
    # # fig.axes[8].legend( ncol=1, loc='upper right', frameon=False, fontsize=ftz_gs12 )
    # fig.axes[8].legend( ncol=2, bbox_to_anchor=[0,1], loc='lower left', frameon=False, fontsize=ftz_tick )
    # fig.axes[12].set_xlabel("$\\theta$ $(rad)$",fontsize=ftz_label)

    # ## gs10 / artist
    # # rplt = (dij-d)/(Sabp['psfi']['lcell']-d)*2+1 # dr = 4
    # rplt = 2
    # p,pnext = pol2cart_janus(Sabp['lff'],dij,thetai,thetaj,dt=1,rplt=rplt)
    # list_janus = []
    # list_janus.extend(particle(np.array([0,0,0]),r=0.5,color='black'))
    # list_janus.extend(particle(p,r=0.5,color='royalblue'))
    # list_janus.extend(particle(pnext,r=0.5,color='lightsteelblue'))
    # listcc = [2,3,4]
    
    # ax = fig.axes[1]
    # for ri in listcc:
    #     ax.add_patch(mpatches.Circle((0,0),ri,ls='--',color='0.5',fill=False))
    # for i,pp in enumerate(list_janus): 
    #     ax.add_patch(pp)

    # ax.set_xlim(-5,5)    
    # ax.set_ylim(-5,5)
    # ax.set_aspect('equal')
    # ax.axis('off')
                       
    # fig.text(0.01,0.45,("$\\theta_i$=%.2f"%(thetai)+"\n"
    #             "$\\theta_j$=%.2f"%(thetaj)+"\n"
    #             "$v_r$=%.2f"%(Sabp['lff'][0](dij,thetai,thetaj))+"\n"
    #             "$v_\\theta$=%.2f"%(Sabp['lff'][1](dij,thetai,thetaj))+"\n"
    #             "$\\omega$=%.2f"%(Sabp['lff'][2](dij,thetai,thetaj))),fontsize=9,linespacing=1.5)

    # ## gs10 / radial basis function
    # if Sabp['basis_name'] == 'Trigo':
        
    #     lbase_rad = base2ptrigo.radtrigo( Sabp['Order'], Sabp['FuncRad'], Sabp['VectorRad'] )[2]
    #     for f in lbase_rad:
    #         y = f(r)
    #         y = y/np.max(np.abs(y))
    #         fig.axes[2].plot(r,y)
    #     fig.axes[2].set_title('normalize radial basis function',fontsize=ftz_title)
    #     fig.axes[2].set_xlabel("$r$ $(um)$",fontsize=ftz_label)
    #     fig.axes[2].tick_params(axis='both',which='major',labelsize=ftz_tick)
    #     fig.axes[2].set_xlim(r[0],r[-1])
    #     fig.axes[2].set_ylim(0,1)
    
    # elif Sabp['basis_name'] == 'Mesh':

    #     lbase_rad = base2pmesh.polarmesh( Sabp['vecr'], Sabp['veca'], Sabp['veca'], 'Step' )[2]
    #     for f in lbase_rad:
    #         y = f(r)
    #         y = y/np.max(np.abs(y))
    #         fig.axes[2].plot(r,y)
    #     fig.axes[2].set_title('normalize radial basis function',fontsize=ftz_title)
    #     fig.axes[2].set_xlabel("$r$ $(um)$",fontsize=ftz_label)
    #     fig.axes[2].tick_params(axis='both',which='major',labelsize=ftz_tick)
    #     fig.axes[2].set_xlim(r[0],r[-1])
    #     fig.axes[2].set_ylim(0,1)

    # ## gs10 / phicof 
    # # gs12, ax7, phi cof vr
    # fig.axes[0].plot(np.abs(Sabp['cof2p'][0][0:50].flatten()),'o',markersize=4,markerfacecolor='none')
    # fig.axes[0].set_yscale('log')
    # fig.axes[0].set_ylim(1e-1,1e4)
    # # fig.axes[0].set_xlabel('num',fontsize=ftz_label)
    # fig.axes[0].tick_params(axis='both',which='major',labelsize=ftz_tick)
    # fig.axes[0].set_title('|50 first coef in cof2p|',fontsize=ftz_title)  

    # ## add txt
    # create_txt(fig,Sabp,ftz_txt)


###########################
## additional functions ##
###########################


def plot_pair_corel ( fig, axi, hcpt, typeplot = 'lin'):

    dx = np.diff(hcpt['vecxy'])[0]; dy = np.diff(hcpt['vecxy'])[0]
    Lx = hcpt['vecxy'][-1]-hcpt['vecxy'][0]; Ly = hcpt['vecxy'][-1]-hcpt['vecxy'][0];  
    xmesh = hcpt['vecm'][0][:,:,0]; ymesh = hcpt['vecm'][1][:,:,0]
    histo = hcpt['Corel'][0][:,:,0]
    pcorel = histo / (np.sum(histo)*(dx*dy)/(Lx*Ly)) -1
         
    if typeplot == 'log':
        pcm1 = axi.pcolor(xmesh,ymesh,pcorel,cmap='RdBu_r',norm=colors.SymLogNorm(linthresh=0.01,linscale=0.3,vmin=-1,vmax=1,base=10))
        fig.colorbar(pcm1)
    elif typeplot == 'lin':
        pcm1 = axi.pcolor( xmesh, ymesh, pcorel, cmap='RdBu_r', vmin=-1, vmax=1 )
        fig.colorbar(pcm1)
        
    axi.set_xlabel('$x$ $(um)$')
    axi.set_ylabel('$y$ $(um)$')

 
def interpreter(Obj):

    # Obj must be a path to a folder     
    # os.path.isdir(Obj):
    PathFolder = Obj
    # collect all the file in Paath
    ls = []
    for x in os.listdir(PathFolder):
        ls.append(x)
    list_fname = [PathFolder + '/' + ls[i] for i in range(len(ls)) ]
    list_fname.sort()
    list_Rabp = [ [] for i in range(len(list_fname)) ]
    for i, fnamei in enumerate(list_fname):
        with open( fnamei, 'rb' ) as inp:    
            list_Rabp[i] = dill.load(inp) 
        list_Rabp[i]['filename'] = os.path.basename(fnamei)
            
    return list_Rabp
    
