
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import os
import dill

import numpy as np
from sfiabp.base  import base2ptrigo
from sfiabp.base  import base2pmesh
from sfiabp.display import library

"""--------------------------------------------------------------------
                interactive display for Sabp
--------------------------------------------------------------------"""


def sfidisp_versus(PathFolder1, PathFolder2, **kwargs):

    #### function events ####

    class KeyHandler:
        def __init__(self,list_Sabp1,list_Sabp2,DicOpt):

            self.glob = dict(
                iS1file0 = 0,
                iS1file1 = 0,
                nS1file = [ len(i) for i in list_Sabp1 ], 
                iS2file0 = 0,
                iS2file1 = 0,
                nS2file = [ len(i) for i in list_Sabp2 ],
                dij = 5, # current position (um)
                rbound = [0,10],
                thetai = 0,
                thetaj = 0, # current angle (rad)
                dijinc = DicOpt['dijinc']
                )

        # mouse click
        def handleEvent (self,event):
            print(event)
            if event.inaxes in fig.axes[0:6]:
                print(event)
                self.glob['thetai'] = event.xdata
                self.glob['thetaj'] = event.ydata
                self.refresh_figure()

        # key press function
        def on_press(self,event):
            
            glob = self.glob
            print(event.key)

            if event.key in ['a','e']:
                if event.key == 'a': 
                    glob['dij'] = (glob['dij']-glob['dijinc'])%glob['rbound'][1]
                elif event.key == 'e': 
                    glob['dij'] = (glob['dij']+glob['dijinc'])%glob['rbound'][1]
                self.refresh_figure()
            
            elif event.key in ['l','ù','p','m']:
                if event.key == 'l':
                    glob['iS1file1'] = (glob['iS1file1']-1)%glob['nS1file'][glob['iS1file0']]
                elif event.key == 'ù':
                    glob['iS1file1'] = (glob['iS1file1']+1)%glob['nS1file'][glob['iS1file0']]
                elif event.key == 'p':
                    glob['iS1file0'] = (glob['iS1file0']+1)%len(glob['nS1file'])
                    glob['iS1file1'] = glob['iS1file1']%glob['nS1file'][glob['iS1file0']]
                elif event.key == 'm':
                    glob['iS1file0'] = (glob['iS1file0']-1)%len(glob['nS1file'])
                    glob['iS1file1'] = glob['iS1file1']%glob['nS1file'][glob['iS1file0']]
                self.refresh_figure()

            elif event.key in ['4','6','8','5']:
                if event.key == '4':
                    glob['iS2file1'] = (glob['iS2file1']-1)%glob['nS2file'][glob['iS2file0']]
                elif event.key == '6':
                    glob['iS2file1'] = (glob['iS2file1']+1)%glob['nS2file'][glob['iS2file0']]
                elif event.key == '8':
                    glob['iS2file0'] = (glob['iS2file0']+1)%len(glob['nS2file'])
                    glob['iS2file1'] = glob['iS2file1']%glob['nS2file'][glob['iS2file0']]
                elif event.key == '5':
                    glob['iS2file0'] = (glob['iS2file0']-1)%len(glob['nS2file'])
                    glob['iS2file1'] = glob['iS2file1']%glob['nS2file'][glob['iS2file0']]
                self.refresh_figure()

        def refresh_figure(self):
            fig.clf()
            create_figure(fig,list_Sabp1,list_Sabp2,self.glob,DicOpt)

    #### main ####

    ## additional parameters
    DicOpt = dict(d = kwargs.get('d', 0),
              dijinc = kwargs.get('dijinc', 0.3),          
              tishift = kwargs.get('tishift', 0),         
              tjshift = kwargs.get('tjshift', 0),
              rlim = kwargs.get('rlim', [0,10]),
              mask_thres = kwargs.get('mask_thres', 0), 
              Prefix = kwargs.get('Prefix', '') )
    
    if -np.pi>DicOpt['tishift'] or np.pi<DicOpt['tishift'] or -np.pi>DicOpt['tjshift'] or np.pi<DicOpt['tjshift']:
        raise KeyError('tishift, tjshift must be in the range -pi<=x<=pi')
    
    ## get data
    list_Sabp1 = library.interpreter(PathFolder1,DicOpt['Prefix'])
    list_Sabp2 = library.interpreter(PathFolder2,DicOpt['Prefix'])
    library.implementLff(list_Sabp1,DicOpt)
    library.implementLff(list_Sabp2,DicOpt)

    # global index
    keyhand = KeyHandler(list_Sabp1,list_Sabp2,DicOpt)
    # intialization
    fig = plt.figure(figsize=(18,8),constrained_layout=True)
    # create figure
    create_figure(fig,list_Sabp1,list_Sabp2,keyhand.glob,DicOpt)

    fig.canvas.mpl_connect("button_press_event",keyhand.handleEvent)
    fig.canvas.mpl_connect("key_press_event",keyhand.on_press)
    
    print('ok')


###########################
## create main figure ##
###########################


def create_figure(fig,list_Sabp1,list_Sabp2,glob,DicOpt):

    # dij = glob['dij']   
    # thetai = glob['thetai']
    # thetaj = glob['thetaj']
    # lcell = np.min([Sabp1['psfi']['lcell'],Sabp2['psfi']['lcell']])
    # glob['rbound'] = [0,lcell]
    # r = np.linspace(glob['rbound'][0],glob['rbound'][1],100)

    ## cursors
    thetai = glob['thetai']
    thetaj = glob['thetaj']
    dij = glob['dij']

    Sabp1 = list_Sabp1[glob['iS1file0']][glob['iS1file1']]
    Sabp2 = list_Sabp2[glob['iS2file0']][glob['iS2file1']]

    #### structure ####

    gs = fig.add_gridspec(3,2,height_ratios=[0.15,1,0.05],width_ratios=[1,0.6])
    gs10 = gs[1,0].subgridspec(2,3)
    for i in range(2):
        for j in range(3):
            fig.add_subplot(gs10[i,j])
    gs11 = gs[1,1].subgridspec(3,2)
    for i in range(3):
        for j in range(2):
            fig.add_subplot(gs11[i,j])

    # format_axes(fig)

    ## gs10 / plot mesh 

    library.plotmesh_versus(fig,fig.axes[0:3],Sabp1,dij,'S1',DicOpt)
    library.plotmesh_versus(fig,fig.axes[3:6],Sabp2,dij,'S2',DicOpt)

    ## gs12 / fun angle
    list_axes = [fig.axes[6],fig.axes[8],fig.axes[10]]
    library.plotRad(fig,list_axes,dij,thetai,thetaj,[Sabp1['lff'],Sabp2['lff']],['S1','S2'],DicOpt)
    # ## gs12 / fun angle
    list_axes = [fig.axes[7],fig.axes[9],fig.axes[11]]
    library.plotAng(fig,list_axes,dij,thetai,thetaj,[Sabp1['lff'],Sabp2['lff']],['S1','S2'],DicOpt)
    
    ## text
    ftz_txt = 9.5 # fontsize text
    fig.text(0.01,0.89,("$\\bf{sfidisp\\_versus.py}$\n" 
                         "- press a, e to change the distance r\n"
                         "- click on the 2D mesh to change $\\theta_1$, $\\theta_2$\n"
                         "- press p,m for dim0 and l,ù for dim1 to change list_S1[dim0][dim1]\n"
                         "- press 8,5 for dim0 and 4,6 for dim1 to change list_S2[dim0][dim1]\n"
                         "- press q to quit"), fontsize=ftz_txt-1.3, linespacing=1.2)

    txtidentity( fig, 0.23, 0.896, 1, Sabp1, glob['iS1file0'], glob['iS1file1'], ftz_txt )
    txtidentity( fig, 0.62, 0.896, 2, Sabp2, glob['iS2file0'], glob['iS2file1'], ftz_txt )

    
###########################
## additional functions ##
###########################


def txtidentity( fig, x, y, id, Sabp, iSfile0, iSfile1, ftz_text ):
        
        psfi = Sabp['psfi']
        if psfi['inverse_mode']['name'] == 'pinv':
            fig.text( x, y, ("$\\bf{List\\ Sabp%d}$[%d][%d] : \n%s\n"%(id,iSfile0,iSfile1,Sabp['filename'])+
    "frame number : %d, dtframe (s): %.2f, xbox : [%.1f,%.1f], ybox : [%.1f,%.1f], cell length : %d\n"%\
    (psfi['frame_weight'],Sabp['psfi']['dtframe'],Sabp['psfi']['xboxlim'][0],Sabp['psfi']['xboxlim'][1],Sabp['psfi']['yboxlim'][0],Sabp['psfi']['yboxlim'][1],Sabp['psfi']['lcell']) +                                                                
     "drift mode: %s, B mode: %s, inverse mode : %s, basis_name : %s"%(psfi['drift_mode'],psfi['B_mode'],psfi['inverse_mode']['name'],Sabp['basis_name'])),
                                                    fontsize=ftz_text, linespacing=1.75 )
             
        elif psfi['inverse_mode']['name'] == 'tiko':                 
            fig.text( x, y, ("$\\bf{List_Sabp%d}$[%d][%d] : \n%s\n"%(id,iSfile0,iSfile1,Sabp['filename'])+
    "frame number : %d, dtframe (s): %.2f, xbox : [%.1f,%.1f], ybox : [%.1f,%.1f], cell length : %d\n"%\
    (psfi['frame_weight'],Sabp['psfi']['dtframe'],Sabp['psfi']['xboxlim'][0],Sabp['psfi']['xboxlim'][1],Sabp['psfi']['yboxlim'][0],Sabp['psfi']['yboxlim'][1],Sabp['psfi']['lcell']) +                                                                
    "drift mode: %s, B mode: %s, inverse mode : %s, alpha : %.3e, basis_name : %s"%(psfi['drift_mode'],psfi['B_mode'],psfi['inverse_mode']['name'],psfi['inverse_mode']['alpha'],Sabp['basis_name'])),
                                                    fontsize=ftz_text, linespacing=1.45 )
        

def format_axes(fig):

    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        ax.tick_params(labelbottom=False, labelleft=False)






