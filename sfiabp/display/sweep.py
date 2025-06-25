
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as lines
import ipywidgets as widgets

import numpy as np
from sfiabp.base  import base2ptrigo
from sfiabp.base  import base2pmesh
from sfiabp.display import library


"""--------------------------------------------------------------------
                interactive display for Sabp
--------------------------------------------------------------------"""


def sfidisp_sweep(Path, **kwargs):

    #### function events ####

    class KeyHandler:

        def __init__(self,fig,list_Sabp,DicOpt):
            
            self.fig = fig
            self.list_Sabp = list_Sabp
            self.DicOpt = DicOpt

            self.glob = dict(
                iSfile0 = 0,
                iSfile1 = 0,
                nSfile0 = len(list_Sabp), # number of S file first dim
                nSfile1 = [ len(list_Sabp[i]) for i in range(len(list_Sabp)) ], # number of S file second dim 
                dij = 5, # current position (um)
                thetai = 0,
                thetaj = 0, # current angle (rad)
                dijinc = DicOpt['dijinc'],
                rlim = DicOpt['rlim']
                )
                        
            if DicOpt['ModeTheo'] :
                self.glob['Mode_View'] = ['spl','vs']
                self.glob['idic'] = 0
                self.glob['nidic'] = 2
            else :
                self.glob['Mode_View'] = ['spl']
                self.glob['idic'] = 0
                self.glob['nidic'] = 1

        # mouse click
        def handleEvent (self,event):
            print(event)
            if self.glob['idic'] == 0: # normal display
                if event.inaxes in fig.axes[3:7]:
                    self.glob['thetai'] = event.xdata
                    self.glob['thetaj'] = event.ydata
                    self.refresh_figure()
            elif self.glob['idic'] == 1: # mesh lff theory
                if event.inaxes in fig.axes[0:9]:
                    self.glob['thetai'] = event.xdata
                    self.glob['thetaj'] = event.ydata
                    self.refresh_figure()

        # key press function
        def on_press(self,event):
            glob = self.glob
            print(event.key)

            if event.key in ['a','e','z','left','right','up','down']:
                if event.key == 'a': 
                    glob['dij'] = (glob['dij']-glob['dijinc'])%glob['rlim'][1]
                elif event.key == 'e': 
                    glob['dij'] = (glob['dij']+glob['dijinc'])%glob['rlim'][1]
                elif event.key == 'z': 
                    glob['idic'] = (glob['idic']+1)%glob['nidic'] 
                elif event.key == 'left':
                    glob['iSfile1'] = (glob['iSfile1']-1)%glob['nSfile1'][glob['iSfile0']]
                elif event.key == 'right':
                    glob['iSfile1'] = (glob['iSfile1']+1)%glob['nSfile1'][glob['iSfile0']]
                elif event.key == 'up':
                    glob['iSfile0'] = (glob['iSfile0']-1)%glob['nSfile0']
                    glob['iSfile1'] = glob['iSfile1']%glob['nSfile1'][glob['iSfile0']]
                elif event.key == 'down':
                    glob['iSfile0'] = (glob['iSfile0']+1)%glob['nSfile0']
                    glob['iSfile1'] = glob['iSfile1']%glob['nSfile1'][glob['iSfile0']]
                self.refresh_figure()

        # refresh with two possible figure modes
        def refresh_figure(self):
            fig.clf()
            if self.glob['idic'] == 0: # normal display
                # create_figure(fig,list_Sabp,lff,tishift,tjshift,d)
                create_figure(self.fig,self.list_Sabp,self.glob,self.DicOpt)
            elif self.glob['idic'] == 1: # mesh lff theory
                # create_figure_vs(fig,list_Sabp,lff,tishift,tjshift,d)
                create_figure_vs(self.fig,self.list_Sabp,self.glob,self.DicOpt)

    #### main ####

    ## additional parameters
    DicOpt = dict(d = kwargs.get('d', 0),
              dijinc = kwargs.get('dijinc', 0.3),          
              tishift = kwargs.get('tishift', 0),         
              tjshift = kwargs.get('tjshift', 0),
              rlim = kwargs.get('rlim', [1,10]),
              lff = kwargs.get('exact_fun', []),
              mask_thres = kwargs.get('mask_thres', 0),
              Prefix = kwargs.get('Prefix', '')  )
    
    if -np.pi>DicOpt['tishift'] or np.pi<DicOpt['tishift'] or -np.pi>DicOpt['tjshift'] or np.pi<DicOpt['tjshift']:
        raise KeyError('tishift, tjshift must be in the range -pi<=x<=pi')
    
    DicOpt['ModeTheo'] = False
    if DicOpt['lff'] != [] : 
        DicOpt['ModeTheo'] = True
   
    ## get data
    list_Sabp = library.interpreter(Path,Prefix=DicOpt['Prefix'])
    library.implementLff(list_Sabp,DicOpt)
    # check list_Sabp[0][0].keys()

    # intialization
    fig = plt.figure(figsize=(18,8),constrained_layout=True)
    # global index
    keyhand = KeyHandler(fig,list_Sabp,DicOpt)
    # fig = plt.figure(figsize=(18,8))
    # create figure
    create_figure(fig,list_Sabp,keyhand.glob,DicOpt)

    fig.canvas.mpl_connect("button_press_event",keyhand.handleEvent)
    fig.canvas.mpl_connect("key_press_event",keyhand.on_press)

    return fig, keyhand
    

###########################
## create main figure ##
###########################

def create_figure(fig,list_Sabp,glob,DicOpt):

    ## cursors
    thetai = glob['thetai']
    thetaj = glob['thetaj']
    iSfile0 = glob['iSfile0']
    iSfile1 = glob['iSfile1']
    dij = glob['dij']

    Sabp = list_Sabp[iSfile0][iSfile1] # init S file

    #### structure ####

    gs = fig.add_gridspec(3,5,width_ratios=[0.04,0.3,0.9,0.8,0.04],height_ratios=[0.15,1,0.05])
    
    gs10 = gs[1,1].subgridspec(3,1)
    for i in range(3):
        fig.add_subplot(gs10[i])
    gs11 = gs[1,2].subgridspec(2,2)
    for i in range(2):
        for j in range(2):
            fig.add_subplot(gs11[i,j])
    gs12 = gs[1,3].subgridspec(3,2)
    for i in range(3):
        for j in range(2):
            fig.add_subplot(gs12[i,j])

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

    # format_axes(fig)

    ftz_mesh_title = 10 # fontsize mesh title
    ftz_title = 9 # fontsize title
    ftz_tick = 9
    ftz_label = 10
    ftz_txt = 9.3 # fontsize text

    ## plot mesh
    library.plotmesh(fig,fig.axes[3:7],Sabp,dij,'',DicOpt)
    
    ## plot the interactions as function of the distance and the angle
    histo=Sabp['psfi']['histo'][0]
    vecr=Sabp['psfi']['histo_vecr']
    if DicOpt['ModeTheo']:
        list_axes = [fig.axes[7],fig.axes[9],fig.axes[11]]    
        library.plotRad(fig,list_axes,dij,thetai,thetaj,[Sabp['lff'],DicOpt['lff']],['sfi','exact'],DicOpt, histo=histo,vecr=vecr)
        list_axes = [fig.axes[8],fig.axes[10],fig.axes[12]]
        library.plotAng(fig,list_axes,dij,thetai,thetaj,[Sabp['lff'],DicOpt['lff']],['sfi','exact'],DicOpt)
    else : 
        list_axes = [fig.axes[7],fig.axes[9],fig.axes[11]]
        library.plotRad(fig,list_axes,dij,thetai,thetaj,[Sabp['lff']],['sfi'],DicOpt,histo=histo,vecr=vecr)
        list_axes = [fig.axes[8],fig.axes[10],fig.axes[12]]
        library.plotAng(fig,list_axes,dij,thetai,thetaj,[Sabp['lff']],['sfi'],DicOpt)
    
    ## plot several cof
    library.plotCof(fig,fig.axes[0],Sabp,DicOpt)
    ## plot specific information 
    library.plotAddInfo(fig,fig.axes[2],Sabp,DicOpt)

    ### gs10 / artist ###
    # rplt = (dij-d)/(Sabp['psfi']['lcell']-d)*2+1 # dr = 4
    rplt = 2
    p,pnext = pol2cart_janus(Sabp['lff'],dij,thetai,thetaj,dt=1,rplt=rplt)
    list_janus = []
    list_janus.extend(particle(np.array([0,0,0]),r=0.5,color='black'))
    list_janus.extend(particle(p,r=0.5,color='royalblue'))
    list_janus.extend(particle(pnext,r=0.5,color='lightsteelblue'))
    listcc = [2,3,4]
    
    ax = fig.axes[1]
    for ri in listcc:
        ax.add_patch(mpatches.Circle((0,0),ri,ls='--',color='0.5',fill=False))
    for i,pp in enumerate(list_janus): 
        ax.add_patch(pp)

    ax.set_xlim(-5,5)    
    ax.set_ylim(-5,5)
    ax.set_aspect('equal')
    ax.axis('off')
                       
    fig.text(0.02,0.45,("$\\theta_i$=%.2f"%(thetai)+"\n"
                "$\\theta_j$=%.2f"%(thetaj)+"\n"
                "$v_r$=%.2f"%(Sabp['lff'][0](dij,thetai,thetaj))+"\n"
                "$v_\\theta$=%.2f"%(Sabp['lff'][1](dij,thetai,thetaj))+"\n"
                "$\\omega$=%.2f"%(Sabp['lff'][2](dij,thetai,thetaj))),fontsize=9,linespacing=1.5)
    
    ## add txt
    create_txt(fig,Sabp,glob,ftz_txt)


###########################
## create versus figure ##
###########################


def create_figure_vs(fig,list_Sabp,glob,DicOpt):

    ## cursors
    thetai = glob['thetai']
    thetaj = glob['thetaj']
    iSfile0 = glob['iSfile0']
    iSfile1 = glob['iSfile1']
    dij = glob['dij']

    Sabp = list_Sabp[iSfile0][iSfile1] # init S file

    #### structure ####

    gs = fig.add_gridspec(3,4,height_ratios=[0.15,1,0.03],width_ratios=[0.04,1,0.7,0.04])
    
    gs11 = gs[1,1].subgridspec(3,3)
    for i in range(3):
        for j in range(3):
            fig.add_subplot(gs11[i,j])
    gs12 = gs[1,2].subgridspec(3,2)
    for i in range(3):
        for j in range(2):
            fig.add_subplot(gs12[i,j])

    # format_axes(fig)

    ftz_mesh_title = 10 # fontsize mesh title
    ftz_title = 9 # fontsize title
    ftz_tick = 9
    ftz_label = 10
    ftz_txt = 9.3 # fontsize text

    ## plot mesh
    list_axes = fig.axes[0:9]
    library.plotmeshVstheo(fig,list_axes,Sabp,dij,DicOpt)

    ## plot the interactions as function of the distance and the angle
    histo=Sabp['psfi']['histo'][0]
    vecr=Sabp['psfi']['histo_vecr']
    list_axes = [fig.axes[9],fig.axes[11],fig.axes[13]]    
    library.plotRad(fig,list_axes,dij,thetai,thetaj,[Sabp['lff'],DicOpt['lff']],['sfi','exact'],DicOpt, histo=histo,vecr=vecr)
    list_axes = [fig.axes[10],fig.axes[12],fig.axes[14]]
    library.plotAng(fig,list_axes,dij,thetai,thetaj,[Sabp['lff'],DicOpt['lff']],['sfi','exact'],DicOpt)

    # txt 
    create_txt(fig,Sabp,glob,ftz_txt)


###########################
## additional functions ##
###########################

 
def format_axes(fig):
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        ax.tick_params(labelbottom=False, labelleft=False)


def pol2cart_janus( lff, r, thetai, thetaj, dt=1, rplt=2 ): # um, rad, rad
        
    p = np.array([rplt*np.cos(thetaj),rplt*np.sin(thetaj),thetai+thetaj]) # position
    er = np.array([np.cos(thetaj),np.sin(thetaj)])
    d_er = lff[0](r,thetai,thetaj)*er*dt
    etheta = np.array([-np.sin(thetaj),np.cos(thetaj)])
    d_etheta = lff[1](r,thetai,thetaj)*etheta*dt
    d_omega = lff[2](r,thetai,thetaj)*dt
    pnext = np.copy(p)
    pnext[0:2] += d_er + d_etheta
    pnext[2] += d_omega

    return p, pnext


def particle(c,r=0.5,color='lightgray'): # um, um, deg, um
        
    lw = 1.3 # linewidth
    lar = 3*r # length arrow
    # style = mpatches.ArrowStyle('Fancy', head_length=1, head_width=1.5, tail_width=0.1)
    style = mpatches.ArrowStyle('->',head_length=0.5, head_width=0.3, widthA=1.0,
                                        widthB=1.0, lengthA=0.2, lengthB=0.2, angleA=0, 
                                            angleB=0, scaleA=None, scaleB=None)
    l = [ mpatches.Circle(c[0:2], r, ec=color,fc="none",linewidth=lw), 
        # mpatches.Wedge((cx,cy),r,t+90,t+270,ec="none",fc="lightsteelblue"),
        mpatches.Wedge(c[0:2],r-0.003,c[2]*180/np.pi+90,c[2]*180/np.pi+270,ec="none",fc=color),
        # mpatches.Arrow(cx, cy, 2*r*np.cos(t*np.pi/180), 2*r*np.sin(t*np.pi/180), width=0.06, ='miter')
        mpatches.FancyArrowPatch(c[0:2],(c[0]+lar*np.cos(c[2]),c[1]+lar*np.sin(c[2])),arrowstyle = style, 
                                        mutation_scale=10, mutation_aspect=1, color=color, linewidth=lw) ]    
    return l


def create_txt(fig,Sabp,glob,ftz_txt):

    ## text
    # fig.add_artist(lines.Line2D([0.01,0.82],[0.88,0.88],linewidth=1,color='black'))

    fig.text(0.02,0.89,("$\\bf{sfidisp\\_sweep.py}$\n" 
                          "- press a, e to change the distance r\n"
                          "- click on the 2D mesh to change $\\theta_1$, $\\theta_2$\n"
                          "- press z to switch the mode view\n"
                          "- press up, down for dim0 and left, right for dim1\n  to change the S file, list_S[dim0][dim1]\n"
                          "- press q to quit"),
                            fontsize=7.5,linespacing=1.2)

# ("$\\bf{List\\ Sabp%d}$[%d][%d] : \n%s\n"%(id,iSfile0,iSfile1,Sabp['filename'])+
 
    fig.text( 0.24, 0.894, ("$\\bf{List\\ Sabp}$[%d][%d] : \n- %s\n"%(glob['iSfile0'],glob['iSfile1'],Sabp['filename'])+
               "- frame number : %d, dtframe (s): %.2f, cell length (um) %d\n"%(Sabp['psfi']['frame_weight'], Sabp['psfi']['dtframe'], Sabp['psfi']['lcell']) + 
               "- xbox : [%.1f,%.1f], ybox : [%.1f,%.1f]"%(Sabp['psfi']['xboxlim'][0],Sabp['psfi']['xboxlim'][1],Sabp['psfi']['yboxlim'][0],Sabp['psfi']['yboxlim'][1])),
                             fontsize=ftz_txt, linespacing=2)

    postxt1 = [0.61,0.958]
    posinc = -0.027
    if Sabp['psfi']['inverse_mode']['name'] == 'pinv':
        fig.text(postxt1[0],postxt1[1],("$\\bf{Sfi}$ $\\bf{info}$ : "+
                    "drift mode: %s, B mode: %s, inverse mode : %s, cond. number : %f"%(Sabp['psfi']['drift_mode'],\
                     Sabp['psfi']['B_mode'],Sabp['psfi']['inverse_mode']['name'],Sabp['psfi']['inverse_mode']['conditional_number'])),
                     fontsize=ftz_txt)

    elif Sabp['psfi']['inverse_mode']['name'] == 'tiko':
        fig.text(postxt1[0],postxt1[1],("$\\bf{Sfi}$ $\\bf{info}$ : "+
                    "drift mode: %s, B mode: %s, inverse mode : %s, alpha : %.2e"%(Sabp['psfi']['drift_mode'],\
                     Sabp['psfi']['B_mode'],Sabp['psfi']['inverse_mode']['name'],Sabp['psfi']['inverse_mode']['alpha'])),
                     fontsize=ftz_txt)

    postxt2 = [postxt1[0],postxt1[1] + posinc]
    if Sabp['basis_name'] == 'Trigo': 
        fig.text( postxt2[0], postxt2[1], ("- basis name : %s, order : %d, radial function : %s"% 
                        (Sabp['basis_name'],Sabp['Order'],Sabp['FuncRad'])), fontsize=ftz_txt)

    elif Sabp['basis_name'] == 'Mesh': 
        fig.text( postxt2[0], postxt2[1], ("- basis name : %s, kernel function : %s"%(Sabp['basis_name'],Sabp['FuncKern'])), fontsize=ftz_txt, linespacing=1.8)

 
    fig.text( postxt1[0], postxt1[1] + 2*posinc, "- active vel U(um/s): %.3f, Diffusion coefficients Dxx: %.3f, Dyy: %.3f, Dr: %.3f"%(Sabp['active_vel'],\
                                                        Sabp['D_average'][0,0],Sabp['D_average'][1,1],Sabp['D_average'][2,2]) , fontsize=ftz_txt )



