
import matplotlib.animation as animation
import matplotlib.cm as cm 
import matplotlib.pyplot as plt
import matplotlib

import numpy as np


###############################
##                          ##
###############################


def create_video(list_frame,d,xframelim,yframelim,dtframe,ksamp,PathVideo):

    def init():
        # initialize an empty list of cirlces
        return []

    def update(i,list_frame,r,xframelim,yframelim,dtframe):
        print('i =',i)
        fsz = 8.5
        ax.clear()
        ax.set_xlim(xframelim[0],xframelim[1]); ax.set_ylim(yframelim[0],yframelim[1])
        ax.set_xlabel('x (um)',fontsize=fsz); ax.set_ylabel( 'y (um)',fontsize=fsz) 
        ax.tick_params(axis='y', which='major', labelsize=fsz)
        ax.tick_params(axis='x', which='major', labelsize=fsz)
        tcpt = i*dtframe
        ax.set_title('i = '+str(i)+', t (s) = '+str("%.1f"% tcpt),fontsize=fsz)
        ax.set_aspect(1)
        npar = len(list_frame[i])
        patches = []
        for j in range(npar):
            circle_i = plt.Circle((list_frame[i][j,0], list_frame[i][j,1]), r, color='royalblue')
            darr = np.array([ r*np.cos(list_frame[i][j,2]), r*np.sin(list_frame[i][j,2]) ])
            arrow_i = plt.arrow( list_frame[i][j,0], list_frame[i][j,1], darr[0], darr[1],
                                    width=0.5,edgecolor='None',facecolor='k')
            patches.append(ax.add_patch(circle_i))
            patches.append(ax.add_patch(arrow_i))
        return patches

    # sample the data
    list_frame_samp  = list_frame[::ksamp]
    matplotlib.use('Agg')
    fig,ax = plt.subplots()
    # test_plot(0,list_frame, list_probe, r, blsize)
    ani = animation.FuncAnimation(fig=fig, func=update, init_func=init, fargs=(list_frame_samp,d/2,xframelim,yframelim,dtframe*ksamp,),
                                        frames=len(list_frame_samp), interval=dtframe*1000, cache_frame_data=True,repeat=False)

    ani.save( PathVideo )
    return True

def test_plot(i,list_frame, r, blsize, dtframe):
    print('i=',i)
    fsz = 8
    ax.clear()
    ax.set_xlim(0,blsize[0]); ax.set_ylim(0,blsize[1])
    ax.set_xlabel('x (um)',fontsize=fsz); ax.set_ylabel( 'y (um)',fontsize=fsz) 
    ax.tick_params(axis='y', which='major', labelsize=fsz)
    ax.tick_params(axis='x', which='major', labelsize=fsz)
    tcpt = i*dtframe
    ax.set_title('i = '+str(i)+', t (s) = '+str("%.1f"% tcpt),fontsize=fsz)
    ax.set_aspect(1)
    npar = len(list_frame[i])
    # patches = []
    for j in range(npar):
        circle_i = plt.Circle((list_frame[i][j,0], list_frame[i][j,1]), r, color='royalblue')
        darr = np.array([ r*np.cos(list_frame[i][j,2]), r*np.sin(list_frame[i][j,2]) ])
        arrow_i = plt.arrow( list_frame[i][j,0], list_frame[i][j,1], darr[0], darr[1],
                                width=0.5,edgecolor='None',facecolor='k')
        # patches.append(ax.add_patch(circle_i))
        # patches.append(ax.add_patch(arrow_i))
        ax.add_patch(circle_i)
        ax.add_patch(arrow_i)
    plt.show()

def update_probe( i, list_frame, list_probe, r, blsize, dtframe ):
    print('i=',i)
    fsz = 8.5
    ax.clear()
    ax.set_xlim(0,blsize[0]); ax.set_ylim(0,blsize[1])
    ax.set_xlabel('x (um)',fontsize=fsz); ax.set_ylabel( 'y (um)',fontsize=fsz) 
    ax.tick_params(axis='y', which='major', labelsize=fsz)
    ax.tick_params(axis='x', which='major', labelsize=fsz)
    tcpt = i*dtframe
    ax.set_title('i = '+str(i)+', t (s) = '+str("%.1f"% tcpt),fontsize=fsz)
    ax.set_aspect(1)
    npar = len(list_frame[i])
    patches = []
    for j in range(npar):
        if list_probe[i][j]>-1:
            circle_i = plt.Circle((list_frame[i][j,0], list_frame[i][j,1]), r, color='firebrick')
        else:
            circle_i = plt.Circle((list_frame[i][j,0], list_frame[i][j,1]), r, color='royalblue')
        darr = np.array([ r*np.cos(list_frame[i][j,2]), r*np.sin(list_frame[i][j,2]) ])
        arrow_i = plt.arrow( list_frame[i][j,0], list_frame[i][j,1], darr[0], darr[1],
                                width=0.5,edgecolor='None',facecolor='k')
        patches.append(ax.add_patch(circle_i))
        patches.append(ax.add_patch(arrow_i))
    return patches

def test_plot_probe( i, list_frame, list_probe, r, blsize, dtframe ):
    print('i=',i)
    fsz = 8
    ax.clear()
    ax.set_xlim(0,blsize[0]); ax.set_ylim(0,blsize[1])
    ax.set_xlabel('x (um)',fontsize=fsz); ax.set_ylabel( 'y (um)',fontsize=fsz) 
    ax.tick_params(axis='y', which='major', labelsize=fsz)
    ax.tick_params(axis='x', which='major', labelsize=fsz)
    tcpt = i*dtframe
    ax.set_title('i = '+str(i)+', t (s) = '+str("%.1f"% tcpt),fontsize=fsz)
    ax.set_aspect(1)
    npar = len(list_frame[i])
    # patches = []
    for j in range(npar):
        if list_probe[i][j]>-1:
            circle_i = plt.Circle((list_frame[i][j,0], list_frame[i][j,1]), r, color='firebrick')
        else:
            circle_i = plt.Circle((list_frame[i][j,0], list_frame[i][j,1]), r, color='royalblue')
        darr = np.array([ r*np.cos(list_frame[i][j,2]), r*np.sin(list_frame[i][j,2]) ])
        arrow_i = plt.arrow( list_frame[i][j,0], list_frame[i][j,1], darr[0], darr[1],
                                width=0.5,edgecolor='None',facecolor='k')
        # patches.append(ax.add_patch(circle_i))
        # patches.append(ax.add_patch(arrow_i))
        ax.add_patch(circle_i)
        ax.add_patch(arrow_i)
    plt.show()


##################################
##                              ##
##################################


def ToMatShow (M):
    A = np.copy(M.transpose())
    B = np.copy(A[::-1,:])
    return B

def ConvertFrame(frame,lx,ly,A):

    MatRot = lambda A : np.array([ [np.cos(A),-np.sin(A)],[np.sin(A),np.cos(A)] ]) # rotation matrix
    frameRes = np.zeros(np.shape(frame))
    npar = np.shape(frame)[0]
    
    for i in range(npar):    
        # Xijn = np.matmul(MatRot(-Aj),Xij) 
        v = frame[i,:3]
        frameRes[i,0] = v[0]
        frameRes[i,1] = -v[1]+ly
        frameRes[i,2] = v[2]
    return frameRes

# Data plot functions ##############

def PlotDataFrame (frame, lx, ly, PltName, opt,NumFig=0): 

    npar,ncol = frame.shape

    plt.figure(NumFig)
    plt.title(PltName)
    #cmap = plt.cm.winter
    cind = np.arange(npar)
    plt.scatter(frame[:,0], frame[:,1], c = cind, cmap = 'winter')
    # axe limit
    if isinstance(lx,float):
        lx = np.array([0,lx])
        ly = np.array([0,ly])
    if isinstance(lx,int):
        lx = np.array([0,lx])
        ly = np.array([0,ly])
        
    plt.xlim(lx[0], lx[1])
    plt.ylim(ly[0], ly[1])

    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    
    # draw arrows
    # lg = lx[1]/35
    # wid = 0.0015
    lg = lx[1]/45
    wid = 0.0025

    if 'arrow' in opt:
        for k in range(npar):  
            #plt.arrow(frame[k,0], frame[k,1], lg*np.cos(frame[k,2]), lg*np.sin(frame[k,2]), width = wid,head_width=0,length_includes_head=False)
            #plt.arrow(frame[k,0], frame[k,1], lg*np.cos(frame[k,2]), lg*np.sin(frame[k,2]), width = wid,head_width=0,length_includes_head=False)
            plt.quiver(frame[k,0], frame[k,1], lg*np.cos(frame[k,2]), lg*np.sin(frame[k,2]),width=wid,angles='xy',scale_units='xy',scale=1)
    
    if 'note' in opt:
        if ncol == 4:
            for i,k in enumerate(frame[:,3]):
                texti = str(int(k))
                plt.annotate(texti,frame[i,0:2])
        else:
            for k in range (npar):
                texti = str(int(k))
                plt.annotate(texti,frame[int(k),0:2])        

    #plt.annotate(str(1), frame[1,0:2], xycoords = 'data') 
    #plt.arrow(frame[1,0], frame[1,1], lg*np.cos(frame[1,2]), lg*np.sin(frame[1,2]), width = wid,head_width=0,length_includes_head=False)


def PlotDataVideo (data, tp, lx ,ly, opt): 

    nfra = len(data[:])

    for i in range(nfra):
        FrameName = 'Frame = ' + str(i)
        PlotDataFrame(data[i],lx,ly,FrameName,opt)
        plt.pause(tp)
        plt.clf()

def PlotCorelSingle(data,xedges,yedges):
        
    # reorient corr in cartesian convention
    data_car  = np.copy(np.transpose(data))
    data_car  = np.copy(np.flip(data_car,axis=0))
    extent_opt = [xedges[0],xedges[-1],yedges[0],yedges[-1]]
    plt.clf()
        
    # plt.matshow(corr_car,fignum=2,interpolation='none',cmap=plt.get_cmap('jet'))
    plt.matshow(data_car,fignum=1,interpolation='none',cmap='gray_r',extent=extent_opt)
    plt.title("Original Image")    
    plt.colorbar()
        
    # axes limit
    #plt.xlim(xedges[0],xedges[-1])
    #plt.ylim(yedges[0],yedges[-1])
    pp=1
    # plt.show ()
        
    # def PlotCorelVideo(self,tp,SwMode):
        
    #     # frame number
    #     nfra = self.pcor.shape[0]

    #     for i in range (nfra):
            
    #         if SwMode == 'Cor':
    #             self.PlotCorelSingle(self.pcor[i,:,:])

    #         elif SwMode == 'Sum':
    #             self.PlotCorelSingle(self.pcorsum[i,:,:])
            
    #         plt.pause(tp)

def PlotMat(ax, M, name='matrix'):
    
    maxi = M.max()
    mini = M.min()
    nl = np.shape(M)[0]
    nc = np.shape(M)[1]

    # if norm:    
    #     fnorm = plt.Normalize(vmin=mini,vmax=maxi)
    #     mat = fnorm(M)
     
    # grid_col = np.einsum('a,b->ba',np.arange(nc),np.ones(nl))
    # grid_lin = np.einsum('a,b->ab',np.arange(nl)[::-1],np.ones(nc))
    # plot = ax.pcolor(grid_col, grid_lin, M, cmap='coolwarm')
    
    plot = ax.matshow(M, cmap='Reds')


    title = name + ', min=' + format(mini,'.3e') + ', max=' + format(maxi,'.3e')
    ax.set_title(title,fontsize=10)
    ax.set(xlabel='column number '+ str(nc))
    ax.set(ylabel='line number '+ str(nl))

    return plot
