import os
import dill
import numpy as np

import matplotlib.colors as colors

from sfiabp.base  import base2ptrigo
from sfiabp.base  import base2pmesh


"-----------------------------------------------------------"
" interpreter(Obj) : convert a path to a list list of dict  "
"-----------------------------------------------------------"


def interpreter(Obj,Prefix=''):

    if isinstance(Obj,(dict,list)):

        if isinstance(Obj,dict):
            list_Sabp = [[Obj]]
            list_Sabp[0][0]['filename'] = ""
        elif isinstance(Obj,list):
            if isinstance(Obj,list):
                list_Sabp = [Obj]
                for i in range(len(list_Sabp[0])):
                    list_Sabp[0][i]['filename'] = "" 
   
    elif isinstance(Obj,str):
        
        if os.path.isfile(Obj):
            PathFile = Obj
            with open( PathFile, 'rb' ) as inp:    
                sabp = dill.load(inp)
            if isinstance(sabp,dict):
                list_Sabp = [[sabp]]
                list_Sabp[0][0]['filename'] = os.path.basename(PathFile) 
            elif isinstance(sabp,list):
                list_Sabp = [sabp]
                for i in range(len(list_Sabp[0])):
                    list_Sabp[0][i]['filename'] = os.path.basename(PathFile)
            
        elif os.path.isdir(Obj):
            PathFolder = Obj
            # collect all the file in Paath
            ls = []
            for x in os.listdir(PathFolder):
                if x.startswith(Prefix) and x.endswith('.pkl'):
                    ls.append(x)
            list_fname = [PathFolder + '/' + ls[i] for i in range(len(ls)) ]
            list_fname.sort()
            list_Sabp = [ [] for i in range(len(list_fname)) ]
            for i, fnamei in enumerate(list_fname):
                with open( fnamei, 'rb' ) as inp:   
                    sabp = dill.load(inp)
                # in case of S dict
                if isinstance(sabp,dict):
                    list_Sabp[i].append(sabp) 
                    list_Sabp[i][0]['filename'] = os.path.basename(fnamei)
                # in case of list
                elif isinstance(sabp,list):
                    list_Sabp[i] = sabp
                    for j in range(len(sabp)):
                        list_Sabp[i][j]['filename'] = os.path.basename(fnamei)

    return list_Sabp


" ------------------------------------------------------------------ "
" implementLff: add lff function to each dict Sabp of the list_Sabp "
" ------------------------------------------------------------------ "


def implementLff(list_Sabp,DicOpt):

    # implement the 3d function (key 'lff') for each Sabp dict
    for i in range(len(list_Sabp)):
        for j in range(len(list_Sabp[i])):
            Sabp = list_Sabp[i][j]
            if 'active_vel' not in Sabp:
                Sabp['active_vel'] = Sabp['cof1p'][0][2]
            if Sabp['basis_name'] == 'Trigo': 
                Sabp['lff'] = base2ptrigo.InitListFunc(Sabp['Order'], Sabp['FuncRad'], Sabp['VectorRad'],Sabp['cof2p'][0])[0]
            elif Sabp['basis_name'] == 'Mesh':
                # use mask_thres argument
                mbol = Sabp['psfi']['histo'][0]>DicOpt['mask_thres']
                # refresh histo
                Sabp['psfi']['histo'] = [Sabp['psfi']['histo'][0]*mbol]
                # reshape coefficient
                tabcof = base2pmesh.TocofCat( Sabp['vecr'], Sabp['veca'], Sabp['veca'], Sabp['cof2p'][0] )
                # get basis function
                lbasecat = base2pmesh.polarmesh( Sabp['vecr'], Sabp['veca'], Sabp['veca'], Sabp['FuncKern'])[1]
                # update cof with mask_thres 
                new_tabcof = [ tabcof[i]*mbol for i in range(3) ]
                # create function
                Sabp['lff'] = base2pmesh.Init_ListFuncFull( new_tabcof, lbasecat )


" ------------------------------------------------------------------ "
" plotmesh_versus: plot 3 quads (histogram, ver, omega) "
" ------------------------------------------------------------------ "


def plotmesh_versus(fig,axes,Sabp,dij,label,DicOpt):
    
    fontsize_title = 9

    npt_angle = 32+1
    vi = np.linspace(0, 2*np.pi, num=npt_angle, endpoint=True) + DicOpt['tishift']
    vj = np.linspace(0, 2*np.pi, num=npt_angle, endpoint=True) + DicOpt['tjshift']
    tig_eval, tjg_eval = np.meshgrid(vi[:-1], vj[:-1], indexing = 'ij')
    tig, tjg = np.meshgrid(vi, vj, indexing = 'ij')
    
    vecr = Sabp['psfi']['histo_vecr']
    veca = Sabp['psfi']['histo_veca']
    dvecr = vecr[1]-vecr[0]
    dveca = veca[1]-veca[0]

    ## gs10 / histogram
    # index histo
    ir = np.where(dij-vecr >= 0, dij-vecr, np.inf).argmin()
    histo_ir = Sabp['psfi']['histo'][0][ir]
    # tig_histo, tjg_histo = np.meshgrid(veca[:-1], veca[:-1], indexing = 'ij')
    tig_histo, tjg_histo, histo_ir = reshape_histo(veca,veca,vi,vj,histo_ir)
    pcm0 = axes[0].pcolormesh( tig_histo, tjg_histo, histo_ir,
                                    cmap='viridis',norm=colors.LogNorm(vmin=1,vmax=1e4,clip=True))  
    fig.colorbar(pcm0)
    axes[0].set_title( "(%s) "%(label) + 'histogram $d\\theta$=%.1f°, dr=%.1f'%(dveca*180/np.pi,dvecr),fontsize=fontsize_title)
    axes[0].set_xlabel('$\\theta_1$')
    axes[0].set_ylabel('$\\theta_2$')
    
    ## gs10 / mesh lff1
    list_titles = [ "(%s) "%(label) + "$v_r$", "(%s) "%(label) + "$\\omega$"]
    list_vmx = [ [-2,2], [-1,1] ]
    # list_cbarmesh = []  
    for i,(iax,iff) in enumerate(zip([1,2],[0,2])):
        pcm0 = axes[iax].pcolormesh( tig, tjg, Sabp['lff'][iff](dij, tig_eval, tjg_eval), cmap='RdBu_r', vmin=list_vmx[i][0], vmax=list_vmx[i][1] )
        fig.colorbar(pcm0)
        axes[iax].set_title(list_titles[i],fontsize=fontsize_title)        
        axes[iax].set_xlabel('$\\theta_1$')
        axes[iax].set_ylabel('$\\theta_2$')


" ------------------------------------------------------------------ "
" plotmesh: plot 4 quads (histogram, ver, vtheta, omega) "
" ------------------------------------------------------------------ "


def plotmesh(fig,axes,Sabp,dij,label,DicOpt):
    
    ftz_mesh_title = 10
    ftz_label = 9

    npt_angle = 32+1
    vi = np.linspace(0, 2*np.pi, num=npt_angle, endpoint=True) + DicOpt['tishift']
    vj = np.linspace(0, 2*np.pi, num=npt_angle, endpoint=True) + DicOpt['tjshift']
    tig_eval, tjg_eval = np.meshgrid(vi[:-1], vj[:-1], indexing = 'ij')
    tig, tjg = np.meshgrid(vi, vj, indexing = 'ij')
    
    vecr = Sabp['psfi']['histo_vecr']
    veca = Sabp['psfi']['histo_veca']
    dvecr = vecr[1]-vecr[0]
    dveca = veca[1]-veca[0]

    ## gs10 / histogram
    # index histo
    ir = np.where(dij-vecr >= 0, dij-vecr, np.inf).argmin()
    histo_ir = Sabp['psfi']['histo'][0][ir]
    # tig_histo, tjg_histo = np.meshgrid(veca[:-1], veca[:-1], indexing = 'ij')
    tig_histo, tjg_histo, histo_ir = reshape_histo(veca,veca,vi,vj,histo_ir)
    pcm0 = axes[0].pcolormesh( tig_histo, tjg_histo, histo_ir,
                                    cmap='viridis',norm=colors.LogNorm(vmin=1,vmax=1e4,clip=True))  
    fig.colorbar(pcm0)
    axes[0].set_title( "%s"%(label) + 'histogram $d\\theta$=%.1f°, dr=%.1f'%(dveca*180/np.pi,dvecr),fontsize=ftz_mesh_title)
    axes[0].set_xlabel('$\\theta_1$',fontsize=ftz_label)
    axes[0].set_ylabel('$\\theta_2$',fontsize=ftz_label)
    
    ## gs10 / mesh lff1
    list_titles = [ "%s"%(label) + "$v_r$", "%s"%(label) + "$v_\\theta$", "%s"%(label) + "$\\omega$"]
    list_vmx = [ [-2,2], [-2,2], [-1,1] ]
    # list_cbarmesh = []  
    for i,(iax,iff) in enumerate(zip([1,2,3],[0,1,2])):
        pcm0 = axes[iax].pcolormesh( tig, tjg, Sabp['lff'][iff](dij, tig_eval, tjg_eval), cmap='RdBu_r', vmin=list_vmx[i][0], vmax=list_vmx[i][1] )
        fig.colorbar(pcm0)
        axes[iax].set_title(list_titles[i],fontsize=ftz_mesh_title)        
        axes[iax].set_xlabel('$\\theta_1$',fontsize=ftz_label)
        axes[iax].set_ylabel('$\\theta_2$',fontsize=ftz_label)


" ------------------------------------------------------------------ "
" plotmeshVstheo: plot 9 quads (histogram, ver, vtheta, omega) "
" ------------------------------------------------------------------ "


def plotmeshVstheo(fig,axes,Sabp,dij,DicOpt):

    ftz_mesh_title = 9
    ftz_tick = 9
    ftz_label = 9

    lff = DicOpt['lff']

    npt_angle = 32+1
    vi = np.linspace(0, 2*np.pi, num=npt_angle, endpoint=True) + DicOpt['tishift']
    vj = np.linspace(0, 2*np.pi, num=npt_angle, endpoint=True) + DicOpt['tjshift']
    tig_eval, tjg_eval = np.meshgrid(vi[:-1], vj[:-1], indexing = 'ij')
    tig, tjg = np.meshgrid(vi, vj, indexing = 'ij')
    
    vecr = Sabp['psfi']['histo_vecr']
    veca = Sabp['psfi']['histo_veca']
    dvecr = vecr[1]-vecr[0]
    dveca = veca[1]-veca[0]

    ## gs11 / Mesh
    # histogram
    # index histo
    ir = np.where(dij-vecr >= 0, dij-vecr, np.inf).argmin()
    histo = Sabp['psfi']['histo'][0][ir]
    # tig_histo, tjg_histo = np.meshgrid(veca[:-1], veca[:-1], indexing = 'ij')
    tig_histo, tjg_histo, histo = reshape_histo(veca,veca,vi,vj,histo)
    pcm0 = axes[0].pcolormesh( tig_histo, tjg_histo, histo,
                                    cmap='viridis',norm=colors.LogNorm(vmin=1,vmax=1e4,clip=True))  
    cbar = fig.colorbar(pcm0)
    axes[0].set_title('histogram $d\\theta$=%.1f°, dr=%.1f'%(dveca*180/np.pi,dvecr),fontsize=ftz_mesh_title)
    axes[0].set_xlabel('$\\theta_1$')
    axes[0].set_ylabel('$\\theta_2$')
    axes[0].set_aspect('equal')
    axes[0].tick_params(axis='both',which='major',labelsize=ftz_tick)
    cbar.ax.tick_params(labelsize=ftz_tick)

    # sfi / mesh velocities vr, vtheta, omega 
    list_titles = ["$v_r$","$v_\\theta$","$\\omega$"]
    list_vmx = [ [-2,2], [-1,1], [-1,1] ]
    for i,axi in enumerate( [fig.axes[ix] for ix in [1,4,7]] ):
        pcm0 = axi.pcolormesh( tig, tjg, Sabp['lff'][i](dij, tig_eval, tjg_eval), cmap='RdBu_r', vmin=list_vmx[i][0], vmax=list_vmx[i][1] )
        cbar = fig.colorbar(pcm0)
        axi.set_title(list_titles[i],fontsize=ftz_mesh_title)        
        axi.set_xlabel('$\\theta_1$')
        axi.set_ylabel('$\\theta_2$')
        axi.set_aspect('equal')
        axi.tick_params(axis='both',which='major',labelsize=ftz_tick)
        axi.tick_params(labelsize=ftz_tick)
        cbar.ax.tick_params(labelsize=ftz_tick)

    # exact / mesh velocities vr, vtheta, omega 
    list_titles = ["$v_{r,theo}$","$v_{\\theta,theo}$","$\\omega_{theo}$"]
    list_vmx = [ [-2,2], [-1,1], [-1,1] ]
    for i,axi in enumerate( [fig.axes[ix] for ix in [2,5,8]] ):
        pcm0 = axi.pcolormesh( tig, tjg, lff[i](dij, tig_eval, tjg_eval), cmap='RdBu_r', vmin=list_vmx[i][0], vmax=list_vmx[i][1] )
        cbar = fig.colorbar(pcm0)
        axi.set_title(list_titles[i],fontsize=ftz_mesh_title)        
        axi.set_xlabel('$\\theta_1$')
        axi.set_ylabel('$\\theta_2$')
        axi.set_aspect('equal')
        axi.tick_params(axis='both',which='major',labelsize=ftz_tick)
        cbar.ax.tick_params(labelsize=ftz_tick)

    # err / mesh velocities vr, vtheta, omega 
    list_titles_err = ["|$v_r$-$v_{r,theo}|$","$|\\omega$-$\\omega_{theo}|$"]
    list_vmx_err = [ [-2,2], [-1,1] ]
    for i,axi in enumerate( [fig.axes[ix] for ix in [3,6]] ):
        pcm0 = axi.pcolormesh( tig, tjg, np.abs(Sabp['lff'][i](dij, tig_eval, tjg_eval)-lff[i](dij, tig_eval, tjg_eval)),
                                                                         cmap='RdBu_r', vmin=list_vmx_err[i][0], vmax=list_vmx_err[i][1] )
        cbar = fig.colorbar(pcm0)
        axi.set_title(list_titles_err[i],fontsize=ftz_mesh_title)        
        axi.set_xlabel('$\\theta_1$')
        axi.set_ylabel('$\\theta_2$')
        axi.set_aspect('equal')
        axi.tick_params(axis='both',which='major',labelsize=ftz_tick)
        cbar.ax.tick_params(labelsize=ftz_tick)


" ------------------------------------------------------------------ "
" plot: vr, vtheta, omega as function of the distance r "
" ------------------------------------------------------------------ "


def plotRad(fig,axes,dij,thetai,thetaj,list_lff,list_label,DicOpt,histo=[],vecr=[]):

    ftz_title = 9
    ftz_label = 9
    ftz_tick = 9
    ftz_legend = 9 

    ## plot ver, vtheta, omega 
    list_ylim_rad = [ (-10,10), (-1,1), (-1,1) ]
    list_ylabel = ['$v_r$ $(\\mu m.s^{-1})$','$v_\\theta$ $(\\mu m.s^{-1})$',"$\\omega$ $(s^{-1})$"] 
    r = np.linspace(DicOpt['rlim'][0],DicOpt['rlim'][1],100)

    for i,axi in enumerate( axes ):
        for j in range(len(list_lff)):
            axi.plot(r,list_lff[j][i](r, thetai, thetaj),label=list_label[j])

        axi.set_xlim(r[0],r[-1])
        axi.set_ylim(list_ylim_rad[i])
        axi.set_ylabel(list_ylabel[i],fontsize=ftz_label)
        axi.tick_params(axis='both',which='major',labelsize=ftz_tick)
        # vertical line, current distance r
        axi.axvline(dij,ls='--',c='red',lw=1)
        # first shaded area
        axi.axvspan(0,DicOpt['d'],alpha=0.3,color='grey')
        # second shaded area (collision effect)
        # z = 2*Sabp['cof1p'][0][2]*Sabp['psfi']['dtframe']
        # fig.axes[8].axvspan(d,d+z,alpha=0.1,color='grey')
        # fig.axes[9].axvspan(d,d+z,alpha=0.1,color='grey')

    ## plot histo1d
    if type(histo) is np.ndarray:
        vr,h1,h1n=histo1d(vecr,histo)
        kfach1d = np.abs( np.array([ list_ylim_rad[i][0]*0.5 for i in range(3) ]) )
        for i,axi in enumerate( axes ):
            # histogram 1d
            axi.plot(vr,h1n*kfach1d[i]+list_ylim_rad[i][0],ls='-',label='pair cor.',c='0.8')

    # set legend
    axes[0].legend(ncol=1,loc='upper right',frameon=False,fontsize=ftz_legend)
    axes[0].set_title( "r=%.1f"%(dij) + ",  $\\theta_1$=%.2f"%(thetai) + ",  $\\theta_2$=%.2f"%(thetaj),fontsize=ftz_title)
    axes[2].set_xlabel("$r$ $(\\mu m)$")


" ------------------------------------------------------------------ "
" plot: vr, vtheta, omega as function of the angle "
" ------------------------------------------------------------------ "


def plotAng(fig,axes,dij,thetai,thetaj,list_lff,list_label,DicOpt):
    
    ftz_title = 9
    ftz_label = 9
    ftz_tick = 9
    ftz_legend = 9 

    ## plot ver, vtheta, omega 
    vtheta = np.linspace(0,2*np.pi,100)
    list_ylim_theta = [ (-10,10), (-1,1), (-1,1) ]
    list_ylabel = ['$v_r$ $(\\mu m.s^{-1})$','$v_\\theta$ $(\\mu m.s^{-1})$',"$\\omega$ $(s^{-1})$"] 

    for i,axi in enumerate( axes ):
        for j in range(len(list_lff)):
            # axi.plot(vtheta,list_lff[j][i](dij, thetai, vtheta),label="$\\theta_1$=%.2f, "%(thetai) + list_label[j],ls='-' )
            axi.plot(vtheta,list_lff[j][i](dij, thetai, vtheta),label=list_label[j],ls='-' )

        # axi.plot(vtheta,Sabp['lff'][i](dij, vtheta, thetaj),label=label,ls='-',color='tab:orange')
        axi.set_xlim(vtheta[0],vtheta[-1])    
        axi.set_ylim(list_ylim_theta[i])
        axi.set_ylabel(list_ylabel[i],fontsize=ftz_label)
        axi.tick_params(axis='both',which='major',labelsize=ftz_tick)

    axes[0].set_title( "$\\theta_1$=%.2f"%(thetai),fontsize=ftz_title)    
    # axes[0].legend( ncol=2, bbox_to_anchor=[0,1], loc='lower left', frameon=False, fontsize=ftz_tick )
    axes[0].legend( ncol=1,loc='upper right',frameon=False,fontsize=ftz_legend)
    axes[2].set_xlabel("$\\theta_2$ $(rad)$",fontsize=ftz_label)


" ------------------------------------------------------------------ "
" Additional Plots "
" ------------------------------------------------------------------ "


def plotCof(fig,axes,Sabp,DicOpt):

    ftz_title = 9
    ftz_label = 9
    ftz_tick = 9

    ## gs10 / phicof 
    # gs12, ax7, phi cof vr
    axes.plot(np.abs(Sabp['cof2p'][0][0:50].flatten()),'o',markersize=4,markerfacecolor='none')
    axes.set_yscale('log')
    axes.set_ylim(1e-1,1e4)
    # fig.axes[0].set_xlabel('num',fontsize=ftz_label)
    axes.tick_params(axis='both',which='major',labelsize=ftz_tick)
    axes.set_title('|50 first coef in cof2p|',fontsize=ftz_title)  


def plotAddInfo(fig,axes,Sabp,DicOpt):

    ftz_title = 9
    ftz_label = 9
    ftz_tick = 9

    r = np.linspace(DicOpt['rlim'][0],DicOpt['rlim'][1],100)

    ## gs10 / radial basis function
    if Sabp['basis_name'] == 'Trigo':
        
        lbase_rad = base2ptrigo.polartrigo( Sabp['Order'], Sabp['FuncRad'], Sabp['VectorRad'] )[2]
        for f in lbase_rad:
            y = f(r)
            y = y/np.max(np.abs(y))
            axes.plot(r,y)
        axes.set_title('normalize radial basis function',fontsize=ftz_title)
        axes.set_xlabel("$r$ $(um)$",fontsize=ftz_label)
        axes.tick_params(axis='both',which='major',labelsize=ftz_tick)
        axes.set_xlim(r[0],r[-1])
        axes.set_ylim(0,1)
    
    elif Sabp['basis_name'] == 'Mesh':

        lbase_rad = base2pmesh.polarmesh( Sabp['vecr'], Sabp['veca'], Sabp['veca'], 'Step' )[2]
        for f in lbase_rad:
            y = f(r)
            y = y/np.max(np.abs(y))
            axes.plot(r,y)
        axes.set_title('normalize radial basis function',fontsize=ftz_title)
        axes.set_xlabel("$r$ $(um)$",fontsize=ftz_label)
        axes.tick_params(axis='both',which='major',labelsize=ftz_tick)
        axes.set_xlim(r[0],r[-1])
        axes.set_ylim(0,1)


" ------------------------------------------------------------------ "
" Additional functions "
" ------------------------------------------------------------------ "


def histo1d(vecr,h):
    axr = vecr[:-1] + np.diff(vecr)/2
    lenr = np.shape(h)[0]
    h1d = np.array([ np.sum(h[i,:,:]) for i in range(lenr) ])
    h1dn = np.copy(h1d)/axr
    h1dn = h1dn/h1dn[-1]
    return axr, h1d, h1dn


def reshape_histo(vi,vj,vin,vjn,histo):
    
    vic = np.mean(np.array([vi[0],vi[-1]]))
    vjc = np.mean(np.array([vj[0],vj[-1]]))
    vinc = np.mean(np.array([vin[0],vin[-1]]))
    vjnc = np.mean(np.array([vjn[0],vjn[-1]]))

    dti = vi[1]-vi[0]; dtj = vj[1]-vj[0]
    nti = np.floor_divide(vinc-vic,dti)
    ntj = np.floor_divide(vjnc-vjc,dtj)
    histor = np.roll(histo,(int(nti),int(-ntj)),axis=(0,1))
    vir = np.linspace(vin[0],vin[-1],int((vin[-1]-vin[0])/dti+1))
    vjr = np.linspace(vjn[0],vjn[-1],int((vjn[-1]-vjn[0])/dtj+1))
    tir, tjr = np.meshgrid(vir[:-1], vjr[:-1], indexing = 'ij')
    tir, tjr = np.meshgrid(vir, vjr, indexing = 'ij')
    return tir, tjr, histor
