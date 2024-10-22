import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors


class VVMPlot():

    def __init__(self):
        plt.rcParams.update({'font.size':20})

    def hovmoller(self,
                  x=np.arange(128),
                  time=np.arange(720),
                  data=np.random.rand(720,128),
                  figsize=(10,12),
                  cmap=cm.Spectral_r,
                  levels=np.arange(0,1.1,1e-1),
                  x_ticks=np.round(np.linspace(0,25.6,5),1),
                  time_ticks=np.hstack(( np.arange(5,25,2), np.arange(1,6,2))),
                  title=r'$NO_{x}$',
                  case_name='pbl',
                  region='domain average',
                  cb_title='[ppb]',
                  path_savefig='.'
                  ):

        fig, ax = plt.subplots(figsize=figsize)

        Plot = ax.pcolormesh(x,time,data,cmap=cmap,norm=colors.BoundaryNorm(levels,cmap.N))
        cb = plt.colorbar(Plot,orientation='vertical',extend='max',ticks=levels)
        cb.ax.tick_params(length=8)
        cb.ax.set_title(cb_title,fontsize=18)

        ax.set_xticks(np.linspace(0,x[-1],5),x_ticks)
        ax.set_yticks(np.arange(0,721,60),time_ticks)
        ax.set_xlim(0,x[-1])
        ax.set_ylim(0,720)
        ax.tick_params(length=8)
        ax.grid(':')
        ax.set_xlabel('X')
        ax.set_ylabel('LST')
        ax.set_title(title,loc='left')
        ax.set_title(region+'\ncase: '+case_name,loc='right')
        plt.savefig(path_savefig+'/'+title,dpi=300)
        #plt.show()
        plt.close()

    def BL_height(self,
                  time=np.arange(720),
                  zc=np.linspace(0,1,50),
                  data_lines=np.random.rand(7,720),
                  data_shading=np.random.rand(50,720),
                  figsize=(16,10),
                  cmap=cm.RdBu_r,
                  levels=np.arange(-0.05,0.06,0.01),
                  time_ticks=np.hstack(( np.arange(5,25,2), np.arange(1,6,2))),
                  label=[r'$\theta$ + 0.5 K',r'max d$\theta$/dz','TKE','Enstrophy',r"$top (\overline{w'\theta'}+)$",r"$min (\overline{w'\theta'})$",r"$top (\overline{w'\theta'}-)$"],
                  line_color=['k','darkgray','hotpink','seagreen','royalblue','dodgerblue','lightskyblue'],
                  title=r'vertical $\theta$ transport',
                  case_name='pbl',
                  region='domain average',
                  cb_title='[mK/s]',
                  path_savefig='.'
                  ):
        
        fig, ax = plt.subplots(figsize=figsize)

        Plot = ax.pcolormesh(time,zc,data_shading,cmap=cmap,norm=colors.BoundaryNorm(levels,cmap.N))
        cb = plt.colorbar(Plot,orientation='vertical',extend='both',ticks=levels)
        cb.ax.tick_params(length=8)
        cb.ax.set_title(cb_title,fontsize=18)

        for i in range(len(data_lines)):
            plt.plot(time,data_lines[i],'o',color=line_color[i],markersize=3,label=label[i])
        
        ax.set_xticks(np.arange(0,721,60),time_ticks)
        ax.set_xlim(0,720)
        ax.set_ylim(0,zc[-1])
        ax.legend(fontsize=15,loc='upper right')
        ax.grid(':')
        ax.set_xlabel('LST')
        ax.set_ylabel('height')
        ax.set_title(title,loc='left')
        ax.set_title(region+'\ncase: '+case_name,loc='right')
        plt.savefig(path_savefig+'/'+title,dpi=300)
        #plt.show()
        plt.close()
