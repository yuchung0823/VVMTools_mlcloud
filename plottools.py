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
                  cb_title='[ppb]'
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
        ax.set_title('case: '+case_name,loc='right')

        plt.show()

    
Plot = VVMPlot()
Plot.hovmoller()