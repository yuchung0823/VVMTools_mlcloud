from vvmtools import VVMTools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors


class VVMPlot_BL(VVMTools):
    """
    A subclass of VVMTools to handle plotting for boundary layer analysis, 
    such as Hovmöller diagrams and boundary layer height visualizations.
    """
    def __init__(self,
                 case_path,
                 time=np.arange(721),
                 fontsize=25,
                 xc_ticks_linspace=5,
                 time_str_hr=5,
                 dt_min=2,
                 time_ticks_space_hr=2,
                 case_name='',
                 region_name=''
                 ):
        """
        Initializes the VVMPlot_BL instance for plotting.

        :param case_path: Path to the case simulation data.
        :param time: Array of time steps for plotting.
        :param fontsize: Font size for plot text.
        :param xc_ticks_linspace: Number of ticks for the x-axis.
        :param time_str_hr: Start time (in hours) for the time axis ticks.
        :param dt_min: Time step in the simulation.
        :param time_ticks_space_hr: Spacing of time ticks in hours.
        :param case_name: Name of the simulation case.
        :param region_name: Name of the region under analysis.
        """
        super().__init__(case_path)
        self.time = time
        self.zc = self.DIM['zc']/1000
        self.yc = self.DIM['yc']/1000
        self.xc = self.DIM['xc']/1000
        self.fs = fontsize
        self.xc_ticks_space=int(xc_ticks_linspace) # Set number of x-axis ticks
        self.xc_ticks=np.round(np.linspace(0,self.xc[-1]+(self.xc[1]-self.xc[0])/2,xc_ticks_linspace),1)
        self.time_ticks_space = int(time_ticks_space_hr*60/dt_min) # Time axis tick spacing
        self.time_ticks=np.hstack(( np.arange(time_str_hr,25,time_ticks_space_hr), np.arange(1,time_str_hr+1,time_ticks_space_hr)))
        self.case_name=case_name
        self.region_name=region_name
        plt.rcParams.update({'font.size':self.fs}) # Update matplotlib font settings globally
        
    def hovmoller(self,
                  data,
                  figsize=(14,12),
                  cmap=cm.Spectral_r,
                  levels=np.arange(0,51,2.5),
                  title=r'$NO_{x,}$ $_{sfc}$',
                  cb_title='[ppb]',
                  path_savefig='.'):
        """
        Plot a Hovmöller diagram of the input data.

        :param data: 2D array of data (time vs x-coordinates).
        :param figsize: Size of the figure.
        :param cmap: Colormap for the plot.
        :param levels: Contour levels for the color mapping.
        :param title: Title for the plot.
        :param cb_title: Title for the colorbar.
        :param path_savefig: Path to save the figure.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Create a colormap plot (pcolormesh) using x-coordinates and time
        Plot = ax.pcolormesh(self.xc,self.time,data,cmap=cmap,norm=colors.BoundaryNorm(levels,cmap.N))
        cb = plt.colorbar(Plot,orientation='vertical',extend='max',ticks=levels[::2])
        cb.ax.tick_params(length=8)
        cb.ax.set_title(cb_title,fontsize=self.fs-5)

        # Settings
        ax.set_xticks(np.linspace(0,self.xc[-1],5),self.xc_ticks)
        ax.set_yticks(self.time[::self.time_ticks_space],self.time_ticks)
        ax.set_xlim(0,self.xc[-1])
        ax.set_ylim(0,self.time[-1])
        ax.tick_params(length=8)
        ax.grid(':')
        ax.set_xlabel('X')
        ax.set_ylabel('LST')
        ax.set_title(title,loc='left')
        ax.set_title(self.case_name+'\n'+self.region_name,loc='right',fontsize=self.fs-2)
        plt.savefig(path_savefig+'/hovmoller_'+self.case_name,dpi=300,bbox_inches="tight")
        plt.close()

    def BL_height(self,
                  data_lines=np.random.rand(7,720),
                  data_shading=np.random.rand(50,720),
                  figsize=(18,10),
                  cmap=cm.RdBu_r,
                  levels=np.arange(-0.04,0.041,0.005),
                  label=[r'$\theta$ + 0.5 K',r'max d$\theta$/dz','TKE','Enstrophy',r"top$ (\overline{w'\theta'}+)$",r"min$ (\overline{w'\theta'})$",r"top$ (\overline{w'\theta'}-)$"],
                  line_color=['dimgray','hotpink','darkorange','seagreen','dodgerblue','navy','darkviolet'],
                  title=r'Vertical $\theta$ transport',
                  cb_title='[K·m/s]',
                  path_savefig='.'):
        """
        Plot boundary layer height with multiple data lines and shaded regions.

        :param data_lines: Array of data lines to plot (e.g., boundary layer heights).
        :param data_shading: 2D array of shaded data (e.g., vertical profiles).
        :param figsize: Size of the figure.
        :param cmap: Colormap for the shaded data.
        :param levels: Contour levels for color shading.
        :param label: Labels for the data lines.
        :param line_color: Colors for the data lines.
        :param title: Title for the plot.
        :param cb_title: Title for the colorbar.
        :param path_savefig: Path to save the figure.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Create a colormap plot for the shading data 
        Plot = ax.pcolormesh(self.time,self.zc[1:],data_shading,cmap=cmap,norm=colors.BoundaryNorm(levels,cmap.N))
        cb = plt.colorbar(Plot,orientation='vertical',extend='both',ticks=levels[::2])
        cb.ax.tick_params(length=8)
        cb.ax.set_title(cb_title,fontsize=self.fs-5)

        # Plot the data lines (e.g., boundary layer heights)
        for i in range(len(data_lines)):
            plt.plot(self.time,data_lines[i],'o',color=line_color[i],markersize=3,label=label[i])
        
        # Settings
        ax.set_xticks(self.time[::self.time_ticks_space],self.time_ticks)
        ax.set_xlim(0,self.time[-1])
        ax.set_ylim(0,self.zc[-1])
        ax.legend(loc='upper right',markerscale=4)
        ax.grid(':')
        ax.set_xlabel('LST')
        ax.set_ylabel('Height')
        ax.set_title(title,loc='left')
        ax.set_title(self.case_name+'\n'+self.region_name,loc='right',fontsize=self.fs-2)
        plt.savefig(path_savefig+'/BL_height_'+self.case_name+'_'+self.region_name,dpi=300,bbox_inches="tight")
        plt.close()
