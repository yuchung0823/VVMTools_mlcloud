import numpy as np
from vvmtools.analyze import DataRetriever
from vvmtools.plot import DataPlotter
from VVManalyze import VVMTools_BL
import matplotlib.pyplot as plt

# prepare expname and data coordinate
expname  = 'S1'
nx = 128; x = np.arange(nx)*0.2
ny = 128; y = np.arange(ny)*0.2
nz = 50;  z = np.arange(nz)*0.04
nt = 721; t = np.arange(nt)*np.timedelta64(2,'m')+np.datetime64('2024-01-01 05:00:00')
time = np.arange(721)

# create dataPlotter class
figpath           = './fig/'
data_domain       = {'x':x, 'y':y, 'z':z, 't':t}
data_domain_units = {'x':'km', 'y':'km', 'z':'km', 't':'LocalTime'}
dplot = DataPlotter(expname, figpath, data_domain, data_domain_units)

# TODO: Use your tracer data, noted that you need to normalize the data by its maximum
# Normalize: data/data.max()
path = '/data/chung0823/data_VVM/VVM_Data/OceanGrass_S1_traffic'
myTool = VVMTools_BL(path)

# Compute theta, TKE (Turbulent Kinetic Energy), Enstrophy, and vertical heat flux (w'theta')
# func_config={"domain_range":(None,None,None,None,None,None)}
func_configs=[{"domain_range":(None,None,None,None,None,None)}, # Full domain
              {"domain_range":(None,None,None,None,0,64)},      # Ocean region
              {"domain_range":(None,None,None,None,64,128)}]    # Grass region
region_lists=['domain','ocean','grass'] # Names of the regions for boundary layer analysis

trs = ['tr01','tr02','tr03']
heights = ['sfc','750m','1500m']
for i in range(3):
    for region, func_config in zip(region_lists, func_configs):
        data_zt2d  = myTool.get_var_parallel(var=trs[i],time_steps=np.arange(721),domain_range=func_config['domain_range'],compute_mean=True,axis=(1,2),cores=20)
        data_zt2d /= np.max(data_zt2d)

        th = myTool.get_var_parallel('th',time_steps=time,domain_range=func_config["domain_range"],compute_mean=True,axis=(1,2),cores=20)
        TKE = myTool.func_time_parallel(func=myTool.calc_TKE,time_steps=time,func_config=func_config,cores=20)
        Enstrophy = myTool.func_time_parallel(func=myTool.calc_Enstrophy,time_steps=time,func_config=func_config,cores=20)
        w_th = myTool.func_time_parallel(func=myTool.calc_w_th,time_steps=time,func_config=func_config,cores=20)

        # Calculate boundary layer heights based on different criteria
        h_BL_th_plus05 = myTool.find_BL_boundary(th,howToSearch='th_plus05K')
        h_BL_dthdz = myTool.find_BL_boundary(th,howToSearch='dthdz')
        h_BL_TKE = myTool.find_BL_boundary(TKE,howToSearch='threshold',threshold=0.08)
        h_BL_Enstrophy = myTool.find_BL_boundary(Enstrophy,howToSearch='threshold',threshold=1e-5)
        h_BL_wth = myTool.find_BL_boundary(w_th,howToSearch='wth',threshold=1e-3)

        fig, ax, cax = dplot.draw_zt(data = data_zt2d.T,
                                    levels = np.arange(0,1.1,0.1),
                                    extend = 'neither',
                                    pblh_dicts={r'$\theta$ + 0.5 K': h_BL_th_plus05,\
                                                r'max d$\theta$/dz': h_BL_dthdz,\
                                                'TKE': h_BL_TKE,\
                                                'Enstrophy': h_BL_Enstrophy ,\
                                                r"top$ (\overline{w'\theta'}+)$":h_BL_wth[0] ,\
                                                r"min$ (\overline{w'\theta'})$":h_BL_wth[1] ,\
                                                r"top$ (\overline{w'\theta'}-)$":h_BL_wth[2] ,\
                                                },\
                                    title_left  = f'Tracer_{heights[i]} (Normalized by max)',
                                    title_right = f'{region}',
                                    cmap_name   = 'Greys',
                                    figname     = '',
                            )

        ###
        ### If you want to delete the legend, turn this block on
        ax.get_legend().remove()
        plt.savefig(f'{figpath}/pbl_{heights[i]}_{region}.png', dpi=200)
        plt.close('all')
        ###

        plt.show()