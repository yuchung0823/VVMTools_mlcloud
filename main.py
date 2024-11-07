import numpy as np
from VVManalyze import VVMTools_BL
from vvmtools.plot import DataPlotter
import matplotlib.pyplot as plt

# prepare expname and data coordinate
nx = 128; x = np.arange(nx)*0.2
ny = 128; y = np.arange(ny)*0.2
nz = 50;  z = np.arange(nz)*0.04
nt = 721; t = np.arange(nt)*np.timedelta64(2,'m')+np.datetime64('2024-01-01 05:00:00')
time = np.arange(nt)

# Define the path to the simulation data for different cases
path = lambda case:'/data/chung0823/data_VVM/VVM_Data/%s/'%(case)

# List of simulation cases and their corresponding names
case_list = ['pbl_hetero_dthdz_8','pbl_hetero_dthdz_11','OceanGrass_S1_new','OceanGrass_S1_landmid_new','OceanGrass_S2_new_new','OceanGrass_S2_landmid_newnew']
case_names = ['S1_Coastal','S2_Coastal','S1_Ocean','S1_Grass','S2_Ocean','S2_Grass']

# Domain configurations for different regions (whole domain, ocean, grass)
func_configs=[{"domain_range":(None,None,None,None,None,None)}, # Full domain
              {"domain_range":(None,None,None,None,0,64)},      # Ocean region
              {"domain_range":(None,None,None,None,64,128)}]    # Grass region
region_lists=['domain','ocean','grass'] # Names of the regions for boundary layer analysis
case_name = ['S1','S2']  # List of simulation cases corresponding names

# Loop through the first two cases to analyze BL height in different regions
for i in range(2):
    case = case_list[i]
    myTool = VVMTools_BL(path(case))

    # Loop through each region (domain, ocean, grass)
    for region, func_config in zip(region_lists, func_configs):
       
        expname = case_name[i]
        
        # Compute theta, TKE (Turbulent Kinetic Energy), Enstrophy, and vertical heat flux (w'theta')
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

        # create dataPlotter class
        figpath           = './fig/'
        data_domain       = {'x':x, 'y':y, 'z':z, 't':t}
        data_domain_units = {'x':'km', 'y':'km', 'z':'km', 't':'LocalTime'}
        dplot = DataPlotter(expname, figpath, data_domain, data_domain_units)

        # draw z-t diagram
        # input data dimension is (nz, nt)
        # [output] figure, axis, colorbar axis
        w_th = np.hstack((np.full((nt,1),np.nan), w_th))
        fig, ax, cax = dplot.draw_zt(data = w_th.T, \
                                    levels = np.arange(-0.04,0.041,0.005), \
                                    extend = 'both', \
                                    pblh_dicts={r'$\theta$ + 0.5 K': h_BL_th_plus05,\
                                                r'max d$\theta$/dz': h_BL_dthdz,\
                                                'TKE': h_BL_TKE,\
                                                'Enstrophy': h_BL_Enstrophy ,\
                                                r"top$ (\overline{w'\theta'}+)$":h_BL_wth[0] ,\
                                                r"min$ (\overline{w'\theta'})$":h_BL_wth[1] ,\
                                                r"top$ (\overline{w'\theta'}-)$":h_BL_wth[2] ,\
                                                },\
                                    title_left  = r'Vertical $\theta$ transport', \
                                    title_right = region, \
                                    figname     = 'BL_height_'+expname+'_'+region,\
                            )

        #plt.show()

