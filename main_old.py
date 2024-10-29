from VVManalyze import VVMTools_BL
from plottools import VVMPlot_BL
import numpy as np

# Define the path to the simulation data for different cases
path = lambda case:'/home/chung0823/cloud2024/VVM_Data/%s/'%(case)

# List of simulation cases and their corresponding names
case_list = ['pbl_hetero_dthdz_8','pbl_hetero_dthdz_11','OceanGrass_S1_new','OceanGrass_S1_landmid_new','OceanGrass_S2_new_new','OceanGrass_S2_landmid_newnew']
case_names = ['S1_Coastal','S2_Coastal','S1_Ocean','S1_Grass','S2_Ocean','S2_Grass']

# Time range for the analysis (721 time steps)
time = np.arange(721)

############################ Hovmoller diagram ############################

# Configuration for domain range, used to subset the data for analysis
func_config={"domain_range":(0,1,None,None,None,None)}

# Loop through the first 6 cases to generate Hovmöller diagrams
for i in range(6):
    case = case_list[i]
    case_name = case_names[i]

    # Initialize the analysis and plotting tools
    myTool = VVMTools_BL(path(case))
    Plot = VVMPlot_BL(path(case),case_name=case_name,region_name='y-axis avg')
   
    # Retrieve NO and NO2 data and calculate their sum (NOx)
    NO = myTool.get_var_parallel('NO',time_steps=time,domain_range=func_config['domain_range'], compute_mean=True, axis=(0),cores=20)
    NO2 = myTool.get_var_parallel('NO2',time_steps=time,domain_range=func_config['domain_range'], compute_mean=True, axis=(0),cores=20)
    NOx = NO + NO2
   
    # Plot Hovmöller diagram of NOx data
    Plot.hovmoller(data=NOx)

############################ BL height ############################
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
        
        # Compute TKE (Turbulent Kinetic Energy), Enstrophy, and vertical heat flux (w'theta')
        TKE = myTool.func_time_parallel(func=myTool.calc_TKE,time_steps=time,func_config=func_config,cores=20)
        Enstrophy = myTool.func_time_parallel(func=myTool.calc_Enstrophy,time_steps=time,func_config=func_config,cores=20)
        w_th = myTool.func_time_parallel(func=myTool.calc_w_th,time_steps=time,func_config=func_config,cores=20)

        # Calculate boundary layer heights based on different criteria
        h_BL_th_plus05 = myTool.h_BL_th_plus05(time_steps=time,func_config=func_config)
        h_BL_dthdz = myTool.h_BL_dthdz(time_steps=time,func_config=func_config)
        h_BL_TKE = myTool.find_BL_boundary(TKE,threshold=0.08)
        h_BL_Enstrophy = myTool.find_BL_boundary(Enstrophy,threshold=1e-5)
        h_BL_wth = myTool.find_wth_boundary(w_th)

        # Initialize the plotting tool for the specific case and region
        Plot = VVMPlot_BL(path(case),case_name=case_name[i],region_name=region)

        # Plot the BL height with multiple criteria (lines) and shaded w'theta' data
        Plot.BL_height(
            data_lines=[h_BL_th_plus05, h_BL_dthdz, h_BL_TKE, h_BL_Enstrophy, h_BL_wth[0], h_BL_wth[1], h_BL_wth[2]],
            data_shading=w_th.T,
            )
