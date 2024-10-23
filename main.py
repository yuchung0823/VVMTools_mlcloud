from VVManalyze import VVMTools_BL
from plottools import VVMPlot_BL
import numpy as np

path = lambda case:'/data/chung0823/VVM_cloud_dynamics_2024/DATA/%s/'%(case)
case_list = ['pbl_hetero_dthdz_8','pbl_hetero_dthdz_11']
case_names = ['S1_Coastal','S2_Coastal']

path_Aaron = lambda case:'/data/mlcloud/Aaron/VVM/DATA/%s/'%(case)
case_list_Aaron = ['OceanGrass_S1_new','OceanGrass_S1_landmid_new','OceanGrass_S2_new_new','OceanGrass_S2_landmid_newnew']
case_names_Aaron = ['S1_Ocean','S1_Grass','S2_Ocean','S2_Grass']

time = np.arange(721)

############################ Hovmoller diagram ############################
func_config={"domain_range":(0,1,None,None,None,None)}
for i in range(6):

    if i<2:
        case = case_list[i]
        case_name = case_names[i]
        myTool = VVMTools_BL(path(case))
        Plot = VVMPlot_BL(path(case),case_name=case_name,region_name='y-axis avg')

    else:
        case = case_list_Aaron[i-2]
        case_name = case_names_Aaron[i-2]
        myTool = VVMTools_BL(path_Aaron(case))
        Plot = VVMPlot_BL(path_Aaron(case),case_name=case_name,region_name='y-axis avg')
    
    
    NO = myTool.get_var_parallel('NO',time_steps=time,domain_range=func_config['domain_range'], compute_mean=True, axis=(0),cores=20)
    NO2 = myTool.get_var_parallel('NO2',time_steps=time,domain_range=func_config['domain_range'], compute_mean=True, axis=(0),cores=20)
    NOx = NO + NO2
   
    Plot.hovmoller(data=NOx)

############################ BL height ############################
func_configs=[{"domain_range":(None,None,None,None,None,None)}, {"domain_range":(None,None,None,None,0,64)}, {"domain_range":(None,None,None,None,64,128)}]
region_lists=['domain','ocean','grass']
case_name = ['S1','S2']
for i in range(2):

    case = case_list[i]
    myTool = VVMTools_BL(path(case))

    for region, func_config in zip(region_lists, func_configs):

        TKE = myTool.func_time_parallel(func=myTool.calc_TKE,time_steps=time,func_config=func_config,cores=20)
        Enstrophy = myTool.func_time_parallel(func=myTool.calc_Enstrophy,time_steps=time,func_config=func_config,cores=20)
        w_th = myTool.func_time_parallel(func=myTool.calc_w_th,time_steps=time,func_config=func_config,cores=20)

        h_BL_th_plus05 = myTool.h_BL_th_plus05(time_steps=time,func_config=func_config)
        h_BL_dthdz = myTool.h_BL_dthdz(time_steps=time,func_config=func_config)
        h_BL_TKE = myTool.find_BL_boundary(TKE,threshold=0.08)
        h_BL_Enstrophy = myTool.find_BL_boundary(Enstrophy,threshold=1e-5)
        h_BL_wth = myTool.find_wth_boundary(w_th)

        Plot = VVMPlot_BL(path(case),case_name=case_name[i],region_name=region)
        Plot.BL_height(
            data_lines=[h_BL_th_plus05, h_BL_dthdz, h_BL_TKE, h_BL_Enstrophy, h_BL_wth[0], h_BL_wth[1], h_BL_wth[2]],
            data_shading=w_th.T,
            )
