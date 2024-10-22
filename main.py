from VVManalyze import newVVMTools
from plottools import VVMPlot
import numpy as np

path = lambda case:'/data/chung0823/VVM_cloud_dynamics_2024/DATA/%s/'%(case)
path_Aaron = lambda case:'/data/mlcloud/Aaron/VVM/DATA/%s/'%(case)

case_list = ['pbl_hetero_dthdz_8','pbl_hetero_dthdz_11']
case_list_Aaron = ['OceanGrass_S1_new','OceanGrass_S1_landmid_new','OceanGrass_S2_new_new','OceanGrass_S2_landmid_newnew']

time = np.arange(720)
func_config={"domain_range":(0,1,None,None,None,None)}
Plot = VVMPlot()

for i in range(6):

    if i<2:
        case = case_list[i]
        myTool = newVVMTools(path(case))

    else:
        case = case_list_Aaron[i-2]
        myTool = newVVMTools(path_Aaron(case))
    
    ####
    NO = myTool.get_var_parallel('NO',time_steps=time,domain_range=func_config['domain_range'], compute_mean=True, axis=(0),cores=20)
    NO2 = myTool.get_var_parallel('NO2',time_steps=time,domain_range=func_config['domain_range'], compute_mean=True, axis=(0),cores=20)
    NOx = NO + NO2

    Plot.hovmoller(data=NOx,case_name=case)


func_configs=[{"domain_range":(None,None,None,None,None,None)}, {"domain_range":(None,None,None,None,0,64)}, {"domain_range":(None,None,None,None,64,128)}]
region_lists=['domain_avg','ocean','grass']
for i in range(2):

    case = case_list[i]
    myTool = newVVMTools(path(case))

    for region_list, func_config in zip(region_lists, func_configs):

        TKE = myTool.func_time_parallel(func=myTool.calc_TKE,time_steps=time,func_config=func_config,cores=20)
        Enstrophy = myTool.func_time_parallel(func=myTool.calc_Enstrophy,time_steps=time,func_config=func_config,cores=20)
        w_th = myTool.func_time_parallel(func=myTool.calc_w_th,time_steps=time,func_config=func_config,cores=20)

        h_BL_th_plus05 = myTool.h_BL_th_plus05(time_steps=time,func_config=func_config)
        h_BL_dthdz = myTool.h_BL_dthdz(time_steps=time,func_config=func_config)
        h_BL_TKE = myTool.find_BL_boundary(TKE,threshold=0.08)
        h_BL_Enstrophy = myTool.find_BL_boundary(Enstrophy,threshold=1e-5)
        h_BL_wth = myTool.find_wth_boundary(w_th) # x3

        Plot.BL_height(
            zc=myTool.DIM['zc'][1:],
            data_lines=[h_BL_th_plus05, h_BL_dthdz, h_BL_TKE, h_BL_Enstrophy, h_BL_wth[0], h_BL_wth[1], h_BL_wth[2]],
            data_shading=w_th.T,
            region=region_list,
            case_name=case
            )
