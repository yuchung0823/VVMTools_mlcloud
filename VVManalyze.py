from vvmtools import VVMTools
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors

class newVVMTools(VVMTools):
    def __init__(self,case_path):
        super().__init__(case_path)
    
    def calc_TKE(self,t,func_config):
        u = np.squeeze(self.get_var('u', t, numpy=True,domain_range=func_config['domain_range']))
        v = np.squeeze(self.get_var('v', t, numpy=True,domain_range=func_config['domain_range']))
        w = np.squeeze(self.get_var('w', t, numpy=True,domain_range=func_config['domain_range'])) 

        u_regrid = (u[:, :, 1:] + u[:, :, :-1])[1:, 1:, :] / 2
        v_regrid = (v[:, 1:, :] + v[:, :-1, :])[1:, :, 1:] / 2
        w_regrid = (w[1:, :, :] + w[:-1, :, :])[:, 1:, 1:] / 2
        TKE = u_regrid**2 + v_regrid**2 + w_regrid**2

        return np.nanmean(TKE, axis=(1,2))
    
    def calc_enstrophy(self,t,func_config):
        xi = np.squeeze(self.get_var('xi', t, numpy=True),domain_range=func_config['domain_range'])
        eta = np.squeeze(self.get_var('eta', t, numpy=True),domain_range=func_config['domain_range'])
        if xi.shape ==  eta.shape:
            pass
        else:
            eta = np.squeeze(self.get_var('eta_2', t, numpy=True,domain_range=func_config['domain_range']))
        zeta = np.squeeze(self.get_var('zeta', t, numpy=True,domain_range=func_config['domain_range']))

        return np.nanmean((xi**2+eta**2+zeta**2), axis=(1,2))

    def calc_w_th(self,t,func_config):
        w = np.squeeze(self.get_var('w', t, numpy=True,domain_range=func_config['domain_range']))
        w_regrid = (w[1:]+w[:-1])/2
        w_mean = np.mean(w_regrid,axis=(1,2)).reshape(w_regrid.shape[0],1,1)
        w_prime = w_regrid - w_mean
        th = np.squeeze(self.get_var('th', t, numpy=True,domain_range=func_config['domain_range']))[1:]
        th_mean = np.mean(th,axis=(1,2)).reshape(th.shape[0],1,1)
        th_prime = th - th_mean
        w_th = np.mean(w_prime*th_prime,axis=(1,2))
        return w_th
    
    def h_BL_dthdz(self, time_steps=list(range(0, 720, 1)), domain_range=(None, None, None, None, None, None)):
        zc = self.get_var("zc", 0).to_numpy()
        th_mean = self.get_var_parallel("th", time_steps=time_steps, domain_range=domain_range, compute_mean=True, axis=(1, 2))
        
        dth_dz = (th_mean[:, 1:] - th_mean[:, :-1]) / (zc[1:] - zc[:-1])
        ans = []
        for t in range(len(dth_dz)):
            ans.append(zc[np.argmax(dth_dz[t])])
        return np.array(ans)

    def h_BL_th_plus05(self, time_steps=list(range(0, 720, 1)), domain_range=(None, None, None, None, None, None)):
        zc = self.get_var("zc", 0).to_numpy()
        th_mean = self.get_var_parallel("th", time_steps=time_steps, domain_range=domain_range, compute_mean=True, axis=(1, 2))
        th_find = th_mean - (th_mean[:, 0].reshape(th_mean.shape[0], 1) + 0.5)

        ans = []
        for t in range(len(th_find)):
            ans.append(zc[np.where(th_find[t] < 0.01)[0][-1]])
        return np.array(ans)

    def find_BL_boundary(self, var, threshold):
        zc = self.get_var("zc", 0).to_numpy()
        positive_mask = np.swapaxes(var, 0, 1) > threshold
        index = np.argwhere(positive_mask)
        
        k, i = index[:, 0], index[:, 1]
        i_k_map = {}
        
        for ii in range(len(i)):
            if i[ii] not in i_k_map:
                i_k_map[i[ii]] = k[ii]
            else:
                if k[ii] > i_k_map[i[ii]]:
                    i_k_map[i[ii]] = k[ii]
        
        h = np.zeros(720)
        for key, value in i_k_map.items():
            h[key] = zc[value+1]

        return np.array(h)
    
    def find_wth_boundary(self, var):
        mask = np.where(var>=0,1,-1)
        zc = self.DIM['zc']
        zc_wth = np.zeros((var.shape[0],3))
        for t in range(var.shape[0]):
            temp = mask[t,1:] - mask[t,:-1]

            if np.max(var[t])<1e-3:
                k_lower = k_mid = k_upper = 0

            else:
                try:
                    k_lower = np.argwhere(temp==-2)[0][0]
                except: 
                    k_lower = 0
                
                k_mid = np.argmin(var[t])
                
                try:
                    k_upper = np.argwhere(temp==2)[0][0]
                except:
                    k_upper = 0

            if np.max(var[t, np.argmin(var[t]):]) < 1e-3:
                k_upper = 0
             

            zc_wth[t,0], zc_wth[t,1], zc_wth[t,2] = zc[k_lower], zc[k_mid], zc[k_upper]
        
        return zc_wth


'''
path = lambda case:'/data/chung0823/VVM_cloud_dynamics_2024/DATA/%s/'%(case)
path_Aaron = lambda case:'/data/mlcloud/Aaron/VVM/DATA/%s/'%(case)

case_list = ['pbl_hetero_dthdz_8','pbl_hetero_dthdz_11']
case_list_Aaron = ['OceanGrass_S1','OceanGrass_S1_landmid','OceanGrass_S2','OceanGrass_S2_landmid']

case = case_list[0]
time = np.arange(720)

myTool = newVVMTools(path(case))
func_config={"domain_range":(None,None,None,None,64,128)}

w_th = myTool.func_time_parallel(func=myTool.calc_w_th,time_steps=time,func_config=func_config,cores=20)
zc_wth = myTool.find_wth_boundary(w_th)

z = (myTool.DIM['zc'])[1:]
P = plt.pcolormesh(time,z,w_th.T,vmin=-0.04,vmax=0.04,cmap=cm.RdBu_r)
plt.plot(time,zc_wth[:,0],'o',color='k',markersize=3,label='lower')
plt.plot(time,zc_wth[:,1],'o',color='purple',markersize=3,label='mid')
plt.plot(time,zc_wth[:,2],'o',color='green',markersize=3,label='upper')
plt.colorbar(P)
plt.legend(fontsize=15)
plt.savefig('myTool.png')
plt.show()
'''
