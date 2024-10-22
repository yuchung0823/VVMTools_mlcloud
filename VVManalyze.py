from vvmtools import VVMTools
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors

class newVVMTools(VVMTools):
    def __init__(self,case_path):
        super().__init__(case_path)
    
    def calc_TKE(self, time_steps, func_config):
        u = np.squeeze(self.get_var('u', time_steps, numpy=True,domain_range=func_config['domain_range']))
        v = np.squeeze(self.get_var('v', time_steps, numpy=True,domain_range=func_config['domain_range']))
        w = np.squeeze(self.get_var('w', time_steps, numpy=True,domain_range=func_config['domain_range'])) 

        u_regrid = (u[:, :, 1:] + u[:, :, :-1])[1:, 1:, :] / 2
        v_regrid = (v[:, 1:, :] + v[:, :-1, :])[1:, :, 1:] / 2
        w_regrid = (w[1:, :, :] + w[:-1, :, :])[:, 1:, 1:] / 2
        TKE = u_regrid**2 + v_regrid**2 + w_regrid**2

        return np.nanmean(TKE, axis=(1,2))
    
    def calc_Enstrophy(self, time_steps, func_config):
        xi = np.squeeze(self.get_var('xi', time_steps, numpy=True,domain_range=func_config['domain_range']))
        eta = np.squeeze(self.get_var('eta', time_steps, numpy=True,domain_range=func_config['domain_range']))
        if xi.shape ==  eta.shape:
            pass
        else:
            eta = np.squeeze(self.get_var('eta_2', time_steps, numpy=True,domain_range=func_config['domain_range']))
        zeta = np.squeeze(self.get_var('zeta', time_steps, numpy=True,domain_range=func_config['domain_range']))

        return np.nanmean((xi**2+eta**2+zeta**2), axis=(1,2))

    def calc_w_th(self, time_steps, func_config):
        w = np.squeeze(self.get_var('w', time_steps, numpy=True,domain_range=func_config['domain_range']))
        w_regrid = (w[1:]+w[:-1])/2
        w_mean = np.mean(w_regrid,axis=(1,2)).reshape(w_regrid.shape[0],1,1)
        w_prime = w_regrid - w_mean
        th = np.squeeze(self.get_var('th', time_steps, numpy=True,domain_range=func_config['domain_range']))[1:]
        th_mean = np.mean(th,axis=(1,2)).reshape(th.shape[0],1,1)
        th_prime = th - th_mean
        w_th = np.mean(w_prime*th_prime,axis=(1,2))
        return w_th
    
    def h_BL_dthdz(self, time_steps, func_config):
        zc = self.get_var("zc", 0).to_numpy()
        th_mean = self.get_var_parallel("th", time_steps=time_steps, domain_range=func_config['domain_range'], compute_mean=True, axis=(1, 2))
        
        dth_dz = (th_mean[:, 1:] - th_mean[:, :-1]) / (zc[1:] - zc[:-1])
        ans = []
        for t in range(len(dth_dz)):
            ans.append(zc[np.argmax(dth_dz[t])])
        return np.array(ans)

    def h_BL_th_plus05(self, time_steps, func_config):
        zc = self.get_var("zc", 0).to_numpy()
        th_mean = self.get_var_parallel("th", time_steps=time_steps, domain_range=func_config['domain_range'], compute_mean=True, axis=(1, 2))
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
        zc_wth = np.zeros((3,var.shape[0]))
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
             

            zc_wth[0,t], zc_wth[1,t], zc_wth[2,t] = zc[k_lower], zc[k_mid], zc[k_upper]
        0
        return zc_wth
