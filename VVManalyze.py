from vvmtools import VVMTools
import numpy as np

class VVMTools_BL(VVMTools):
    """
    A subclass of VVMTools to provide additional methods specific to 
    boundary layer calculations such as TKE, enstrophy, and boundary 
    layer height.
    """
    def __init__(self,case_path):
        """
        A subclass of VVMTools to provide additional methods specific to 
        boundary layer calculations such as TKE, enstrophy, and boundary 
        layer height.
        """
        super().__init__(case_path)
    
    def calc_TKE(self, time_steps, func_config):
        """
        Calculate the Turbulent Kinetic Energy (TKE) using the velocity 
        components (u, v, w).
        
        :param time_steps: List of time steps to compute TKE.
        :param func_config: Configuration dictionary that includes domain range.
        :return: Mean TKE over the domain at each time step.
        """
        # Get velocity components (u, v, w) for the specified time steps
        u = np.squeeze(self.get_var('u', time_steps, numpy=True, domain_range=func_config['domain_range']))
        v = np.squeeze(self.get_var('v', time_steps, numpy=True, domain_range=func_config['domain_range']))
        w = np.squeeze(self.get_var('w', time_steps, numpy=True, domain_range=func_config['domain_range'])) 
        
        # Regrid velocities to calculate TKE at cell centers
        u_regrid = (u[:, :, 1:] + u[:, :, :-1])[1:, 1:, :] / 2
        v_regrid = (v[:, 1:, :] + v[:, :-1, :])[1:, :, 1:] / 2
        w_regrid = (w[1:, :, :] + w[:-1, :, :])[:, 1:, 1:] / 2

        # Calculate TKE = 0.5 * (u^2 + v^2 + w^2)
        TKE = u_regrid**2 + v_regrid**2 + w_regrid**2
        return np.nanmean(TKE, axis=(1,2))
    
    def calc_Enstrophy(self, time_steps, func_config):
        """
        Calculate enstrophy using the vorticity components (xi, eta, zeta).
        
        :param time_steps: List of time steps to compute enstrophy.
        :param func_config: Configuration dictionary with domain range.
        :return: Mean enstrophy over the domain at each time step.
        """
        # Get vorticity components (xi, eta, zeta)
        xi = np.squeeze(self.get_var('xi', time_steps, numpy=True, domain_range=func_config['domain_range']))
        eta = np.squeeze(self.get_var('eta', time_steps, numpy=True, domain_range=func_config['domain_range']))
        
        # Check if eta needs to be loaded from a different variable (eta_2)
        if xi.shape ==  eta.shape:
            pass
        else:
            eta = np.squeeze(self.get_var('eta_2', time_steps, numpy=True, domain_range=func_config['domain_range']))
        zeta = np.squeeze(self.get_var('zeta', time_steps, numpy=True, domain_range=func_config['domain_range']))

        # Calculate and return mean enstrophy (xi^2 + eta^2 + zeta^2)
        return np.nanmean((xi**2+eta**2+zeta**2), axis=(1,2))

    def calc_w_th(self, time_steps, func_config):
        """
        Calculate the covariance of vertical velocity (w) and potential temperature (theta).
        
        :param time_steps: List of time steps to compute w'θ'.
        :param func_config: Configuration dictionary with domain range.
        :return: Mean w'θ' over the domain at each time step.
        """
        # Get vertical velocity and regrid to center points
        w = np.squeeze(self.get_var('w', time_steps, numpy=True, domain_range=func_config['domain_range']))
        w_regrid = (w[1:]+w[:-1])/2

        # Calculate w' (perturbation of w)
        w_mean = np.squeeze(self.get_var('w', time_steps, numpy=True, domain_range=(None,None,None,None,None,None)))
        w_mean = (w_mean[1:]+w_mean[:-1])/2
        w_mean = np.mean(w_mean,axis=(1,2)).reshape(w_mean.shape[0],1,1)
        w_prime = w_regrid - w_mean

        # Get potential temperature (theta) and calculate θ' (theta perturbation)
        th = np.squeeze(self.get_var('th', time_steps, numpy=True, domain_range=func_config['domain_range']))[1:]
        th_mean = np.squeeze(self.get_var('th', time_steps, numpy=True, domain_range=(None,None,None,None,None,None)))[1:]
        th_mean = np.mean(th_mean,axis=(1,2)).reshape(th_mean.shape[0],1,1)
        th_prime = th - th_mean

        # Calculate the covariance w'θ' and return the mean over the domain
        w_th = np.mean(w_prime*th_prime,axis=(1,2))
        return w_th
    

    def find_BL_boundary(self, var, howToSearch, threshold=0.01):
        """
        Calculate the boundary layer height based on specified search criteria.

        :param var: Array of the variable used to define the boundary layer (e.g., potential temperature θ).
        :param howToSearch: String defining the boundary search method. Options are:
                            - "th_plus05K": Height where θ exceeds surface value by 0.5K.
                            - "dthdz": Height where the vertical gradient of θ is maximum.
                            - "threshold": Height where `var` first exceeds the threshold value.
                            - "wth": Boundary levels based on transitions in vertical heat flux.
        :param threshold: Threshold value used in the "threshold" and "wth" methods (default is 0.01).
        :return: Array of boundary layer heights for each time step or, for "wth", an array with lower, mid, and upper boundaries.
        """
        # Get height levels (zc) in kilometers
        zc = (self.get_var("zc", 0).to_numpy()/1000)

        if howToSearch == "th_plus05K":
            # Find the height where theta exceeds surface value by 0.5K
            th_find = var - (var[:, 0].reshape(var.shape[0], 1) + 0.5)

            # Identify boundary layer height for each time step
            h_BL_th_plus05 = []
            for t in range(len(th_find)):
                h_BL_th_plus05.append(zc[np.where(th_find[t] < threshold)[0][-1]])            
            return np.array(h_BL_th_plus05)

        elif howToSearch == "dthdz":
            # Compute vertical gradient of theta (dθ/dz)
            dth_dz = (var[:, 1:] - var[:, :-1]) / (zc[1:] - zc[:-1])
            
            # Determine height of maximum gradient for each time step
            zc = zc[1:]
            h_BL_dthdz = []
            for t in range(len(dth_dz)):
                h_BL_dthdz.append(zc[np.argmax(dth_dz[t])])            
            return np.array(h_BL_dthdz)

        elif howToSearch == "threshold":
            # Identify height where 'var' crosses the threshold
            positive_mask = np.swapaxes(var, 0, 1) > threshold
            index = np.argwhere(positive_mask)
            
            # Map each time step to the highest index where the threshold is crossed
            k, i = index[:, 0], index[:, 1]
            i_k_map = {}
            
            for ii in range(len(i)):
                if i[ii] not in i_k_map:
                    i_k_map[i[ii]] = k[ii]
                else:
                    if k[ii] > i_k_map[i[ii]]:
                        i_k_map[i[ii]] = k[ii]
            
            # Compute boundary layer height for each time step based on threshold crossings
            zc = zc[1:]
            h = np.zeros(len(var))
            for key, value in i_k_map.items():
                h[key] = zc[value]
            return np.array(h)

        elif howToSearch == "wth":
            # Identify lower, middle, and upper boundaries based on vertical heat flux transitions
            mask = np.where(var>=0,1,-1)
            zc_wth = np.zeros((3,var.shape[0]))
            zc = zc[1:]
            for t in range(var.shape[0]):
                temp = mask[t,1:] - mask[t,:-1]

                # Default zero boundary if w'θ' values are low
                if np.max(var[t])<threshold:
                    k_lower = k_mid = k_upper = 0
                
                # Find lower, mid, and upper boundaries based on transitions
                else:
                    # Find lower boundary
                    try:
                        k_lower = np.argwhere(temp==-2)[0][0] + 1
                    except: 
                        k_lower = 0
                    
                    # Find mid boundary
                    k_mid = np.argmin(var[t]) + 1
                    
                    # Find upper boundary
                    try:
                        k_upper = np.argwhere(temp==2)[0][0] + 1
                    except:
                        k_upper = 0
                
                # Zero out upper boundary if max value beyond mid-boundary is low
                if np.max(var[t, np.argmin(var[t]):]) < threshold:
                    k_upper = 0

                # Store the boundaries for this time step
                zc_wth[0,t], zc_wth[1,t], zc_wth[2,t] = zc[k_lower], zc[k_mid], zc[k_upper]            
            return zc_wth

        else:
            print("Without this searching approach")
    