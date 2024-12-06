import numpy as np
from vvmtools.analyze import DataRetriever
from vvmtools.plot import DataPlotter
import matplotlib.pyplot as plt

# prepare expname and data coordinate
expname  = 'S1'
nx = 128; x = np.arange(nx)*0.2
ny = 128; y = np.arange(ny)*0.2
nz = 50;  z = np.arange(nz)*0.04
nt = 721; t = np.arange(nt)*np.timedelta64(2,'m')+np.datetime64('2024-01-01 05:00:00')

# create dataPlotter class
figpath           = './fig/'
data_domain       = {'x':x, 'y':y, 'z':z, 't':t}
data_domain_units = {'x':'km', 'y':'km', 'z':'km', 't':'LocalTime'}
dplot = DataPlotter(expname, figpath, data_domain, data_domain_units)


path = '/data/chung0823/data_VVM/VVM_Data/OceanGrass_S1_traffic'
data = DataRetriever(path)
NO = data.get_var_parallel(var='NO',time_steps=np.arange(721),domain_range=(0,1,None,None,None,None),compute_mean=True,axis=(0))
NO2 = data.get_var_parallel(var='NO2',time_steps=np.arange(721),domain_range=(0,1,None,None,None,None),compute_mean=True,axis=(0))
u = data.get_var_parallel(var='u',time_steps=np.arange(721),domain_range=(0,1,None,None,None,None),compute_mean=True,axis=(0))

NO_mean = np.mean(NO, axis=1).reshape(nt,1)
NO2_mean = np.mean(NO2, axis=1).reshape(nt,1)
# data_xt2d  = (NO-NO_mean) + (NO2-NO2_mean)
data_xt2d  = u

fig, ax, cax = dplot.draw_xt(data = data_xt2d,
                                levels = np.arange(-3.,3.001,0.2),
                                extend = 'both',
                                cmap_name = 'RdBu_r',
                                title_left  = 'u [m/s]',
                                title_right = f'Ocean-Grass',
                                figname     = 'hov_u.png',
                               )
plt.show()
