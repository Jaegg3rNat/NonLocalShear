import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
from matplotlib import rc
from matplotlib.patches import Rectangle
from scipy.special import j1 as Bj1
from random import sample
import matplotlib.ticker as tkr
from tqdm import tqdm

def carrying(mu,pe,w,file,v):
    base_folder = file
    velocity_field_name = v
    D = 1e-4
    comp_rad = 0.2
    r = mu * D / comp_rad ** 2

    # Open the HDF5 file and inspect contents
    file_path = f'{base_folder}/{velocity_field_name}_mu{mu:.2f}_Pe{pe:.1f}_w{w:.2f}/dat.h5'
    f = h5py.File(file_path, 'r')

    k = f['density'][:]/r
    x = f['time'][:]
    return x, k
Pe = []
dq02 = []
dq005 = []
mu = 450
for i in range(36):
    Pe.append(i)
for pe in Pe:
    _,k =carrying(mu,pe,0,'../Data/q_0.2Data','rankine')
    _,k1 =carrying(mu,pe,0,'../Data/q_0.05Data','rankine')
    dq02.append(np.mean(k))
    dq005.append(np.mean(k1))

plt.plot(Pe,dq02,'.-', label = 'q0.2')
plt.plot(Pe,dq005,'.-', label = 'q0.05')
plt.xlabel('Pe')
plt.ylabel(' Time Average of Normalize Population Density ')
plt.title(f'Da={mu}')
plt.legend()
plt.show()