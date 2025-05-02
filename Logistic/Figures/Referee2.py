import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
import re
from matplotlib import rc
from matplotlib.patches import Rectangle
from scipy.special import j1 as Bj1
from random import sample
import matplotlib.ticker as tkr
from tqdm import tqdm
### HDF5 File Inspection
def print_name(name, obj):
    """Function to print the name of groups and datasets in an HDF5 file."""
    if isinstance(obj, h5py.Group):
        print(f"Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}, Shape: {obj.shape}, Dtype: {obj.dtype}")


def print_hdf5_contents(file_path):
    """Function to open an HDF5 file and print its contents."""
    with h5py.File(file_path, 'r') as f:
        f.visititems(print_name)


def time_hdf5(file_path):
    """Function to extract time values from HDF5 file."""
    with h5py.File(file_path, 'r') as f:
        t_values = np.array([
            float(re.search(r"t(\d+(\.\d+)?)", name).group(1))
            for name in f.keys() if re.match(r"t\d+(\.\d+)?$", name)
        ])
        t_values.sort()
    return t_values

def carrying(mu,pe,w,file,v):
    base_folder = file
    velocity_field_name = v
    D = 1e-4
    comp_rad = 0.2
    r = mu * D / comp_rad ** 2

    # Open the HDF5 file and inspect contents
    file_path = f'{base_folder}/{velocity_field_name}_mu{mu:.2f}_Pe{pe:.1f}_w{w:.2f}/dat.h5'
    f = h5py.File(file_path, 'r')
    # print_hdf5_contents(file_path)

    k = f['density'][-6000:]/r
    x = f['time'][:]
    return x, k


# Pe1 = [0,1,2,3,4,5,6,8,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
# Pe2 = [0,1,2,3,4,5,6,8,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
mu = 220

Pe1 = [0,1,2,3,4,5,6,8,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
Pe2 = [0,1,2,3,4,5,6,8,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]

# file_path = '../Data/Exp/128_R0.2/sinusoidal_mu510.00_Pe0.0_w6.28/dat.h5'
# print_hdf5_contents(file_path)

dq02 = []
dq005 = []

for pe in Pe1:
    _, k = carrying(mu, pe, 6.28, '../Data', 'sinusoidal')

    dq02.append(np.mean(k))

for pe in Pe2:
    _, k1 = carrying(mu, pe, 0, '../Data', 'rankine')
    dq005.append(np.mean(k1))
# print(dq02)
plt.plot(Pe1, dq02, '.-', label='Sinusoidal')
plt.plot(Pe2, dq005, '.-', label='Rankine')
plt.axhline(y=1, color='k', linestyle='--', lw=1,label = 'Homogeneous')
plt.xlabel(r'$\mathrm{Pe}$',fontsize = 12)
plt.ylabel(r'Time Averaged Density',fontsize = 12)
plt.title(r'$\mathrm{Da}= 220$, Top Hat kernel',fontsize = 12)

plt.legend()
plt.show()
