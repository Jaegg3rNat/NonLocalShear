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
# Overall Details
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
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

# Define the two values of mu you want to compare
mu_values = [280, 450]  # Example: change 350 to another value if desired

Pe = [5 + i for i in range(30)]
Pe.append(0)
Pe.sort()

fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, mu in zip(axs, mu_values):
    dq02 = []
    dq005 = []

    for pe in Pe:
        _, k = carrying(mu, pe, 0, '../Data/q0.2', 'rankine')
        _, k1 = carrying(mu, pe, 0, '../Data/q0.05', 'rankine')
        dq02.append(np.mean(k))
        dq005.append(np.mean(k1))

    ax.plot(Pe, dq02, '.-', label='a=0.2')
    ax.plot(Pe, dq005, '.-', label='a=0.05')
    ax.set_xlabel(r'Characteristic Péclet, $\mbox{Pe}$', fontsize=15)
    ax.set_title(f'Da = {mu}',fontsize = 16)
    ax.legend(fontsize = 12)

axs[0].set_ylabel('Time-averaged \n normalized population abundance', fontsize=15)
plt.tight_layout()
plt.show()