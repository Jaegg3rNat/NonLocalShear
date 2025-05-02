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
import os

from sympy import symbols, solve, exp, Eq
from matplotlib.colors import LinearSegmentedColormap


# = ==========================================
def rho(eps, mu, pe, lamb, flow):
    D = 1e-4
    base_folder = f"../Data/gaussian/lambda{lamb}/128_eps{eps:.3f}"
    if pe == 0:
        fl = 'sinusoidal'
    else:
        fl = flow

    # Open the HDF5 file and inspect contents
    file_path = f'{base_folder}/{fl}_Pe{pe:.1f}_mu{mu:.5f}/dat.h5'  # Change this to your .h5 file path
    f = h5py.File(file_path, 'r')
    density = f[f'density_'][-2000:]
    rho = np.mean(f['density_'][-2000:])
    f.close()
    return rho


def ufp(eps, mu, pe, lamb, flow):
    base_folder = f"../Data/gaussian/lambda{lamb}/128_eps{eps:.3f}"

    # Open the HDF5 file and inspect contents
    file_path = f'{base_folder}/{flow}_Pe{pe:.1f}_mu{mu:.5f}/dat.h5'  # Change this to your .h5 file path
    f = h5py.File(file_path, 'r')

    u = f[f't'][:]

    return u


def check_corrupt_files(eps, lamb, mu_values, pe_values, flow):
    """Check for corrupted or missing .h5 files in the directory structure."""

    D = 1e-4  # Constant, not used here but retained from your function

    # Base directory structure
    base_folder = f"../Data/gaussian/lambda{lamb}/128_eps{eps:.3f}"

    corrupt_files = []

    # Loop through mu and Pe values
    for i, mu in enumerate(mu_values):
        for j, pe in enumerate(pe_values):
            # Determine the correct folder name
            fl = 'sinusoidal' if pe == 0 else flow
            file_path = f"{base_folder}/{fl}_Pe{pe:.1f}_mu{mu:.5f}/dat.h5"

            if not os.path.exists(file_path):  # Check if file is missing
                print(f"Missing file: {file_path}")
                corrupt_files.append(file_path)
                continue

            # Try to open the .h5 file
            try:
                with h5py.File(file_path, "r") as f:
                    _ = f.keys()  # Try reading the file structure
            except Exception as e:
                print(f"Corrupted file at mu={mu}, Pe={pe}: {file_path} - Error: {e}")
                corrupt_files.append(file_path)

    print("\nList of corrupted or missing files:", corrupt_files)
    return corrupt_files


"""
_________________________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________
________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________
"""
# ==========================================
#
# Begin figure
fig = plt.figure(dpi=900, figsize=(10, 4))
# ==========================================
# Create division
subfigs = fig.subfigures(1, 2, hspace=0.0, wspace=-4 * 0.01, width_ratios=[1, 1])
# ==========================================
# Overall Details
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
# ==========================================
# ==========================================

lamb = 0.8
# Read data from file
mu_values, ubar_pos, ubar_neg = np.loadtxt(f'../Data/ubar_values{lamb:.4f}.dat', unpack=True)
eps = 0.357
D = 1e-4

# LEFT PANEL:  CARRYING CAPACITY
axL = subfigs[0].subplots(1, 1, sharey=True)

mu_list = [i * 0.0002 + 0.7644 for i in range(79)]
# mu_list = [mu for mu in mu_list if abs(mu - 0.7866) > 1e-10]

flow = 'rankine'
Pe = [i*10 for i in range(12)]

print(Pe)
print(mu_list)

# Call the function with specific eps and lambda
corrupt_files = check_corrupt_files(eps=eps, lamb=0.8, mu_values=mu_list, pe_values=Pe, flow="rankine")

nx = len(mu_list)
ny = len(Pe)
newmu_list = [(1-mu)/(D/eps**2) for mu in mu_list]
newPe = [pe* eps for pe in Pe]
bounds = np.array([newmu_list[0], newmu_list[-1]])
bounds2 = np.array([newPe[0], newPe[-1]])

####################
rho_matrix = np.zeros((nx, ny))
####################
j = 0
for m in tqdm(mu_list):
    index = np.where(mu_values == round(m, 4))[0]
    ubar = ubar_pos[index]

    for i, pe in enumerate(Pe):
        rho_matrix[j, i] = rho(0.357, mu_list[j], pe, 0.8, flow) / ubar[0]
    j += 1

# plt.plot(mu_list, rho_matrix[:,4], '.-')
# plt.show()
rho_matrix = rho_matrix.T
dZdx = np.gradient(rho_matrix, axis=1)  # Gradient along x
dZdy = np.gradient(rho_matrix, axis=0)  # Gradient along y

# Calculate the magnitude of the gradient
gradient_magnitude = np.sqrt(dZdx ** 2 + dZdy ** 2)

umin = np.min(rho_matrix)
norm = matplotlib.colors.Normalize(vmin=umin, vmax=1-0.002)

colors = ["#000000", "#800080", "#8A2BE2", "#FF0000", "#FF4500"]
cmap = 'inferno'

# Create a custom colormap
# cmap = LinearSegmentedColormap.from_list("gnuplot_style_gradient", colors)

pm = plt.imshow(rho_matrix,norm = norm, cmap=cmap, extent=np.concatenate((bounds, bounds2)), origin='lower',
                aspect='auto')
cbar = fig.colorbar(pm, ax=axL)
cbar.ax.tick_params(labelsize=10)
cbar.set_label('Time-averaged \n normalized population abundance', rotation=270, fontsize=11, labelpad=22)
plt.axvline((1-0.7684)/(D/eps**2), ymin=0, ymax=newPe[-1], ls='--', color='green', alpha=0.8)
# Details
axL.set_xlabel(r'Diffusive Damkhöler, $\mathrm{Da} $', fontsize=13)
axL.set_ylabel(r'Characteristic Péclet, $\mbox{Pe}$', fontsize=13, rotation=90)

###########################################
#
# Sinusoidal Marker box Coordinates
ax, ay = 0.778, 30
bx, by = 0.778, 100
cx, cy = 0.7748, 100
dx, dy = 0.770, 100
#
#
axn,bxn,cxn,dxn = (1-ax)/(D/eps**2), (1-bx)/(D/eps**2), (1-cx)/(D/eps**2), (1-dx)/(D/eps**2)
ayn,byn,cyn,dyn = ay*eps, by*eps, cy*eps, dy*eps
plt.text(axn, ayn, s='B',
         color='white')  # ,bbox={'facecolor':'limegreen','alpha':0.8,'edgecolor':'none','boxstyle':'round,pad=0.25'},
# ha='center', va='center')
plt.text(bxn, byn, s='C',
         color='white')  # ,bbox={'facecolor':'limegreen','alpha':0.8,'edgecolor':'none','boxstyle':'round,pad=0.25'},
# ha='center', va='center')
plt.text(cxn, cyn, s='D',
         color='k')  # ,bbox={'facecolor':'limegreen','alpha':0.8,'edgecolor':'none','boxstyle':'round,pad=0.25'},
# ha='center', va='center')
plt.text(dxn, dyn, s='E',
         color='k')  # ,bbox={'facecolor':'limegreen','alpha':0.8,'edgecolor':'none','boxstyle':'round,pad=0.25'},
# ha='center', va='center')
# # #
# # #
# # # #########
# # Right PANEL:  CARRYING CAPACITY
#
lamb = 0.8

axR = subfigs[1].subplots(2, 2, sharey=True, sharex=True)

bounds = np.array([-0.5, 0.5])
y = np.linspace(*bounds, ny + 1)[1:]
# x_, y_ = np.meshgrid(y, y, indexing="ij")
# vx = np.sin(w * y_)
# vy = np.zeros_like(vx)

# umax = np.max(ufp(eps, 0.78, 0, lamb, 'sinusoidal'))
norm = matplotlib.colors.Normalize(vmin=0., vmax=1)
c_shot = "inferno"
pcm = axR[0, 0].imshow(ufp(eps, ax, ay, lamb, flow).T, norm=norm, cmap=c_shot, origin="lower",
                       extent=np.concatenate((bounds, bounds)))
# g = ay * comp_rad * (D / comp_rad ** 2)
# pcm = axR[0,0].plot(g * (np.sin(w * y)), y, ls='--', color='r', alpha=0.6)
# #
# #
pcm = axR[1, 1].imshow(ufp(eps, dx, dy, lamb, flow).T, norm=norm, cmap=c_shot, origin="lower",
                       extent=np.concatenate((bounds, bounds)))

# axR[1, 1].set_title(np.mean(ufp(dx, dy,0)))
# g = dy * comp_rad * (D / comp_rad ** 2)
# pcm = axR[1,1].plot(g * (np.sin(w * y)), y, ls='--', color='r', alpha=0.6)
# pcm = axR[1,1].quiver(x_,y_, g*vx,g*vy)

pcm = axR[1, 0].imshow(ufp(eps, cx, cy, lamb, flow).T, norm=norm, cmap=c_shot, origin="lower",
                       extent=np.concatenate((bounds, bounds)))

# axR[1, 0].set_title(np.mean(ufp(cx, cy,0)))
# g = cy * comp_rad * (D / comp_rad ** 2)
# pcm = axR[1,0].plot(g * (np.sin(w * y)), y, ls='--', color='r', alpha=0.6)
#
# #
pcm = axR[0, 1].imshow(ufp(eps, bx, by, lamb, flow).T, norm=norm, cmap=c_shot, origin="lower",
                       extent=np.concatenate((bounds, bounds)))
cbar = fig.colorbar(pcm, ax=axR, format=tkr.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(labelsize=10)
cbar.set_label('Normalized population density', rotation=270, fontsize=12, labelpad=18)
# g = by * comp_rad * (D / comp_rad ** 2)
# pcm = axR[0,1].plot(g * (np.sin(w * y)), y, ls='--', color='r', alpha=0.6)
#
#
# axR[0,0].text(-4.-0.5, 1.15-0.5, s='I', fontweight='black',
#                  bbox=dict(facecolor='gainsboro', edgecolor='black', boxstyle='round,pad=0.25'))
axR[0, 0].text(-0.5, 1. - 0.5, s='B', fontweight='black', fontsize=13,
               bbox=dict(facecolor='gainsboro', edgecolor='black', boxstyle='round,pad=0.25'))
axR[0, 1].text(-0.5, 1. - 0.5, s='C', fontweight='black', fontsize=13,
               bbox=dict(facecolor='gainsboro', edgecolor='black', boxstyle='round,pad=0.25'))
axR[1, 0].text(-0.5, 1. - 0.5, s='D', fontweight='black', fontsize=13,
               bbox=dict(facecolor='gainsboro', edgecolor='black', boxstyle='round,pad=0.25'))
axR[1, 1].text(-0.5, 1. - 0.5, s='E', fontweight='black', fontsize=13,
               bbox=dict(facecolor='gainsboro', edgecolor='black', boxstyle='round,pad=0.25'))
axL.text(newmu_list[0], newPe[-1] + 0.5, s='A', fontweight='black', fontsize=13,
         bbox=dict(facecolor='gainsboro', edgecolor='black', boxstyle='round,pad=0.25'))
# #
axR[1, 0].set_xlabel(r'$x$', fontsize=15)
axR[1, 0].set_ylabel(r'$y$', fontsize=15, rotation=90)

axR[1, 1].set_xlabel(r'$x$', fontsize=15)
axR[0, 0].set_ylabel(r'$y$', fontsize=15, rotation=90)

if flow == 'sinusoidal':
    plt.savefig('Fig_HeatM_Vortex.png', bbox_inches="tight")  #
    plt.savefig('Fig_HeatM_SineFC.pdf', bbox_inches="tight")  #

else:
    plt.savefig('Fig_HeatM_Vortex.png', bbox_inches="tight")  #
    plt.savefig('Fig_HeatM_VortexFC.pdf', bbox_inches="tight")  #
