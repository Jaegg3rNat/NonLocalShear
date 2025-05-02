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




def rho(eps, mu, pe, lamb, flow,type = None):

    D = 1e-4
    if type == 'gaussian':
        base_folder = f"../Data/gaussian/lambda{lamb}/128_eps{eps:.3f}"
    else:
        base_folder = f"../Data/lambda{lamb}/128_eps{eps:.3f}"

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
    base_folder = f"../Data/lambda{lamb}/128_eps{eps:.3f}"

    # Open the HDF5 file and inspect contents
    file_path = f'{base_folder}/{flow}_Pe{pe:.1f}_mu{mu:.5f}/dat.h5'  # Change this to your .h5 file path
    f = h5py.File(file_path, 'r')

    u = f[f't'][:]

    return u


# ###########################################################################################
#
# # Begin figure
fig = plt.figure(dpi=900, figsize=(8, 5))

# Create division
subfigs = fig.subfigures(1, 1)
#
# Overall Details
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
#
# #########
# LEFT PANEL:  CARRYING CAPACITY
axL = subfigs.subplots(1, 1, sharey=True)
eps = 0.357
D = 1e-4
lamb = 0.8
# Read data from file
mu_values, ubar_pos, ubar_neg = np.loadtxt(f'../Data/ubar_values{lamb:.4f}.dat', unpack=True)



flow = 'rankine'
type = 'gaussian'

if flow == 'sinusoidal' and type =='homo':
    Pe = [0, 10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180,185,190,200,205,210,220,225,230,240,250,260,270,280,290,300,310,320]
    mu_list = [i * 0.0002 + 0.7844 for i in range(79)]
    mu_list = [mu for mu in mu_list if abs(mu - 0.7866) > 1e-10]
else:
    Pe = [i * 10 for i in range(13)]
    mu_list = [i * 0.0002 + 0.7644 for i in range(79)]
print(Pe)
print(mu_list)

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
        rho_matrix[j, i] = rho(0.357, mu_list[j], pe, 0.8, flow,type) / ubar[0]
    j += 1

# plt.plot(mu_list, rho_matrix[:,4], '.-')
# plt.show()
rho_matrix = rho_matrix.T
dZdx = np.gradient(rho_matrix, axis=1)  # Gradient along x
dZdy = np.gradient(rho_matrix, axis=0)  # Gradient along y

# Calculate the magnitude of the gradient
gradient_magnitude = np.sqrt(dZdx ** 2 + dZdy ** 2)

umin = np.min(rho_matrix)
norm = matplotlib.colors.Normalize(vmin=umin, vmax=1)

colors = ["#000000", "#800080", "#8A2BE2", "#FF0000", "#FF4500"]
cmap = 'inferno'

# Create a custom colormap
# cmap = LinearSegmentedColormap.from_list("gnuplot_style_gradient", colors)

pm = plt.imshow(gradient_magnitude, cmap=cmap, extent=np.concatenate((bounds, bounds2)), origin='lower',
                aspect='auto')
cbar = fig.colorbar(pm, ax=axL)
cbar.ax.tick_params(labelsize=10)

# plt.axvline((1-0.7864)/(D/eps**2), ymin=0, ymax=newPe[-1], ls='--', color='green', alpha=0.8)
# Details


cbar.set_label('Gradient Modulus of Time-averaged \n normalized population abundance', rotation=270, fontsize=14,
               labelpad=30)
# plt.axvline(185.192, ymin=0, ymax=350, ls='--', color='white', alpha=0.8)

# Details
axL.set_xlabel(r'Diffusive Damköhler, $\mbox{Da} $', fontsize=16)
axL.set_ylabel(r'Characteristic Péclet, $\mbox{Pe}$', fontsize=16, rotation=90)
axL.xaxis.set_tick_params(labelsize=14)
axL.yaxis.set_tick_params(labelsize=14)

# # #########
# # INSERT PLOT
# axin1 = axL.inset_axes([0.21, 0.2*3, 0.5/2, 0.6/2])
#
# mu2 = []
# for i in range(15):
#     mu2.append(180 + 1 * i)
# print(mu2)
# Pe2 = []
# for i in range(6):
#     Pe2.append(0 + 1 * i)
# print(Pe2)
# nx = len(mu2)
# ny = len(Pe2)
#
# bounds3 = np.array([mu2[0], mu2[-1]])
# bounds4 = np.array([Pe2[0], Pe2[-1]])
# D = 1e-4
# comp_rad = 0.2
# rho_matrix2 = np.zeros((nx, ny))
#
# for j in (range(nx)):
#     for i in (range(ny)):
#         rho_matrix2[j, i] = rho2(mu2[j], Pe2[i])
#
# dZdx2 = np.gradient(rho_matrix2.T, axis=1)  # Gradient along x
# dZdy2 = np.gradient(rho_matrix2.T, axis=0)  # Gradient along y
#
# # Calculate the magnitude of the gradient
# gradient_magnitude2 = np.sqrt(dZdx2 ** 2 + dZdy2 ** 2)
# umax = np.max(gradient_magnitude2[0,-1])
# norm = matplotlib.colors.Normalize(vmin=0., vmax=umax)
# im = axin1.imshow(gradient_magnitude2, cmap="gnuplot", extent=np.concatenate((bounds3, bounds4)),norm=norm, origin='lower', aspect='auto')
# axin1.tick_params(color = 'w',labelcolor = 'w')
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#
# axins = inset_axes(
#     axin1,
#     width="5%",  # width: 5% of parent_bbox width
#     height="99%",  # height: 50%
#     loc="lower left",
#     bbox_to_anchor=(1.05, 0., 1, 1),
#     bbox_transform=axin1.transAxes,
#     borderpad=0,
# )
# cbar2 = fig.colorbar(im, cax=axins)
# cbar2.ax.tick_params(labelsize=8,color = 'w',labelcolor = 'w')
#
# axin1.set_xlabel(r'$\mbox{Da} $', fontsize=13,color= 'w')
# axin1.set_ylabel(r'$\overline{\mbox{Pe}}$', fontsize=13, rotation=90,color= 'w')
# axin1.axvline(185.192, ymin=0, ymax=11, ls='--', color='white', alpha=0.8)
#
# axin1.set_ylim([Pe2[0], Pe2[-1]])
# axin1.set_xlim([mu2[0], mu2[-1]])
# axL.indicate_inset_zoom(axin1)
# bbox = axL.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
plt.savefig('Fig_SM1FC.png', bbox_inches="tight")
plt.savefig('Fig_SM2FC.pdf')
