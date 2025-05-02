import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
from matplotlib import rc
from matplotlib.patches import Rectangle
from scipy.special import j1 as Bj1
import matplotlib.ticker as tkr


def rho(mu, pe):
    D = 1e-4
    comp_rad = 0.2
    r = mu * D / comp_rad ** 2


    base_folder = f"../Data/q_0.05Data"
    velocity_field_name = "rankine"

    # Open the HDF5 file and inspect contents
    file_path = f'{base_folder}/{velocity_field_name}_mu{mu:.2f}_Pe{pe:.1f}_w{0:.2f}/dat.h5'
    f = h5py.File(file_path, 'r')

    rho = np.mean(f['tot_density'][-1]) / r
    return rho


def carrying_cap(mu, pe, w):
    base_folder = f"../Data/q_0.05Data"
    velocity_field_name = "rankine"

    # Open the HDF5 file and inspect contents
    file_path = f'{base_folder}/{velocity_field_name}_mu{mu:.2f}_Pe{pe:.1f}_w{w:.2f}/dat.h5'
    f = h5py.File(file_path, 'r')

    k = f['density2'][:]
    x = f['time'][:]
    f.close()
    return x, k


def ufp(mu, pe, w):
    base_folder = f"../Data/q_0.05Data"
    velocity_field_name = "rankine"

    # Open the HDF5 file and inspect contents
    file_path = f'{base_folder}/{velocity_field_name}_mu{mu:.2f}_Pe{pe:.1f}_w{w:.2f}/dat.h5'
    f = h5py.File(file_path, 'r')

    time = f['time'][-1]
    D = 1e-4
    comp_rad = 0.2

    u = f[f't{time}'][:] / (mu * D / comp_rad ** 2)
    f.close()

    return u

# ###########################################################################################
#
# # Begin figure
fig = plt.figure(dpi = 900, figsize=(8, 5))

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


mu = [130 + 5 * i for i in range(75)]
print(mu)
Pe = [i for i in range(31)]



nx = len(mu)
ny = len(Pe)
bounds = np.array([mu[0], mu[-1]])
bounds2 = np.array([Pe[0], Pe[-1]])

rho_matrix = np.zeros((nx, ny))
# print(rho_matrix)
for j in (range(nx)):
    for i in (range(ny)):
        rho_matrix[j,i] = rho(mu[j],Pe[i])
rho_matrix = rho_matrix.T

dZdx = np.gradient(rho_matrix, axis=1)  # Gradient along x
dZdy = np.gradient(rho_matrix, axis=0)  # Gradient along y
gradient_magnitude = np.sqrt(dZdx**2 + dZdy**2)

cmap ='gnuplot'
#plot the heatmap
pm = plt.imshow(gradient_magnitude,cmap=cmap,extent=np.concatenate((bounds, bounds2)), origin ='lower',aspect='auto')
cbar = fig.colorbar(pm, ax=axL)
cbar.ax.tick_params(labelsize=14)

cbar.set_label('Gradient Modulus of Time-averaged \n normalized population abundance', rotation=270, fontsize =14,labelpad=30)
plt.axvline(185.192, ymin=0, ymax=350,ls = '--', color = 'white',alpha = 0.8)

#Details
axL.set_xlabel(r'Diffusive Damköhler, $\mbox{Da} $', fontsize=16)
axL.set_ylabel(r'Characteristic Péclet, ${\mbox{Pe}}$', fontsize=16, rotation=90)
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
plt.savefig('Fig_SM2.png', bbox_inches="tight")
plt.savefig('Fig_SM2.pdf')