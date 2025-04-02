'''
Figure 1: 2D system
    Homogeneous Distribution Curve
    Non Local Distribution Curve
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
from matplotlib import rc
from matplotlib.patches import Rectangle


# Functions
def up(mu, path, pe, w):
    f = h5py.File(f'{path}_mu{mu:.2f}_Pe{pe:.1f}_w{w:.2f}/dat.h5', 'r')

    D = 1e-4
    comp_rad = 0.2
    rho = np.mean(f[f'density'][-2000:])
    # print(f[f'density'][-2000:])
    u = rho / (mu * D / comp_rad ** 2)
    return u

def ufh(mu, path, pe, w):
    D = 1e-4
    comp_rad = 0.2
    f = h5py.File(f'{path}_mu{mu:.2f}_Pe{pe:.1f}_w{w:.2f}/dat.h5', 'r')
    t0 = f['time'][-1]
    ufh = f[f"t{t0}"][:] / (mu * D / comp_rad ** 2)
    return ufh


# # Begin figure
fig = plt.figure(dpi=900, figsize=(7, 5))
# =========================================
# Create division
subfigs = fig.subfigures(1, 1)
# =========================================
# Overall Details
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
# =========================================
base_folder = f"../Data/q_0.05Data"
velocity_field_name = "rankine"
# Open the HDF5 file and inspect contents
path = f'{base_folder}/{velocity_field_name}'
# =========================================

#Create Mu list:
mu_list = [130 + x * 5 for x in range(70)]
print(mu_list)

rho_list = [up(mu,path,0,0) for mu in mu_list]

axL = subfigs.subplots(1, 1, sharey=True)

line2, = plt.plot(mu_list, rho_list, '^-', markersize=5.5, color='dodgerblue', label='Patterns')
# line1, = plt.plot(mu_list, utH(), 'o-', color='mediumblue', label='Uniform')

plt.plot()
# # Details
# axL.legend(handles=[line1, line2],prop = font,loc = 2 )
# axL.yaxis.set_ticks([1.0, 1.2, 1.4, 1.6, 1.8])
axL.tick_params(axis='both', labelsize=12)
axL.set_ylim([0.9, 1.55])
axL.set_xlim([130, 500])

axL.set_xlabel(r'Diffusive Damköhler,$\mbox{Da}$', fontsize=14)
axL.set_ylabel('Normalized \n population abundance, $A$', fontsize=14, rotation=90)
# #axsleft.xaxis.get_major_formatter().set_powerlimits([3, 3])
axL.add_patch(Rectangle((0, 0), 185.192, 6,
                        # edgecolor = 'pink',
                        facecolor='blue',
                        fill=True,
                        hatch='////',
                        alpha=0.3,
                        lw=5))



axin1 = axL.inset_axes([0.5, 0.2, 0.3, 0.3])
bounds = np.array([-0.5, 0.5])
umax = np.max(ufh(400, path, 0, 0))
norm = matplotlib.colors.Normalize(vmin=0., vmax=umax)
pcm = axin1.imshow(ufh(400, path, 0, 0).T, norm=norm, cmap="gnuplot", origin="lower",
                         extent=np.concatenate((bounds, bounds)))
axin1.xaxis.set_tick_params(labelsize=12)
axin1.yaxis.set_tick_params(labelsize=12)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

axins = inset_axes(
    axin1,
    width="5%",  # width: 5% of parent_bbox width
    height="99%",  # height: 50%
    loc="lower left",
    bbox_to_anchor=(1.05, 0., 1, 1),
    bbox_transform=axin1.transAxes,
    borderpad=0,
)
cbar2 = fig.colorbar(pcm, cax=axins)
cbar2.ax.tick_params(labelsize=8)
# cbar = fig.colorbar(pcm, ax=axin1, shrink=0.6, ticks=[0, 0.6, 1.3])
cbar2.ax.tick_params(labelsize=10)
cbar2.set_label('Normalized \n Population density', rotation=270, fontsize =10,labelpad=20)

plt.savefig('Fig1_2D.png', bbox_inches="tight")  #
plt.savefig('Fig1_2D.pdf', bbox_inches="tight")
    

