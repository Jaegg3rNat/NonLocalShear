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
import matplotlib.colors as mcolors


def rho(mu, pe, w):
    D = 1e-4
    comp_rad = 0.2
    r = mu * D / comp_rad ** 2


    # base_folder = f"../Data/q_0.05Data"
    base_folder = f"/data/workspaces/nathan/Logistic/q_0.05Data"
    velocity_field_name = "rankine"

    # Open the HDF5 file and inspect contents
    file_path = f'{base_folder}/{velocity_field_name}_mu{mu:.2f}_Pe{pe:.1f}_w{w:.2f}/dat.h5'
    f = h5py.File(file_path, 'r')

    rho = np.mean(f['tot_density'][-1]) / r
    return rho


def carrying_cap(mu, pe, w):
    # base_folder = f"../Data/q_0.05Data"
    base_folder = f"/data/workspaces/nathan/Logistic/q_0.05Data"
    velocity_field_name = "rankine"

    # Open the HDF5 file and inspect contents
    file_path = f'{base_folder}/{velocity_field_name}_mu{mu:.2f}_Pe{pe:.1f}_w{w:.2f}/dat.h5'
    f = h5py.File(file_path, 'r')

    k = f['density2'][:]
    x = f['time'][:]
    f.close()
    return x, k


def ufp(mu, pe, w):
    # base_folder = f"../Data/q_0.05Data"
    base_folder = f"/data/workspaces/nathan/Logistic/q_0.05Data"
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
fig = plt.figure(dpi=900, figsize=(10, 4))

# Create division
subfigs = fig.subfigures(1, 2, hspace=0.0, wspace=-4 * 0.01, width_ratios=[1, 1])
#
# Overall Details
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
#
# #########


# LEFT PANEL:  CARRYING CAPACITY
axL = subfigs[0].subplots(1, 1, sharey=True)

mu = [130 + 5 * i for i in range(75)]
print(mu)
Pe = [i for i in range(31)]

print(Pe)
nx = len(mu)
ny = len(Pe)

bounds = np.array([mu[0], mu[-1]])
bounds2 = np.array([Pe[0], Pe[-1]])
D = 1e-4
comp_rad = 0.2
rho_matrix = np.zeros((nx, ny))
# print(rho_matrix)
for j in tqdm(range(nx)):
    for i in (range(ny)):
        rho_matrix[j, i] = rho(mu[j], Pe[i], 0)
rho_matrix = rho_matrix.T
dZdx = np.gradient(rho_matrix, axis=1)  # Gradient along x
dZdy = np.gradient(rho_matrix, axis=0)  # Gradient along y

# Calculate the magnitude of the gradient
gradient_magnitude = np.sqrt(dZdx ** 2 + dZdy ** 2)


# Define the colors for the gradient
# def create_custom_colormap(vmin, vmax, value_white, n_colors=256):
#     """
#     Create a colormap where:
#     - value_white is white (default 1.0)
#     - Values below value_white are light blue gradient
#     - Values above value_white are red gradient
#
#     Parameters:
#     - vmin: minimum value in data
#     - vmax: maximum value in data
#     - value_white: value that should be white (default 1.0)
#     - n_colors: number of discrete colors in colormap
#     """
#     # Calculate relative position of white point
#     white_pos = (value_white - vmin) / (vmax - vmin)
#
#     # Create colormap with:
#     # - Below white: blues getting lighter until white
#     # - Above white: reds getting stronger from white
#     colors = [
#                  plt.cm.Blues_r(x) for x in np.linspace((1 - white_pos)/2, 1, int(n_colors * white_pos))
#              ] + [
#                  plt.cm.Reds(x) for x in np.linspace(0, 1., int(n_colors * (1 - white_pos)))
#              ]
#
#     return mcolors.ListedColormap(colors)
#
#
# # Example usage
# vmin, vmax = 1., np.max(rho_matrix)  # Your data range
# value_white = 1.01  # Value that should be white
# # print(rho_matrix[0,:])
#
# # Create colormap and norm
# cmap = create_custom_colormap(vmin, vmax, value_white)
# norm = mcolors.Normalize(vmin=vmin, vmax=vmax)


def create_custom_colormap(vmin, vmax, value_white=1.0, n_colors=512):
    white_pos = (value_white - vmin) / (vmax - vmin)

    # Below white: light blue
    blues = [plt.cm.Blues_r(x) for x in np.linspace((1 - white_pos)/5, 1, int(n_colors * white_pos))]

    # Above white: red
    reds = [plt.cm.Reds(x) for x in np.linspace(0, 1, int(n_colors * (1 - white_pos)))]

    colors = blues + reds
    return mcolors.ListedColormap(colors)


# Assuming your data has minimum at -2.5
vmax = np.max(rho_matrix) # Ensures symmetric range around white
vmin = 1-0.6
value_white = 1.0


cmap = create_custom_colormap(vmin, vmax, value_white)
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)



pm = plt.imshow(rho_matrix, cmap=cmap,norm=norm, extent=np.concatenate((bounds, bounds2)), origin='lower',
                aspect='auto')
cbar = fig.colorbar(pm, ax=axL)
cbar.ax.tick_params(labelsize=10)
cbar.set_label('Time-averaged \n normalized population abundance', rotation=270, fontsize=11, labelpad=22)
plt.axvline(185.192, ymin=0, ymax=350, ls='--', color='k', alpha=0.8)
# Details
axL.set_xlabel(r'Diffusive Damköhler, $\mbox{Da} $', fontsize=13)
axL.set_ylabel(r'Characteristic Péclet, $\mbox{Pe}$', fontsize=13, rotation=90)

# # Marker box Coordinates
ax, ay = 280, 10
bx, by = 280, 25
cx, cy = 450, 10
dx, dy = 450, 25
#
plt.text(ax, ay, s=r'\textbf{b}',fontsize = 10,
         color='k')  # ,bbox={'facecolor':'limegreen','alpha':0.8,'edgecolor':'none','boxstyle':'round,pad=0.25'},
# ha='center', va='center')
plt.text(bx, by, s=r'\textbf{c}',fontsize = 10,
         color='k')  # ,bbox={'facecolor':'limegreen','alpha':0.8,'edgecolor':'none','boxstyle':'round,pad=0.25'},
# ha='center', va='center')
plt.text(cx, cy, s=r'\textbf{d}',fontsize = 10,
         color='k')  # ,bbox={'facecolor':'limegreen','alpha':0.8,'edgecolor':'none','boxstyle':'round,pad=0.25'},
# ha='center', va='center')
plt.text(dx, dy, s=r'\textbf{e}',fontsize = 10,
         color='k')  # ,bbox={'facecolor':'limegreen','alpha':0.8,'edgecolor':'none','boxstyle':'round,pad=0.25'},
# ha='center', va='center')
#
#
# # #########
# Right PANEL:  CARRYING CAPACITY


axR = subfigs[1].subplots(2, 2, sharey=True, sharex=True)
w = 2 * np.pi
bounds = np.array([-0.5, 0.5])
y = np.linspace(*bounds, ny + 1)[1:]
# x_, y_ = np.meshgrid(y, y, indexing="ij")
# vx = np.sin(w * y_)
# vy = np.zeros_like(vx)

umax = np.max(ufp(500, 0, 0))
norm = matplotlib.colors.Normalize(vmin=0., vmax=umax)
c_shot = "gnuplot"
# --------------------------------------------------
pcm = axR[0, 0].imshow(ufp(ax, ay,0).T, norm=norm, cmap=c_shot, origin="lower", extent=np.concatenate((bounds, bounds)))
# g = ay * comp_rad * (D / comp_rad ** 2)
# pcm = axR[0,0].plot(g * (np.sin(w * y)), y, ls='--', color='r', alpha=0.6)


axR[1, 1].imshow(ufp(dx, dy,0).T, norm=norm, cmap=c_shot, origin="lower", extent=np.concatenate((bounds, bounds)))
# axR[1, 1].set_title(np.mean(ufp(dx, dy,0)))
# g = dy * comp_rad * (D / comp_rad ** 2)
# pcm = axR[1,1].plot(g * (np.sin(w * y)), y, ls='--', color='r', alpha=0.6)
# pcm = axR[1,1].quiver(x_,y_, g*vx,g*vy)

axR[1, 0].imshow(ufp(cx, cy,0).T, norm=norm, cmap=c_shot, origin="lower", extent=np.concatenate((bounds, bounds)))
# axR[1, 0].set_title(np.mean(ufp(cx, cy,0)))
# g = cy * comp_rad * (D / comp_rad ** 2)
# pcm = axR[1,0].plot(g * (np.sin(w * y)), y, ls='--', color='r', alpha=0.6)

#
axR[0, 1].imshow(ufp(bx, by,0).T, norm=norm, cmap=c_shot, origin="lower", extent=np.concatenate((bounds, bounds)))
cbar = fig.colorbar(pcm, ax=axR,  format=tkr.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(labelsize=10)
# axR[0, 1].set_title(np.mean(ufp(bx, by,0)))
cbar.set_label('Normalized population density', rotation=270, fontsize=12, labelpad=18)
# g = by * comp_rad * (D / comp_rad ** 2)
# pcm = axR[0,1].plot(g * (np.sin(w * y)), y, ls='--', color='r', alpha=0.6)

# --------------------------------------------------
# axR[0,0].text(-4.-0.5, 1.15-0.5, s='I', fontweight='black',
#                  bbox=dict(facecolor='gainsboro', edgecolor='black', boxstyle='round,pad=0.25'))
axR[0, 0].text(-0.5, 1.08 - 0.5, s=r'\textbf{(b)}', fontweight='black', fontsize=14,
               bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.'))
axR[0, 1].text(-0.5, 1.08 - 0.5, s=r'\textbf{(c)}', fontweight='black', fontsize=14,
               bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.'))
axR[1, 0].text(-0.5, 1.07 - 0.5, s=r'\textbf{(d)}', fontweight='black', fontsize=14,
               bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.'))
axR[1, 1].text(-0.5, 1.07 - 0.5, s=r'\textbf{(e)}', fontweight='black', fontsize=14,
               bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.'))
axL.text(129, Pe[-1]+0.5, s=r'\textbf{(a)}', fontweight='black', fontsize=14,
         bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.'))
# --------------------------------------------------
axR[1, 0].set_xlabel(r'$x$', fontsize=15)
axR[1, 0].set_ylabel(r'$y$', fontsize=15, rotation=90)

axR[1, 1].set_xlabel(r'$x$', fontsize=15)
axR[0, 0].set_ylabel(r'$y$', fontsize=15, rotation=90)
# --------------------------------------------------
# ==================================================
# ==================================================
plt.savefig('Fig_HeatM_Vortex2.png', bbox_inches="tight")  #
plt.savefig('Fig_HeatM_Vortex2.pdf', bbox_inches="tight")  #
