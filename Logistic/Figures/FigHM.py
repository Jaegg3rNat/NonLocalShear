import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from matplotlib import rc
import matplotlib.ticker as tkr
import matplotlib.colors as mcolors
import os

from matplotlib.colors import LinearSegmentedColormap


# ------------------------------------------
# ------------------------------------------
def check_corrupt_files(mu_values, pe_values, flow):
    """Check for corrupted or missing .h5 files in the directory structure."""

    D = 1e-4  # Constant, not used here but retained from your function
    # Base directory structure
    base_folder = f"../Data/Sine"

    corrupt_files = []

    # Loop through mu and Pe values
    for i, mu in enumerate(mu_values):
        for j, pe in enumerate(pe_values):
            # Determine the correct folder name
            fl = 'sinusoidal' if pe == 0 else flow
            file_path = f"{base_folder}/{fl}_mu{mu:.2f}_Pe{pe:.1f}_w{2 * np.pi:.2f}/dat.h5"

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
def rho(mu,pe):
    # f = h5py.File(f'../Data/Sine_old/mu{mu:.2f}_w{2 * np.pi:.2f}_Pe{pe:.1f}/dat.h5', 'r')
    f = h5py.File(f'../Data/Sine/sinusoidal_mu{mu:.2f}_Pe{pe:.1f}_w{2 * np.pi:.2f}/dat.h5', 'r')
    D = 1e-4
    comp_rad = 0.2
    r = mu * D / comp_rad ** 2
    rho = np.mean(f['density'][-5000:-1]) / r
    #
    # D = 1e-4
    # comp_rad = 0.2
    # r = mu * D / comp_rad ** 2
    # rho = np.mean(f['conc'][-5000:-1]) / r
    return rho
def ufp(mu,Pe):
    # f = h5py.File(f'../Data/Sine_old/mu{mu:.2f}_w{2*np.pi:.2f}_Pe{Pe:.1f}/dat.h5', 'r')
    f = h5py.File(f'../Data/Sine/sinusoidal_mu{mu:.2f}_Pe{Pe:.1f}_w{2 * np.pi:.2f}/dat.h5', 'r')
    time = f['time'][-1]
    D = 1e-4
    comp_rad = 0.2

    u = f[f't{time}'][:] / (mu * D / comp_rad ** 2)

    return u

# ------------------------------------------
# ------------------------------------------
# Parameters
D = 1e-4
comp_rad = 0.2


# Begin figure
fig = plt.figure(dpi=900, figsize=(10, 4))
# Create division
subfigs = fig.subfigures(1, 2, hspace=0, wspace=-4 * 0.01, width_ratios=[1, 1])
#
# Overall Details
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
#
# LEFT PANEL:  CARRYING CAPACITY
axL = subfigs[0].subplots(1, 1, sharey=True)

mu = [110 + 10 * i for i in range(70)]
Pe = [10 + 5 * i for i in range(63)]
Pe.append(0)
Pe.sort()
print(Pe)
print(mu)

# Call the function with specific eps and lambda
corrupt_files = check_corrupt_files( mu_values=mu, pe_values=Pe, flow="sinusoidal")

nx = len(mu)
ny = len(Pe)
bounds = np.array([mu[0], mu[-1]])
bounds2 = np.array([Pe[0], Pe[-1]])

rho_matrix = np.zeros((nx, ny))
# print(rho_matrix)
for j in tqdm(range(nx)):
    for i in (range(ny)):
        rho_matrix[j,i] = rho(mu[j],Pe[i])
rho_matrix = rho_matrix.T

dZdx = np.gradient(rho_matrix, axis=1)  # Gradient along x
dZdy = np.gradient(rho_matrix, axis=0)  # Gradient along y
gradient_magnitude = np.sqrt(dZdx**2 + dZdy**2)
# Define the colors for the gradient
def create_custom_colormap(vmin, vmax, value_white, n_colors=256):
    """
    Create a colormap where:
    - value_white is white (default 1.0)
    - Values below value_white are light blue gradient
    - Values above value_white are red gradient

    Parameters:
    - vmin: minimum value in data
    - vmax: maximum value in data
    - value_white: value that should be white (default 1.0)
    - n_colors: number of discrete colors in colormap
    """
    # Calculate relative position of white point
    white_pos = (value_white - vmin) / (vmax - vmin)

    # Create colormap with:
    # - Below white: blues getting lighter until white
    # - Above white: reds getting stronger from white
    colors = [
                 plt.cm.Blues_r(x) for x in np.linspace(1 - white_pos, 1, int(n_colors * white_pos))
             ] + [
                 plt.cm.Reds(x) for x in np.linspace(0, 1., int(n_colors * (1 - white_pos)))
             ]

    return mcolors.ListedColormap(colors)


# Example usage
vmin, vmax = 1., np.max(rho_matrix)  # Your data range
value_white = 1.01  # Value that should be white
# print(rho_matrix[0,:])

# Create colormap and norm
cmap = create_custom_colormap(vmin, vmax, value_white)
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
# Define the colors for the gradient
colors = ["#000000", "#800080", "#8A2BE2", "#FF0000", "#FF4500"]
# cmap = 'gnuplot'

# Create a custom colormap
# cmap = LinearSegmentedColormap.from_list("gnuplot_style_gradient", colors)

pm = plt.imshow(rho_matrix,norm = norm,cmap=cmap,extent=np.concatenate((bounds, bounds2)), origin ='lower',aspect='auto')
cbar = fig.colorbar(pm, ax=axL)
cbar.ax.tick_params(labelsize=10)
cbar.set_label('Time-averaged \n normalized population abundance', rotation=270, fontsize =11,labelpad=22)
plt.axvline(185.192, ymin=0, ymax=350,ls = '--', color = 'black',alpha = 0.8)
plt.plot([185.192+10,800],[0,305],ls = '--', color = 'black',alpha = 0.8)
# plt.plot([290,800],[50,240],ls = '--', color = 'white',alpha = 0.8)

#Details
axL.set_xlabel(r'Diffusive Damköhler, $\mbox{Da} $', fontsize=13)
axL.set_ylabel(r'Characteristic Péclet, $\mbox{Pe}$', fontsize=13, rotation=90)


# Marker box Coordinates
ax,ay = 600, 20
bx,by = 600, 120
cx,cy = 600, 180
dx,dy = 600, 250

plt.text(ax, ay, s='B')#,bbox={'facecolor':'limegreen','alpha':0.8,'edgecolor':'none','boxstyle':'round,pad=0.25'},
          # ha='center', va='center')
plt.text(bx, by, s='C')#,bbox={'facecolor':'limegreen','alpha':0.8,'edgecolor':'none','boxstyle':'round,pad=0.25'},
          # ha='center', va='center')
plt.text(cx, cy, s='D')#,bbox={'facecolor':'limegreen','alpha':0.8,'edgecolor':'none','boxstyle':'round,pad=0.25'},
          # ha='center', va='center')
plt.text(dx, dy, s='E')#,bbox={'facecolor':'limegreen','alpha':0.8,'edgecolor':'none','boxstyle':'round,pad=0.25'},
          # ha='center', va='center')

# #########
# Right PANEL:  CARRYING CAPACITY
# axR = subfigs[1].subplots(1, 1, sharey=True)

axR = subfigs[1].subplots(2, 2, sharey=True,sharex = True)
w = 2*np.pi
bounds = np.array([-0.5,0.5])
y = np.linspace(*bounds, ny + 1)[1:]
# x_, y_ = np.meshgrid(y, y, indexing="ij")
# vx = np.sin(w * y_)
# vy = np.zeros_like(vx)

umax = np.max(ufp(450,0))
norm = matplotlib.colors.Normalize(vmin=0., vmax=umax)
c_shot = "gnuplot"
pcm =axR[0,0].imshow(ufp(ax,ay).T, norm=norm, cmap=c_shot, origin="lower", extent=np.concatenate((bounds, bounds)))
g = ay * comp_rad * (D / comp_rad ** 2)
pcm = axR[0,0].plot(g * (np.sin(w * y)), y, ls='--', color='white', alpha=0.8)


pcm = axR[1,1].imshow(ufp(dx,dy).T, norm=norm, cmap=c_shot, origin="lower", extent=np.concatenate((bounds, bounds)))
g = dy * comp_rad * (D / comp_rad ** 2)
pcm = axR[1,1].plot(g * (np.sin(w * y)), y, ls='--', color='white', alpha=0.8)
# pcm = axR[1,1].quiver(x_,y_, g*vx,g*vy)

pcm = axR[1,0].imshow(ufp(cx,cy).T, norm=norm, cmap=c_shot, origin="lower", extent=np.concatenate((bounds, bounds)))
g = cy * comp_rad * (D / comp_rad ** 2)
pcm = axR[1,0].plot(g * (np.sin(w * y)), y, ls='--', color='white', alpha=0.8)


pcm = axR[0,1].imshow(ufp(bx,by).T, norm=norm, cmap=c_shot, origin="lower", extent=np.concatenate((bounds, bounds)))
cbar = fig.colorbar(pcm, ax=axR,  ticks=[1.0, 7, umax],format=tkr.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(labelsize=10)
cbar.set_label('Normalized population density', rotation=270, fontsize =12,labelpad=18)
g = by * comp_rad * (D / comp_rad ** 2)
pcm = axR[0,1].plot(g * (np.sin(w * y)), y, ls='--', color='white', alpha=0.8)


# axR[0,0].text(-4.-0.5, 1.15-0.5, s='I', fontweight='black',
#                  bbox=dict(facecolor='gainsboro', edgecolor='black', boxstyle='round,pad=0.25'))
axR[0,0].text(-0.5, 1.-0.5, s='B', fontweight='black',fontsize = 13,
                 bbox=dict(facecolor='gainsboro', edgecolor='black', boxstyle='round,pad=0.25'))
axR[0,1].text(-0.5, 1.-0.5, s='C', fontweight='black',fontsize = 13,
                 bbox=dict(facecolor='gainsboro', edgecolor='black', boxstyle='round,pad=0.25'))
axR[1,0].text(-0.5, 1.-0.5, s='D', fontweight='black',fontsize = 13,
                 bbox=dict(facecolor='gainsboro', edgecolor='black', boxstyle='round,pad=0.25'))
axR[1,1].text(-0.5, 1.-0.5, s='E', fontweight='black',fontsize = 13,
                 bbox=dict(facecolor='gainsboro', edgecolor='black', boxstyle='round,pad=0.25'))
axL.text(99, 321, s='A', fontweight='black',fontsize = 13,
                 bbox=dict(facecolor='gainsboro', edgecolor='black', boxstyle='round,pad=0.25'))


axR[1,0].set_xlabel(r'$x$', fontsize=15)
axR[1,0].set_ylabel(r'$y$', fontsize=15, rotation=90)

axR[1,1].set_xlabel(r'$x$', fontsize=15)
axR[0,0].set_ylabel(r'$y$', fontsize=15, rotation=90)
# --------------------------------------------------
# ==================================================
# ==================================================
plt.savefig('Fig_HeatM.png', bbox_inches="tight")#
plt.savefig('Fig_HeatM.pdf', bbox_inches="tight") #

