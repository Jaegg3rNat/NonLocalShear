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
    # base_folder = f"../Data/lambda{lamb}/128_eps{eps:.3f}"
    base_folder = f"/data/workspaces/nathan/SDF/lambda{lamb}/128_eps{eps:.3f}"
    if pe == 0:
        fl = 'rankine'
    else:
        fl = flow

    # Open the HDF5 file and inspect contents
    file_path = f'{base_folder}/{fl}_Pe{pe:.1f}_mu{mu:.5f}/dat.h5'  # Change this to your .h5 file path
    f = h5py.File(file_path, 'r')
    rho = np.mean(f['density_'][-2000:])
    f.close()
    return rho


def ufp(eps, mu, pe, lamb, flow):
    # base_folder = f"../Data/lambda{lamb}/128_eps{eps:.3f}"
    base_folder = f"/data/workspaces/nathan/SDF/lambda{lamb}/128_eps{eps:.3f}"

    # Open the HDF5 file and inspect contents
    file_path = f'{base_folder}/{flow}_Pe{pe:.1f}_mu{mu:.5f}/dat.h5'  # Change this to your .h5 file path
    f = h5py.File(file_path, 'r')

    u = f[f't'][:]

    return u


def check_corrupt_files(eps, lamb, mu_values, pe_values, flow):
    """Check for corrupted or missing .h5 files in the directory structure."""

    D = 1e-4  # Constant, not used here but retained from your function

    # Base directory structure
    # base_folder = f"../Data/lambda{lamb}/128_eps{eps:.3f}"
    base_folder = f"/data/workspaces/nathan/SDF/lambda{lamb}/128_eps{eps:.3f}"

    corrupt_files = []

    # Loop through mu and Pe values
    for i, mu in enumerate(mu_values):
        for j, pe in enumerate(pe_values):
            # Determine the correct folder name
            fl = 'rankine' if pe == 0 else flow
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
fig = plt.figure(dpi=900, figsize=(12, 4))
# ==========================================
# Create division
subfigs = fig.subfigures(1, 2, hspace=0.0, wspace=-4 * 0.01, width_ratios=[.8, 1])
# ==========================================
# Overall Details
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
# ==========================================
# ==========================================

lamb = 0.8
# Read data from file
# mu_values, ubar_pos, ubar_neg = np.loadtxt(f'../Data/ubar_values{lamb:.4f}.dat', unpack=True)
mu_values, ubar_pos, ubar_neg = np.loadtxt(f'/data/workspaces/nathan/SDF/ubar_values{lamb:.4f}.dat', unpack=True)
eps = 0.357
dx = 1 / 128
rc = 14 * dx
D = 1e-4

# LEFT PANEL:  CARRYING CAPACITY
axL = subfigs[0].subplots(1, 1, sharey=True)

mu_list = [i * 0.0002 + 0.7844 for i in range(79)]
mu_list = [mu for mu in mu_list if abs(mu - 0.7866) > 1e-10]

flow = 'sinusoidal'
# Pe = [0, 10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180,185,190,200,205,210,220,225,230,240,250,260,270,280,290,300,310,320,350,600,650,700]
Pe = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 260, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500,
      520, 540, 560, 580, 600]
print(Pe)
print(mu_list)

# Call the function with specific eps and lambda
corrupt_files = check_corrupt_files(eps=eps, lamb=0.8, mu_values=mu_list, pe_values=Pe, flow="sinusoidal")

nx = len(mu_list)
ny = len(Pe)
newmu_list = [(mu) / (D / rc ** 2) for mu in mu_list]
newPe = [pe * rc for pe in Pe]
print(newPe)
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

# Define the colors for the gradient
import matplotlib.colors as mcolors


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
#                  plt.cm.Blues_r(x) for x in np.linspace(1 - white_pos, 1, int(n_colors * white_pos))
#              ] + [
#                  plt.cm.Reds(x) for x in np.linspace(0, 1., int(n_colors * (1 - white_pos)))
#              ]
#
#     return mcolors.ListedColormap(colors)
#
#
# # Example usage
# vmin, vmax = np.min(rho_matrix), np.max(rho_matrix)  # Your data range
# value_white = 1.0  # Value that should be white
# # print(rho_matrix[0,:])
#
# # Create colormap and norm
# cmap = create_custom_colormap(vmin, vmax, value_white)
# norm = mcolors.Normalize(vmin=vmin, vmax=vmax)


def create_custom_colormap(vmin, vmax, value_white=1.0, n_colors=256):
    white_pos = (value_white - vmin) / (vmax - vmin)

    # Below white: light blue
    blues = [plt.cm.Blues_r(x) for x in np.linspace((1 - white_pos)/2, 1, int(n_colors * white_pos))]

    # Above white: red
    reds = [plt.cm.Reds(x) for x in np.linspace(0, 1, int(n_colors * (0.99 - white_pos)))]

    colors = blues + reds
    return mcolors.ListedColormap(colors)


# Assuming your data has minimum at -2.5
vmin = np.min(rho_matrix)
print('vmin', vmin)
value_white = 1
vmax = 1.+ value_white-vmin   # Ensures symmetric range around white

cmap = create_custom_colormap(vmin, vmax, value_white)
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)



pm = plt.imshow(rho_matrix, norm=norm, cmap=cmap, extent=np.concatenate((bounds, bounds2)), origin='lower',
                aspect='auto')
cbar = fig.colorbar(pm, ax=axL)
cbar.ax.tick_params(labelsize=10)
cbar.set_label('Time-averaged \n normalized population abundance', rotation=270, fontsize=11, labelpad=22)
plt.axvline((0.7864) / (D / rc ** 2), ymin=0, ymax=newPe[-1], ls='--', color='k', alpha=0.8)
# Details
axL.set_xlabel(r'Normalized Death Ratio, $\mathrm{Da}^-$', fontsize=13)
axL.set_ylabel(r'Characteristic Péclet, $\mbox{Pe}$', fontsize=13, rotation=90)

###########################################
# # LEFT PANEL:  CARRYING CAPACITY
# axR = subfigs[1].subplots(1, 1, sharey=True)
# pm = plt.imshow(gradient_magnitude, cmap=cmap, extent=np.concatenate((bounds, bounds2)), origin='lower',
#                 aspect='auto')
# cbar = fig.colorbar(pm, ax=axR)
# cbar.ax.tick_params(labelsize=10)
# cbar.set_label('Gradient Modulus of Time-averaged \n normalized population abundance', rotation=270, fontsize=11, labelpad=22)
# plt.axvline((1-0.7864)/(D/eps**2), ymin=0, ymax=newPe[-1], ls='--', color='green', alpha=0.8)
# # Details
# axR.set_xlabel(r'Diffusive Damkhöler, $\mathrm{Da} $', fontsize=13)
# axR.set_ylabel(r'Characteristic Péclet, $\mbox{Pe}$', fontsize=13, rotation=90)

# Sinusoidal Marker box Coordinates
ax, ay = 0.794, 20
bx, by = 0.794, 150
cx, cy = 0.794, 300
dx, dy = 0.7862, 15
ex, ey = 0.7866, 15
fx, fy = 0.788, 15

axn, bxn, cxn, dxn, exn, fxn = (ax) / (D / rc ** 2), (bx) / (D / rc ** 2), (cx) / (D / rc ** 2), (dx) / (D / rc ** 2), (
    ex) / (D / rc ** 2), (fx) / (D / rc ** 2)
ayn, byn, cyn, dyn, eyn, fyn = ay * rc, by * rc, cy * rc, dy * rc, ey * rc, fy * rc

print(axn, bxn, cxn, dxn, exn, fxn)
print(ayn, byn, cyn, dyn, eyn, fyn)
plt.text(axn, ayn, s=r'\textbf{b}',fontsize=13,
         color='k')  # ,bbox={'facecolor':'limegreen','alpha':0.8,'edgecolor':'none','boxstyle':'round,pad=0.25'},
# ha='center', va='center')
plt.text(bxn, byn, s=r'\textbf{c}',fontsize=13,
         color='k')  # ,bbox={'facecolor':'limegreen','alpha':0.8,'edgecolor':'none','boxstyle':'round,pad=0.25'},
# ha='center', va='center')
plt.text(cxn, cyn, s=r'\textbf{d}', fontsize=13,
         color='k')  # ,bbox={'facecolor':'limegreen','alpha':0.8,'edgecolor':'none','boxstyle':'round,pad=0.25'},
# ha='center', va='center')
# plt.text(dxn, dyn, s='E',
#          color='k')  # ,bbox={'facecolor':'limegreen','alpha':0.8,'edgecolor':'none','boxstyle':'round,pad=0.25'},
# ha='center', va='center')
# plt.text(exn, eyn, s='F',
#          color='k')  # ,bbox={'facecolor':'limegreen','alpha':0.8,'edgecolor':'none','boxstyle':'round,pad=0.25'},
# ha='center', va='center')
# plt.text(fxn, fyn, s='G',
#          color='k')  # ,bbox={'facecolor':'limegreen','alpha':0.8,'edgecolor':'none','boxstyle':'round,pad=0.25'},
# ha='center', va='center')
# #
# #
# # # #########
# # Right PANEL:  CARRYING CAPACITY
#
lamb = 0.8

axR = subfigs[1].subplots(2, 3, sharey=True, sharex=True)

# Get positions of all axes in the second row in subfigure coordinates
# bboxes = [ax.get_position(original=True) for ax in second_row]
# x0 = min(bbox.x0 for bbox in bboxes)
# y0 = min(bbox.y0 for bbox in bboxes)
# x1 = max(bbox.x1 for bbox in bboxes)
# y1 = max(bbox.y1 for bbox in bboxes)
# # Main Rounded Box
# rect = FancyBboxPatch(
#     (x0 + 0.45, y0 - 0.1),
#     0.35 * (x1 - x0),
#     1.3 * (y1 - y0),
#     transform=subfig_right.transFigure,
#     boxstyle="round,pad=0.05",
#     linewidth=2,
#     edgecolor="black",
#     facecolor="none",
#     linestyle="-",
#     alpha=0.3,
#     zorder=1
# )
# # subfig_right.patches.append(rect)

bounds = np.array([-0.5, 0.5])
y = np.linspace(*bounds, ny + 1)[1:]
# x_, y_ = np.meshgrid(y, y, indexing="ij")
# vx = np.sin(w * y_)
# vy = np.zeros_like(vx)

# umax = np.max(ufp(eps, 0.78, 0, lamb, 'sinusoidal'))
norm = matplotlib.colors.Normalize(vmin=0., vmax=1)
c_shot = "gnuplot"
index = np.where(mu_values == round(ax, 5))[0]
ubar = ubar_pos[index]
pcm = axR[0, 0].imshow(ufp(eps, ax, ay, lamb, flow).T, norm=norm, cmap=c_shot, origin="lower",
                       extent=np.concatenate((bounds, bounds)))
# g = ay * comp_rad * (D / comp_rad ** 2)
# pcm = axR[0,0].plot(g * (np.sin(w * y)), y, ls='--', color='r', alpha=0.6)
# #
# #
pcm = axR[0, 1].imshow(ufp(eps, bx, by, lamb, flow).T, norm=norm, cmap=c_shot, origin="lower",
                       extent=np.concatenate((bounds, bounds)))

# axR[1, 1].set_title(np.mean(ufp(dx, dy,0)))
# g = dy * comp_rad * (D / comp_rad ** 2)
# pcm = axR[1,1].plot(g * (np.sin(w * y)), y, ls='--', color='r', alpha=0.6)
# pcm = axR[1,1].quiver(x_,y_, g*vx,g*vy)
#
pcm = axR[0, 2].imshow(ufp(eps, cx, cy, lamb, flow).T, norm=norm, cmap=c_shot, origin="lower",
                       extent=np.concatenate((bounds, bounds)))

# axR[1, 0].set_title(np.mean(ufp(cx, cy,0)))
# g = cy * comp_rad * (D / comp_rad ** 2)
# pcm = axR[1,0].plot(g * (np.sin(w * y)), y, ls='--', color='r', alpha=0.6)
# #
# # #

norm = matplotlib.colors.Normalize(vmin=0.35, vmax=0.5)
pcm3 = axR[1, 0].imshow(ufp(eps, dx, dy, lamb, flow).T, norm=norm, cmap=c_shot, origin="lower",
                        extent=np.concatenate((bounds, bounds)))
#
pcm3 = axR[1, 1].imshow(ufp(eps, ex, ey, lamb, flow).T, norm=norm, cmap=c_shot, origin="lower",
                        extent=np.concatenate((bounds, bounds)))
norm = matplotlib.colors.Normalize(vmin=0, vmax=0.7)
pcm2 = axR[1, 2].imshow(ufp(eps, fx, fy, lamb, flow).T, norm=norm, cmap=c_shot, origin="lower",
                        extent=np.concatenate((bounds, bounds)))

cbar = fig.colorbar(pcm, ax=axR[0], shrink=0.9, format=tkr.FormatStrFormatter('%.1f'))

cbar3 = fig.colorbar(pcm3, ax=axR[1, 0:2], shrink=0.88, format=tkr.FormatStrFormatter('%.2f'))
# cbar.ax.tick_params(labelsize=10)
# cbar.set_label('Normalized population density', rotation=270, fontsize=12, labelpad=18)

# g = by * comp_rad * (D / comp_rad ** 2)
# pcm = axR[0,1].plot(g * (np.sin(w * y)), y, ls='--', color='r', alpha=0.6)
# #
# #
# # axR[0,0].text(-4.-0.5, 1.15-0.5, s='I', fontweight='black',
# #                  bbox=dict(facecolor='gainsboro', edgecolor='black', boxstyle='round,pad=0.25'))
axR[0, 0].text(-0.5, 1.08 - 0.5, s=r'\textbf{(b)}', fontweight='black', fontsize=13,
               bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.'))
axR[0, 1].text(-0.5, 1.08 - 0.5, s=r'\textbf{(c)}', fontweight='black', fontsize=13,
               bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.'))
axR[0, 2].text(-0.5, 1.07 - 0.5, s=r'\textbf{(d)}', fontweight='black', fontsize=13,
               bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.'))
axR[1, 0].text(-0.5, 1.07 - 0.5, s=r'\textbf{(e)}', fontweight='black', fontsize=13,
               bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.'))
# axR[1, 1].text(-0.5, 1. - 0.5, s='F', fontweight='black', fontsize=13,
#                bbox=dict(facecolor='gainsboro', edgecolor='black', boxstyle='round,pad=0.25'))
# axR[1, 2].text(-0.5, 1. - 0.5, s='G', fontweight='black', fontsize=13,
#                bbox=dict(facecolor='gainsboro', edgecolor='black', boxstyle='round,pad=0.25'))


axL.text(newmu_list[0], newPe[-1] + 1, s=r'\textbf{(a)}', fontweight='black', fontsize=13,
         bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.'))
# axR.text(newmu_list[0], newPe[-1] + 0.5, s='B', fontweight='black', fontsize=13,
#          bbox=dict(facecolor='gainsboro', edgecolor='black', boxstyle='round,pad=0.25'))
# # #
axR[1, 0].set_xlabel(r'$x$', fontsize=15)
axR[1, 0].set_ylabel(r'$y$', fontsize=15, rotation=90)
# #
axR[1, 1].set_xlabel(r'$x$', fontsize=15)
axR[0, 0].set_ylabel(r'$y$', fontsize=15, rotation=90)
axR[1, 2].set_xlabel(r'$x$', fontsize=15)

# After creating all your subplots but before saving
# OR: Manually adjust the position of axR[1,2] (rightmost subplot in second row)

# --- After shifting axR[1,2] ---
pos = axR[1, 2].get_position()

# Shift the subplot left (example)
new_x0 = pos.x0 - 0.05  # Adjust this value as needed
axR[1, 2].set_position([new_x0, pos.y0, pos.width, pos.height])

# --- Now adjust the colorbar (pcm2) ---
cbar2 = fig.colorbar(pcm2, ax=axR[1, 2], shrink=0.88, format=tkr.FormatStrFormatter('%.1f'))

# Get the new position of axR[1,2] after shifting
new_pos = axR[1, 2].get_position()

# Set the colorbar position relative to the shifted subplot
cbar2_ax = cbar2.ax
cbar2_pos = cbar2_ax.get_position()
cbar2_ax.set_position([
    new_pos.x1 - 0.03,  # Place it just to the right of the shifted subplot
    new_pos.y0,  # Same y-start as the subplot
    0.02,  # Width of the colorbar (adjust if needed)
    new_pos.height  # Match the subplot height
])

pos = axR[1, 2].get_position()
axR[1, 2].set_position([pos.x0 - 0.05, pos.y0, pos.width, pos.height])  # Shift left by 0.05

# ... (after creating subplots and colorbars)

# Add the global label to the right of axR
fig.text(
    0.94, 0.5,
    'Normalized population density',
    rotation=270,
    va='center',
    ha='center',
    fontsize=12,
    transform=subfigs[1].transFigure
)

# Add this AFTER your heatmap plot (pm = axL.imshow(...)) but BEFORE saving

# ==========================================
# Custom box parameters (adjust these values)
# ==========================================
box_Da = 0.7868  # x-coordinate (Da value)
box_Pe = 20 * rc  # y-coordinate (Pe value)
box_width = 0.2  # Width in Da units
box_height = 3  # Height in Pe units
box_style = dict(linewidth=1., edgecolor='red', facecolor='none', linestyle='-')

# ==========================================
# Add the box to axL
# ==========================================
box_x = box_Da / (D / rc ** 2)  # Convert Da to axis coordinates
box_y = box_Pe  # Pe is already in axis coordinates

from matplotlib.colors import to_rgba

# Optional: Add label at box center
axL.text(
    box_x, box_y,
    s=r'\textbf{e}',  # Label text
    color='red',
    ha='center',
    va='center',
    fontsize=12,
    bbox=dict(facecolor=to_rgba('gainsboro', alpha=0.2), edgecolor='black', boxstyle='round,pad=0.4'),

)

# =======================================================
# = =======================================================

if flow == 'sinusoidal':
    plt.savefig('Fig_HeatM_Vortex.png', bbox_inches="tight")  #
    plt.savefig('Fig_HeatM_SineFC.pdf', bbox_inches="tight")  #

else:
    plt.savefig('Fig_HeatM_Vortex.png', bbox_inches="tight")  #
    plt.savefig('Fig_HeatM_VortexFC.pdf', bbox_inches="tight")  #
