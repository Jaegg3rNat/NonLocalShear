
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

from sympy import symbols, solve, exp, Eq

def rho(eps, mu, pe, lamb,flow):
    D = 1e-4

    base_folder = f"lambda{lamb}/128_eps{eps:.3f}"
    if pe ==0:
        fl = 'sinusoidal'
    else:
        fl = flow

    # Open the HDF5 file and inspect contents
    file_path = f'{base_folder}/{fl}_Pe{pe:.1f}_mu{mu:.4f}/dat.h5'  # Change this to your .h5 file path
    f = h5py.File(file_path, 'r')
    density = f[f'density_'][-2000:]
    rho = np.mean(f['density_'][-2000:])
    f.close()
    return rho
def ufp(eps, mu, pe, lamb, flow):
    base_folder = f"lambda{lamb}/128_eps{eps:.3f}"


    # Open the HDF5 file and inspect contents
    file_path = f'{base_folder}/{flow}_Pe{pe:.1f}_mu{mu:.4f}/dat.h5'  # Change this to your .h5 file path
    f = h5py.File(file_path, 'r')

    u = f[f't'][:]

    return u

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
mu_values, ubar_pos, ubar_neg = np.loadtxt(f'ubar_values{lamb:.4f}.dat', unpack=True)


# LEFT PANEL:  CARRYING CAPACITY
axL = subfigs[0].subplots(1, 1, sharey=True)

mu_list = []
for i in range(61):
    m = i * 0.0005 + 0.76
    mu_list.append(m)
flow = 'rankine'
if flow == 'sinusoidal':
    Pe = [0]
    for i in range(1,22):
        Pe.append(10+i * 10)
    # Pe.append(250)
    # Pe.append(260)

else:
    Pe = [0]
    for i in range(1,30):
        Pe.append(5+i * 5)

print(Pe)
print(mu_list)
nx = len(mu_list)
ny = len(Pe)

bounds = np.array([mu_list[0], mu_list[-1]])
bounds2 = np.array([Pe[0], Pe[-1]])

####################
rho_matrix = np.zeros((nx, ny))
####################
j = 0
for m in tqdm(mu_list):
    index = np.where(mu_values == round(m, 4))[0]
    ubar = ubar_pos[index]

    for i in tqdm(range(ny)):
        print(f'point {j}, {i}')
        rho_matrix[j, i] = rho(0.357,mu_list[j], Pe[i], 0.8, flow)/ubar

    j+=1

# plt.plot(mu_list, rho_matrix[:,4], '.-')
# plt.show()
rho_matrix = rho_matrix.T
dZdx = np.gradient(rho_matrix, axis=1)  # Gradient along x
dZdy = np.gradient(rho_matrix, axis=0)  # Gradient along y

# Calculate the magnitude of the gradient
gradient_magnitude = np.sqrt(dZdx ** 2 + dZdy ** 2)

umin = np.min(rho_matrix)
norm = matplotlib.colors.Normalize(vmin=umin, vmax=1)

pm = plt.imshow(gradient_magnitude, cmap="inferno", extent=np.concatenate((bounds, bounds2)), origin='lower',
                aspect='auto')
cbar = fig.colorbar(pm, ax=axL)
cbar.ax.tick_params(labelsize=10)
cbar.set_label('Time-averaged \n normalized population abundance', rotation=270, fontsize=11, labelpad=22)
plt.axvline(0.7685, ymin=0, ymax=Pe[-1],ls = '--', color = 'green',alpha = 0.8)
# Details
axL.set_xlabel(r'Growth ratio, $\mu $', fontsize=13)
axL.set_ylabel(r'Characteristic Péclet, $\mbox{Pe}$', fontsize=13, rotation=90)

############################################
if flow == 'sinusoidal':
    # Sinusoidal Marker box Coordinates
    ax, ay = 0.772, 20
    bx, by = 0.785, 40
    cx, cy = 0.7705, 210
    dx, dy = 0.7825, 200
else:

    # Rankine Marker box Coordinates
    ax, ay = 0.7775, 20
    bx, by = 0.7725, 135
    cx, cy = 0.7785, 135
    dx, dy = 0.7875, 135
#
plt.text(ax, ay, s='B',
         color='white')  # ,bbox={'facecolor':'limegreen','alpha':0.8,'edgecolor':'none','boxstyle':'round,pad=0.25'},
# ha='center', va='center')
plt.text(bx, by, s='C',
         color='black')  # ,bbox={'facecolor':'limegreen','alpha':0.8,'edgecolor':'none','boxstyle':'round,pad=0.25'},
# ha='center', va='center')
plt.text(cx, cy, s='D',
         color='black')  # ,bbox={'facecolor':'limegreen','alpha':0.8,'edgecolor':'none','boxstyle':'round,pad=0.25'},
# ha='center', va='center')
plt.text(dx, dy, s='E',
         color='white')  # ,bbox={'facecolor':'limegreen','alpha':0.8,'edgecolor':'none','boxstyle':'round,pad=0.25'},
# ha='center', va='center')
#
#
# # #########
# Right PANEL:  CARRYING CAPACITY
eps = 0.357
lamb = 0.8

axR = subfigs[1].subplots(2, 2, sharey=True, sharex=True)

bounds = np.array([-0.5, 0.5])
y = np.linspace(*bounds, ny + 1)[1:]
# x_, y_ = np.meshgrid(y, y, indexing="ij")
# vx = np.sin(w * y_)
# vy = np.zeros_like(vx)

umax = np.max(ufp(eps, 0.76, 0, lamb,'sinusoidal'))
norm = matplotlib.colors.Normalize(vmin=0., vmax=umax)
c_shot = "inferno"
pcm = axR[0, 0].imshow(ufp(eps, ax, ay,lamb,flow).T, norm=norm, cmap=c_shot, origin="lower", extent=np.concatenate((bounds, bounds)))
# g = ay * comp_rad * (D / comp_rad ** 2)
# pcm = axR[0,0].plot(g * (np.sin(w * y)), y, ls='--', color='r', alpha=0.6)


pcm = axR[1, 1].imshow(ufp(eps,dx, dy,lamb,flow).T, norm=norm, cmap=c_shot, origin="lower", extent=np.concatenate((bounds, bounds)))

# axR[1, 1].set_title(np.mean(ufp(dx, dy,0)))
# g = dy * comp_rad * (D / comp_rad ** 2)
# pcm = axR[1,1].plot(g * (np.sin(w * y)), y, ls='--', color='r', alpha=0.6)
# pcm = axR[1,1].quiver(x_,y_, g*vx,g*vy)

pcm = axR[1, 0].imshow(ufp(eps,cx, cy,lamb,flow).T, norm=norm, cmap=c_shot, origin="lower", extent=np.concatenate((bounds, bounds)))

# axR[1, 0].set_title(np.mean(ufp(cx, cy,0)))
# g = cy * comp_rad * (D / comp_rad ** 2)
# pcm = axR[1,0].plot(g * (np.sin(w * y)), y, ls='--', color='r', alpha=0.6)

#
pcm = axR[0, 1].imshow(ufp(eps,bx, by,lamb,flow).T, norm=norm, cmap=c_shot, origin="lower", extent=np.concatenate((bounds, bounds)))
cbar = fig.colorbar(pcm, ax=axR,  format=tkr.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(labelsize=10)
cbar.set_label('Normalized population density', rotation=270, fontsize=12, labelpad=18)
# g = by * comp_rad * (D / comp_rad ** 2)
# pcm = axR[0,1].plot(g * (np.sin(w * y)), y, ls='--', color='r', alpha=0.6)


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
axL.text(mu_list[0], Pe[-1]+0.5, s='A', fontweight='black', fontsize=13,
         bbox=dict(facecolor='gainsboro', edgecolor='black', boxstyle='round,pad=0.25'))

axR[1, 0].set_xlabel(r'$x$', fontsize=15)
axR[1, 0].set_ylabel(r'$y$', fontsize=15, rotation=90)

axR[1, 1].set_xlabel(r'$x$', fontsize=15)
axR[0, 0].set_ylabel(r'$y$', fontsize=15, rotation=90)



plt.savefig('Fig_HeatM_Vortex.png', bbox_inches="tight")  #
plt.savefig('Fig_HeatM_Vortex.pdf', bbox_inches="tight")  #