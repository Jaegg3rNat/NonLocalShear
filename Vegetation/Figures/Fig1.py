import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
from matplotlib import rc
from tqdm import tqdm
import matplotlib.ticker as tkr

# Begin figure
fig = plt.figure(dpi=900, figsize=(10, 6))
# ==========================================
# Create division
subfigs = fig.subfigures(1, 2, hspace=0.0, wspace=-10 * 0.06, width_ratios=[0.8, 1])
# ==========================================
# Overall Details
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
# ==========================================
# ==========================================

lamb = 0.8
eps = 0.357

axL = subfigs[0].subplots(1, 1, sharey=True)

# Read data from file
# loadtxt homogeneous
mu_values, ubar_pos, ubar_neg = np.loadtxt(f'../Data/ubar_values{lamb:.4f}.dat', unpack=True)
ubar_pos = np.maximum(0, ubar_pos)
# plot homogeneous

print(mu_values)

# Data of non-local model
base_folder = f"../Data/lambda_{lamb}"
rho_ = []
mu_list = np.arange(0.7844, 0.798, 0.0002)
mu_list = np.append(mu_list, np.arange(0.8, 1., 0.002))
print('mu list', mu_list)
# ===========================================
D = 1e-4
dx = 1 / 128
rc = 14 * dx
mu_l = list(mu_list)
newMu = [(mu) / (D / rc ** 2) for mu in mu_l]

flow = 'rankine'
pe = 0
#
#
for mu in tqdm(mu_list):
    index = np.where(mu_values == round(mu, 5))[0]
    file_path = f'{base_folder}/{flow}_Pe{pe:.1f}_mu{mu:.5f}/dat.h5'  # Change this to your .h5 file path
    f = h5py.File(file_path, 'r')
    density = f[f'density_'][-2000:]
    rho_.append(np.mean(density) / ubar_pos[index])
    f.close()

#
axL.plot(mu_values / (D / rc ** 2), ubar_pos / ubar_pos, 'b-', label=r'$Homogeneous$')
axL.plot(newMu, rho_, 'o', mfc='none', color='r', label='Non-Local Model')
axL.axvline(x=0.7863 / (D / rc ** 2), color='k', linestyle='--', alpha=0.6)
axL.axvline(x=0.828 / (D / rc ** 2), color='g', linestyle='--', alpha=0.4)
axL.axvline(x=0.916 / (D / rc ** 2), color='g', linestyle='--', alpha=0.4)

print('Da Critical', 0.7863 / (D / rc ** 2))
#
axL.set_xlabel(r'Normalized Death Ratio, $\mathrm{Da}^-$', fontsize=13)

axL.set_xlim([newMu[0] - 4, newMu[-25]])
axL.set_ylim([0.85, 1.35])
plt.legend()
axL.set_ylabel('Normalized Population Abundance, ' r'$A$', fontsize=13)
#
axL.text(x=0.868 / (D / rc ** 2), y=1.2, s=r' C ', fontsize=12, color='k', ha='center',
         bbox={'facecolor': 'ghostwhite', 'alpha': 0.8, 'edgecolor': 'gray', 'boxstyle': 'round,pad=0.25'})
axL.text(x=0.808 / (D / rc ** 2), y=1.15, s=r'B', fontsize=12, color='k', ha='center',
         bbox={'facecolor': 'ghostwhite', 'alpha': 0.8, 'edgecolor': 'gray', 'boxstyle': 'round,pad=0.25'})
axL.text(x=0.929 / (D / rc ** 2), y=1.25, s=r'D', fontsize=12, color='k', ha='center',
         bbox={'facecolor': 'ghostwhite', 'alpha': 0.8, 'edgecolor': 'gray', 'boxstyle': 'round,pad=0.25'})
# axL.text(x=( 0.766) / (D / rc ** 2), y=1.1, s=r'$\mathcal{H}^0$', fontsize=12, color='k', ha='center',
#          bbox={'facecolor': 'ghostwhite', 'alpha': 0.8, 'edgecolor': 'gray', 'boxstyle': 'round,pad=0.25'})

axL.text(x=newMu[0] - 4.1, y=1.35, s='A', fontweight='black', fontsize=13,
         bbox=dict(facecolor='gainsboro', edgecolor='black', boxstyle='round,pad=0.25'))
#
# ===========================================
# ===========================================
axR = subfigs[1].subplots(3, 1, sharey=False, sharex=True)
#
#
base_folder = f"../Data/lambda_{lamb}"
mu = 0.7906
# # # Example usage
file_path = f'{base_folder}/{flow}_Pe{pe:.1f}_mu{mu:.5f}/dat.h5'  # Change this to your .h5 file path
f = h5py.File(file_path, 'r')
u = f[f't'][:]

#     ### Plot
#
bounds = np.array([-0.5, 0.5])
axR[0].imshow(u, cmap="gnuplot", origin="lower",
              extent=np.concatenate((bounds, bounds)))
mu = 0.85
# # # Example usage
file_path = f'{base_folder}/{flow}_Pe{pe:.1f}_mu{mu:.5f}/dat.h5'  # Change this to your .h5 file path
f = h5py.File(file_path, 'r')
u = f[f't'][:]

#     ### Plot
#
axR[1].imshow(u, cmap="gnuplot", origin="lower",
              extent=np.concatenate((bounds, bounds)))
mu = 0.95
# # # Example usage
file_path = f'{base_folder}/{flow}_Pe{pe:.1f}_mu{mu:.5f}/dat.h5'  # Change this to your .h5 file path
f = h5py.File(file_path, 'r')
u = f[f't'][:]

#     ### Plot
#
pcm = axR[2].imshow(u.T, cmap="gnuplot", origin="lower",
                    extent=np.concatenate((bounds, bounds)))

cbar = fig.colorbar(pcm, ax=axR, format=tkr.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(labelsize=10)
cbar.set_label('Normalized population density', rotation=270, fontsize=14, labelpad=18)

axR[2].set_xlabel(r'$x$', fontsize=15)
axR[0].set_ylabel(r'$y$', fontsize=15, rotation=90)
axR[1].set_ylabel(r'$y$', fontsize=15, rotation=90)
axR[2].set_ylabel(r'$y$', fontsize=15, rotation=90)

axR[0].text(-0.5, 1. - 0.5, s='B', fontweight='black', fontsize=13,
            bbox=dict(facecolor='gainsboro', edgecolor='black', boxstyle='round,pad=0.25'))
axR[1].text(-0.5, 1. - 0.5, s='C', fontweight='black', fontsize=13,
            bbox=dict(facecolor='gainsboro', edgecolor='black', boxstyle='round,pad=0.25'))
axR[2].text(-0.5, 1. - 0.5, s='D', fontweight='black', fontsize=13,
            bbox=dict(facecolor='gainsboro', edgecolor='black', boxstyle='round,pad=0.25'))
# plt.show()


plt.savefig('Fig_NoFlowFC.png', bbox_inches="tight")  #
plt.savefig('Fig_NoFlowFC.pdf', bbox_inches="tight")
