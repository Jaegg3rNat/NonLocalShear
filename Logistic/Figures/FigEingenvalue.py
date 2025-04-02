import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
from matplotlib import rc
from matplotlib.patches import Rectangle
from scipy.special import j1 as Bj1
import matplotlib.ticker as tkr
import pandas as pd


# %Import all data
def data(type):
    data = pd.read_csv(f"ev{str(type)}.csv")
    return data


# ###########################################################################################
#
# # Begin figure
fig = plt.figure(dpi=900, figsize=(4, 3))

# Create division
subfigs = fig.subfigures(1, 1)
#
# Overall Details
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
#
# #########
# PANEL:  CARRYING CAPACITY
ax = subfigs.subplots(1, 1, sharey=True)

l = data(0.0)
l2 = data(0.5)
l3 = data(10.0)

# Plots
ax.axhline(0, 0, 300, color='k', lw=0.85, alpha=0.65)
ax.axvline(185.192, -1, 1, ls='dashed', color='gray', alpha=0.5)

line1, = ax.plot(l3.iloc[:, 0], l3.iloc[:, 1], ls='-', lw=5, color='k', label=r'$\mbox{Pe} = 10.0$')
line2, = ax.plot(l2.iloc[:, 0], l2.iloc[:, 1], '-', lw=2.5, color='r', label=r'$\mbox{Pe} = 0.5$')
line3, = ax.plot(l.iloc[:, 0], l.iloc[:, 1], ls='dashed', color='deepskyblue', label=r'$\mbox{Pe} = 0.0$')

# Details
ax.set_ylim([-0.6, 0.6])
ax.set_xlim([180, 190])
ax.legend(handles=[line1, line2, line3])
ax.set_xlabel(r'Diffusive Damköhler, $\mbox{Da} $', fontsize=11)
ax.set_ylabel('Maximum of the real part \n of largest eigenvalue', fontsize=12, rotation=90)

plt.savefig('Fig_Eigenvalue.png', bbox_inches="tight")  #
plt.savefig('Fig_Eigenvalue.pdf', bbox_inches="tight")  #
