# Packges
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import fftpack
import matplotlib
from matplotlib import rc


def carrying_cap(mu, pe):
    f = h5py.File(f'../Codes/simulation_results/128_R0.2/rankine_mu{mu:.2f}_Pe{pe:.1f}_w{0:.2f}/dat.h5', 'r')
    D = 1e-4
    comp_rad = 0.2
    k = f['density'][:] / (mu * D / comp_rad ** 2)
    x = f['time'][:]
    print(x[-10:-1])
    return x, k


def func(mu, pe):
    # f = h5py.File(f'../Data/HeatMap/Rankine/mu{mu:.2f}_w{2 * np.pi:.2f}_Pe{pe:.1f}/dat.h5', 'r')
    f = h5py.File(f'../Codes/simulation_results/128_R0.2/rankine_mu{mu:.2f}_Pe{pe:.1f}_w{0:.2f}/dat.h5', 'r')

    time = f['time'][-1]
    D = 1e-4
    comp_rad = 0.2
    bounds = np.array([-0.5, 0.5])
    y = np.linspace(*bounds, 128 + 1)[1:]
    u = f[f't{time}'][:] / (mu * D / comp_rad ** 2)

    u1 = np.max(u.T[int(2 * 128 / 4):int(3 * 128 / 4), :])
    i1 = np.where(u.T == u1)[0][0]
    u2 = np.max(u.T[int(1 * 128 / 4):int(2 * 128 / 4), :])
    i2 = np.where(u.T == u2)[0][0]
    # print(ii)
    # print(u.T[80,23])
    # print(u1)
    # COMPUTE THE VELOCITY
    g = pe * comp_rad * (D / comp_rad ** 2)
    va = g * np.sin(2 * np.pi * y[i1])
    vb = g * np.sin(2 * np.pi * y[i2])
    vrel = va - vb
    tos = 0.25 / 2 / vrel
    return tos


###BEGIN FIGURE
fig = plt.figure(dpi=900, figsize=(7, 2))
# Overall Details
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Create Axis
subfigs = fig.subfigures(1, 2, hspace=0, wspace=0.01, width_ratios=[1, 0.5])
# LEFT PANEL:  CARRYING CAPACITY
axsnest0 = subfigs[0].subplots(1, 1, sharey=True)
#
axsnest1 = subfigs[1].subplots(1, 1, sharey=True)

# LEFT PANEL
#########
# TOP: CARRYING CAPACITY VS TIME
mu = 450
p1 = 3
p2 = 10
p3 = 20
p4 = 25
t0, c0 = carrying_cap(mu, 0)
t1, c1 = carrying_cap(mu, p1)
t2, c2 = carrying_cap(mu, p2)
t3, c3 = carrying_cap(mu, p3)
t4, c4 = carrying_cap(mu, p4)

# colors = ['#00008B', '#0000CD', '#4682B4', '#6495ED', '#00BFFF', '#87CEFA']
colors = ['#003f5c', '#444e86', '#955196', '#dd5182', '#ff6e54', '#ffa600']
line0, = axsnest0.plot(t0, c0, color=colors[0], label=r'$\mbox{Pe} = 0$')
line1, = axsnest0.plot(t1, c1, color=colors[1], label=r'$\mbox{Pe} =$' f' {p1:.0f}')
line2, = axsnest0.plot(t2, c2, color=colors[2], label=r'$\mbox{Pe} =$' f' {p2:.0f}')
line3, = axsnest0.plot(t3, c3, color=colors[3], label=r'$\mbox{Pe} =$' f' {p3:.0f}')
line4, = axsnest0.plot(t4, c4, color=colors[4], label=r'$\mbox{Pe} =$' f' {p4:.0f}')

# details
axsnest0.set_xlabel('Simulation time', fontsize=12)
axsnest0.set_ylabel('Normalized \n Population Abundance', fontsize=10, rotation=90)
axsnest0.set_ylim([1., 1.8])
axsnest0.set_xlim([8000, 10000])
axsnest0.text(8000, 1.81, s='A', fontweight='black',
              bbox=dict(facecolor='gainsboro', edgecolor='black', boxstyle='round,pad=0.25'))
# font = {'family': 'serif',
#         'weight': 'normal',
#         'size': 11,
#         }
axsnest0.legend(handles=[line0, line1, line2, line3, line4], ncol=3, fontsize=7, loc=2)

axin1 = axsnest0.inset_axes([0.7, 0.83, 0.2, 0.3])
axin1.plot(t1, c1, color=colors[1])
axin1.plot(t2, c2, color=colors[2])
axin1.plot(t3, c3, color=colors[3])
axin1.plot(t4, c4, color=colors[4])
axin1.set_ylim([1.15, 1.6])
axin1.set_xlim([9750, 10000])
axin1.xaxis.set_tick_params(labelsize=6)
axin1.yaxis.set_tick_params(labelsize=6)
axsnest0.indicate_inset_zoom(axin1)

# BOTTOM: FOURIER TRANSFORM

l1 = 30000
step = int(1 / 0.01)  # This will be 5 in your case

fk = fftpack.fft(c1[-2*l1::step])
fk = abs(fk)
q = fftpack.fftfreq(len(c1[-2*l1::step]), 1)

fk1 = fftpack.fft(c2[-l1:])
q1 = fftpack.fftfreq(len(c2[-l1:]), 0.01)

fk2 = fftpack.fft(c3[-l1:])
q2 = fftpack.fftfreq(len(c3[-l1:]), 0.01)
fk3 = fftpack.fft(c4[-l1:])
fk3 = abs(fk3)
q3 = fftpack.fftfreq(len(c4[-l1:]), 0.01)
#
line11, = axsnest1.loglog(q[:int(len(q) / 2)], fk[:int(len(q) / 2)], '.-', color=colors[1],
                          label=f'$P_e = {p1:.0f}$')
line22, = axsnest1.loglog(q1[:int(len(q1) / 2)], abs(fk1)[:int(len(q1) / 2)], '.-', color=colors[2],
                          label=f'$P_e = {p2:.0f}$')
# line33, = axsnest1.loglog(q2[:int(len(q2) / 2)], abs(fk2)[:int(len(q2) / 2)], '.-', color=colors[3],
#                           label=f'$P_e = {p3:.0f}$')
line44, = axsnest1.loglog(q2[:int(len(q3) / 2)], fk3[:int(len(q3) / 2)], '.-', color=colors[4],
                          label=f'$P_e = {p4:.0f}$')
# #details
# axsnest1.set_ylim([0.1, 7000])
# axsnest1.set_xlim([0.0005, 0.2])
# xticks = [0.001, 0.002, 0.003, 0.004, 0.005]
# axsnest1.set_xticks(xticks)
# axsnest1.set_xticklabels([f"{tick*1000:.0f}e-3" for tick in xticks], fontsize=8)

axsnest1.set_xlabel('Frequency, $\omega$', fontsize=12)
axsnest1.set_ylabel('Power spectrum of \n population abundance', fontsize=12, rotation=90)

# axsnest1.text(5* 1e-4, 7 * 1e3, s='B', fontweight='black',
#               bbox=dict(facecolor='gainsboro', edgecolor='black', boxstyle='round,pad=0.25'))

# axsnest1.legend(handles=[line11, line22, line33],loc = 1)

# RIGHT PANEL: CARACHTERISTIC FREQUENCY
# pe_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# freq_sim = []
# freq_anal = []
# mu = 300
# for p in pe_list:
#     x, k = carrying_cap(mu, p)
#     fk = fftpack.fft(k[-l1:])
#     max = np.max(abs(fk)[1:])
#     # print(max)
#     xx = abs(fk)[1:]
#     ind = np.where(xx == max)[0]
#     freq_sim.append(x[ind[0] + 1])
#     freq_anal.append(1 / (2 * func(mu, p)))
#
# line, = axR.plot(pe_list, freq_anal, '--', color='k', label='Analytical')
# line1, = axR.plot(pe_list, freq_sim, 'o', color='b', label='Fourier')
# axR.text(-1, 0.3, s='C', fontweight='black',
#          bbox=dict(facecolor='gainsboro', edgecolor='black', boxstyle='round,pad=0.25'))
# axR.legend(handles=[line, line1], prop=font)

# freq_sim = []
# freq_anal = []
# mu = 700
# for p in pe_list:
#     x, k = carrying_cap(mu, p)
#     fk = fftpack.fft(k[-l1:])
#     max = np.max(abs(fk)[1:])
#     # print(max)
#     xx = abs(fk)[1:]
#     ind = np.where(xx == max)[0]
#     freq_sim.append(x[ind[0] +1])
#     freq_anal.append(1/(2*func(mu, p)))
#
# axR.plot(pe_list,freq_sim,'.-')
# axR.plot(pe_list,freq_anal,'--',color = 'y')

# axR.set_xlabel(r'$P_e $', fontsize=18)
# axR.set_ylabel(r'$\omega_{max}$', fontsize=18, rotation=90)
plt.savefig('Fig_FourierVortex.png', bbox_inches="tight")  #
plt.savefig('Fig_FourierVortex.pdf', bbox_inches="tight")  #
# plt.show()
