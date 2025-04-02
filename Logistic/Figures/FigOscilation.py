# Packges
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import fftpack
import matplotlib
from matplotlib import rc


def carrying_cap(mu, pe):
    f = h5py.File(f'../Data/Sine_old/mu{mu:.2f}_w{2 * np.pi:.2f}_Pe{pe:.1f}/dat.h5', 'r')
    D = 1e-4
    comp_rad = 0.2
    k = f['conc2'][:] / (mu * D / comp_rad ** 2)
    x = f['time'][:]
    return x, k


def func(mu, pe):
    f = h5py.File(f'../Data/Sine_old/mu{mu:.2f}_w{2 * np.pi:.2f}_Pe{pe:.1f}/dat.h5', 'r')
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
fig = plt.figure(dpi=900, figsize=(7, 5))
# Overall Details
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Create Axis
subfigs = fig.subfigures(2, 1, hspace=0.0, wspace=0.0, height_ratios=[1, 1])
subfigsnest = subfigs[1].subfigures(1, 2,wspace=0.25, width_ratios=[1., 1])
plt.subplots_adjust(wspace=0.9,hspace=0.2,left=0.01,top=0.6,right=1,bottom=0.0)
axsnest1 = subfigsnest[0].subplots(1, 1, sharey=True)
axR = subfigsnest[1].subplots(1, 1, sharey=True)
axsnest0 = subfigs[0].subplots(1, 1, sharey=True)

# LEFT PANEL
#########
# PANEL A
mu = 600
t0, c0 = carrying_cap(mu, 0)
t1, c1 = carrying_cap(mu, 10)
t2, c2 = carrying_cap(mu, 50)
t3, c3 = carrying_cap(mu, 80)

t4, c4 = carrying_cap(mu, 180)
t5, c5 = carrying_cap(mu, 250)
#
# colors = ['#00008B', '#0000CD', '#4682B4', '#6495ED', '#00BFFF', '#87CEFA']
colors = ['#003f5c', '#444e86', '#955196', '#dd5182', '#ff6e54', '#ffa600']
line0, = axsnest0.plot(t0, c0, color=colors[0], label=r'$\mbox{Pe} = 0$')
line1, = axsnest0.plot(t1, c1, color=colors[1], label=r'$\mbox{Pe} = 10$')
line2, = axsnest0.plot(t2, c2, color=colors[2], label=r'$\mbox{Pe} = 50$')
line3, = axsnest0.plot(t3, c3, color=colors[3], label=r'$\mbox{Pe} = 80$')
#
line4, = axsnest0.plot(t4, c4, color=colors[4], label=r'$\mbox{Pe} = 180$')
line5, = axsnest0.plot(t5, c5, color=colors[5], label=r'$\mbox{Pe} = 250$')
# # details
#
axsnest0.set_xlabel('Simulation time', fontsize=12)
axsnest0.set_ylabel('Normalized \n Population Abundance', fontsize=10, rotation=90)
axsnest0.set_ylim([0.8, 1.8])
axsnest0.set_xlim([0, 2000])
axsnest0.text(-150, 1.9, s='A', fontweight='black',
              bbox=dict(facecolor='gainsboro', edgecolor='black', boxstyle='round,pad=0.25'))
font = {'family': 'serif',
        'weight': 'normal',
        'size': 11,
        }
axsnest0.legend(handles=[line0, line1, line2, line3,line4,line5],loc = 3,ncol=3)

axin1 = axsnest0.inset_axes([0.8, 0.2, 0.15, 0.4])
axin1.plot(t1, c1, color=colors[1])
axin1.plot(t2, c2, color=colors[2])
axin1.plot(t3, c3, color=colors[3])
axin1.set_ylim([1.5, 1.7])
axin1.set_xlim([1950, 2000])
axsnest0.indicate_inset_zoom(axin1)
#
# # PANEL B
l1 = 10000
fk = fftpack.fft(c1[-l1:])
q = fftpack.fftfreq(len(c1[-l1:]), 0.01)

fk1 = fftpack.fft(c2[-l1:])
q1 = fftpack.fftfreq(len(c2[-l1:]), 0.01)

fk2 = fftpack.fft(c3[-l1:])
q2 = fftpack.fftfreq(len(c3[-l1:]), 0.01)

line11, = axsnest1.loglog(q[:int(len(q) / 2)], abs(fk)[:int(len(q) / 2)], '.-', color=colors[1], label=r'$P_e = 10$')
line22, = axsnest1.loglog(q1[:int(len(q1) / 2)], abs(fk1)[:int(len(q1) / 2)], '.-', color=colors[2],
                         label=r'$P_e = 50$')
line33, = axsnest1.loglog(q2[:int(len(q2) / 2)], abs(fk2)[:int(len(q2) / 2)], '.-', color=colors[3],
                         label=r'$P_e = 80$')
# details
axsnest1.set_ylim([0.05, 1e3])
axsnest1.set_xlim([0, 1])
axsnest1.set_xlabel(r'Frequency, $\omega$', fontsize=12)
axsnest1.set_ylabel('Power spectrum of \n population abundance', fontsize=12, rotation=90)
axsnest1.text(3*0.001,   3*1e3, s='B', fontweight='black',
              bbox=dict(facecolor='gainsboro', edgecolor='black', boxstyle='round,pad=0.25'))

# axsnest1.legend(handles=[line11, line22, line33])
#
# RIGHT PANEL: CARACHTERISTIC FREQUENCY
pe_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]
freq_sim = []
freq_anal = []
mu = 600
for p in pe_list:
    x, k = carrying_cap(mu, p)
    fk = fftpack.fft(k[-l1:])
    max = np.max(abs(fk)[1:])
    # print(max)
    xx = abs(fk)[1:]
    ind = np.where(xx == max)[0]
    freq_sim.append(x[ind[0] + 1])
    freq_anal.append(1 / (2 * func(mu, p)))

line, = axR.plot(pe_list, freq_anal, '--', color='k', label='Analytical')
line1, = axR.plot(pe_list, freq_sim, 'o', color='b', label='Fourier')
axR.text(-1, 0.3, s='C', fontweight='black',
         bbox=dict(facecolor='gainsboro', edgecolor='black', boxstyle='round,pad=0.25'))
axR.legend(handles=[line, line1], prop=font)
#
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

axR.set_xlabel(r'Characteristic Péclet, $\mbox{Pe}$', fontsize=12)
axR.set_ylabel('Natural frequecy \n' r'peak, $\omega_{max}$', fontsize=12, rotation=90)
plt.savefig('Fig_Fourier.png', bbox_inches="tight")  #
plt.savefig('Fig_Fourier.pdf', bbox_inches="tight")  #
