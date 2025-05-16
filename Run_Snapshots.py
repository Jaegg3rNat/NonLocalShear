import numpy as np
import matplotlib.pyplot as plt
# from tqdm import tqdm
import h5py
# from celluloid import Camera
# import os
import sys
from matplotlib import rc
import matplotlib
import matplotlib.ticker as tkr
import re

# Set up global plotting parameters
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

### HDF5 File Inspection
def print_name(name, obj):
    """Function to print the name of groups and datasets in an HDF5 file."""
    if isinstance(obj, h5py.Group):
        print(f"Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}, Shape: {obj.shape}, Dtype: {obj.dtype}")


def print_hdf5_contents(file_path):
    """Function to open an HDF5 file and print its contents."""
    with h5py.File(file_path, 'r') as f:
        f.visititems(print_name)


def time_hdf5(file_path):
    """Function to extract time values from HDF5 file."""
    with h5py.File(file_path, 'r') as f:
        t_values = np.array([
            float(re.search(r"t(\d+(\.\d+)?)", name).group(1))
            for name in f.keys() if re.match(r"t\d+(\.\d+)?$", name)
        ])
        t_values.sort()
    return t_values


def fig(mu, w, Pe, flow_type):
    """
    Generate and save figures from simulation data stored in an HDF5 file.

    Parameters:
    mu (float): Adimensional growth rate.
    w (float): Frequency parameter.
    Pe (float): Peclet number.
    """
    # Define domain and parameters
    nx = 128
    bounds = np.array([-0.5, 0.5])
    y = np.linspace(*bounds, nx + 1)[1:]
    D = 1e-4
    comp_rad = 0.2
    r = mu * D / comp_rad ** 2
    g = Pe * comp_rad * (D / comp_rad ** 2)


    base_folder = f"Logistic/Codes/simulation_results/{nx}_R{comp_rad:.1f}"
    velocity_field_name = flow_type

    # Open the HDF5 file and inspect contents
    file_path = f'{base_folder}/{velocity_field_name}_mu{mu:.2f}_Pe{Pe:.1f}_w{w:.2f}/dat.h5'
    with h5py.File(file_path, 'r') as f:
        # f.visititems(print_name)  # List contents of the file
        t = time_hdf5(file_path)
        # Iterate through the saved time steps and generate figures
        for j in range(8000):

            ii = 0.1 + 0.5 * j
            i = t[int(ii / 0.01)]
            print(i)
            # ti = f['time'][int(i / 0.01)]
            u = f[f"t{i}"][:]
            conc = f['density'][:int(ii / 0.01)]
            conc_time = f['time'][:int(ii / 0.01)]

            # Create a new figure with two subfigures
            fig = plt.figure(dpi=400, figsize=(10, 4))
            subfigs = fig.subfigures(1, 2, hspace=0, wspace=0, width_ratios=[1, 1])

            # LEFT PANEL: Population density plot
            umax = np.max(u)
            norm = matplotlib.colors.Normalize(vmin=0., vmax=umax)
            axL = subfigs[0].subplots(1, 1, sharey=True)
            pm = axL.imshow(u.T, cmap="gnuplot", extent=np.concatenate((bounds, bounds)), norm=norm, origin='lower', aspect='auto')
            cbar = fig.colorbar(pm, ax=axL, extend='both', format=tkr.FormatStrFormatter('%.1f'))
            cbar.ax.tick_params(labelsize=10)
            cbar.ax.get_yaxis().set_ticks([0, round(umax / 2, 1), round(umax, 1)])
            cbar.set_label('Population density', rotation=270, fontsize=13, labelpad=16)
            axL.set_xlim([bounds[0], bounds[1]])
            axL.set_xlabel(r'$x$', fontsize=14)
            axL.set_ylabel(r'$y$', fontsize=14)
            axL.set_title(f"$t = {ii:0.1f}$", fontsize=14)

            # RIGHT PANEL: Scaled population abundance over time
            axR = subfigs[1].subplots(1, 1, sharey=True)
            axR.plot(conc_time, conc / r, c="g")
            dot, = axR.plot(conc_time[-1], conc[-1] / r, 'o', c="k", label=f'$= {conc[-1] / r : .3f}$')
            axR.set_xlabel(r'Simulation time, t', fontsize=13)
            axR.set_ylabel('Scaled \n Population Abundance', fontsize=13)
            axR.set_ylim([0.1, 2])
            axR.set_xlim([0, 1200])
            axR.legend(handles=[dot], fontsize=10)

            # Save the figure
            plt.savefig(f'{base_folder}/{velocity_field_name}_mu{mu:.2f}_Pe{Pe:.1f}_w{w:.2f}/fig{j:05d}')
            plt.close()

# Call the fig function with parameters
fig(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4])

