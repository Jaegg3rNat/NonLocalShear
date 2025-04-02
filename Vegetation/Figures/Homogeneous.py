import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
from tqdm.auto import tqdm
"""
_________________________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________
_________________________________________________  ANALITYCAL SOLUTION  ______________________________________________________________
_________________________________________________________________________________________________________________________________________
"""


from sympy import symbols, Eq, exp, solve

#SOLVE THE ANALITYCAL EQUATION
def equilibrium_solution(delta, mu):
    # Add this code where you need the solution

    chic = 1
    chif = delta + chic
    x = symbols('x')

    equation = Eq(exp(delta * x) * (1 - x), mu)
    solutions = solve(equation, x)

    # Print real solutions
    real_solutions = [sol.evalf() for sol in solutions if sol.is_real]
    print(f"\nEquilibrium solutions for exp({delta}x)(1-x) = {mu}:")

    # Convert symbolic solutions to float and separate positive/negative
    float_solutions = []
    for sol in real_solutions:
        try:
            float_value = float(sol.evalf())
            float_solutions.append(float_value)
        except:
            float_solutions.append(0)  # Assign 0 if the solution is not real

    if not float_solutions:
        return [0, 0]  # Return zero if no real solutions exist

    pos_sol = max(float_solutions)
    neg_sol = min(float_solutions)

    # If only one solution exists, use it for both
    if len(float_solutions) == 1:
        pos_sol = neg_sol = float_solutions[0]

    return [neg_sol, pos_sol]

"""
_________________________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________
________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________
"""
#CREATE A LIST OF SOLUTIONS AND SAVE IT AS EXTERNAL FILE

def equilibrium_solution_list(lamb):
    mu_list = np.arange(0.76, 0.79, 0.0005)
    ubar_pos = []
    ubar_neg = []
    for mu in mu_list:
        ubar = equilibrium_solution(lamb, mu)
        ubar_pos.append(ubar[1])
        ubar_neg.append(ubar[0])

    # Save data to file
    with open(f'ubar_values{lamb:.4f}.dat', 'w') as f:
        f.write("# delta    ubar_pos    ubar_neg\n")  # Header
        for mu, ubar_pos, ubar_neg in zip(mu_list, ubar_pos, ubar_neg):
            f.write(f"{mu:.6f}    {ubar_pos:.6f}    {ubar_neg:.6f}\n")
# equilibrium_solution_list(0.8)
"""
_________________________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________
________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________
"""

#PLOT THE EQUILIBRIUM SOLUTION OF ONE LIST
def plot_equilibrium_solution(lamb):
    # Read data from file
    # loadtxt automatically handles the commented header line (starting with #)
    mu_values, ubar_pos, ubar_neg = np.loadtxt(f'ubar_values{lamb:.4f}.dat', unpack=True)

    # Create the plot
    plt.figure(figsize=(8, 6))
    ubar_pos = np.maximum(0,ubar_pos)
    plt.plot(mu_values, ubar_pos, 'b-', label=r'$u_0+$')
    # plt.plot(mu_values, ubar_neg, 'r-', label=r'$\bar{u}^-$')

    plt.plot(mu_values, np.zeros(len(mu_values)), '--', color='k')
    # plt.plot(1,0, 'o', markersize=10, color='k')
    plt.xlabel('Delta')
    plt.xlim([0, 1.2])
    # plt.ylim([-0.5, 1])
    plt.ylabel('Ubar')
    plt.title('Equilibrium Solutions')
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()

# plot_equilibrium_solution(lambd)
"""
_________________________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________
________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________
"""

#PLOT EQUILIBRIUM SLUTION OF ALL LIST YOU DEFINED AS PARAMETER

def plot_equilibrium_solutions(lambd_list):
    plt.figure(figsize=(8, 6))

    for lamb in lambd_list:
        try:
            # Read data from file
            mu_values, ubar_pos, ubar_neg = np.loadtxt(f'ubar_values{lamb:.4f}.dat', unpack=True)

            # Ensure non-negative values
            ubar_pos = np.maximum(0, ubar_pos)

            # Plot the data
            plt.plot(mu_values, ubar_pos,'.-', label=rf'$\lambda = {lamb:.2f}$')

        except FileNotFoundError:
            print(f"Warning: File 'ubar_values{lamb:.2f}.dat' not found. Skipping...")

    # Add reference lines and labels
    plt.plot(mu_values, np.zeros(len(mu_values)), '--', color='k', label="Zero Line")
    plt.xlabel(r'$\mu$')
    plt.xlim([0, 1.2])
    plt.ylabel(r'$u_0$')
    plt.axvline(1,linestyle = '--',alpha = 0.5,  color = 'k')
    plt.title('Homogeneous Equilibrium Solutions')
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()

"""
_________________________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________
________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________
"""



"""
_________________________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________
"""
#DO A SNAPSHOT OF A SPECIF POINT IN PARAMETER SPACE
def snapshot(lambd,pe,flow,mu,eps):
    base_folder = f"lambda{lambd}/128_eps{eps:.3f}"



    # # Example usage
    file_path = f'{base_folder}/{flow}_Pe{pe:.1f}_mu{mu:.4f}/dat.h5'  # Change this to your .h5 file path
    # explore_h5(file_path)
    #
    #
    f = h5py.File(file_path, 'r')
    #

    #
    u = f[f't'][:]
    density2 = f['density_'][:]
    vec_time = f['time_'][:]
        #     ### Plot
    plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(wspace=0.05)
    plt.subplot(1, 2, 1)
    plt.imshow(u.T, cmap="gnuplot", origin="lower")
    plt.colorbar(ticks=np.linspace(np.min(u), np.min(u) + 0.9 * (np.max(u) - np.min(u)), 7))
    #


    plt.subplot(1, 2, 2)
    line = np.array(density2)
    plt.plot(vec_time, line, c="k")
    plt.title(f'Mean ,{np.mean(line)}')
    plt.show()

#
#
flow = 'rankine'
# flow ='sinusoidal'
snapshot(0.8,145,flow, 0.775,0.357)
#
"""
_________________________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________
________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________
"""
#CREATE THE FIGURE OF MANY SNAPSHOTS
def snapshot1(lambd, pe, flow, mu, eps):
    base_folder = f"lambda{lambd}/128_eps{eps:.3f}"

    # Example usage
    file_path = f'{base_folder}/{flow}_Pe{pe:.1f}_mu{mu:.4f}/dat.h5'  # Change this to your .h5 file path

    # Open the .h5 file
    f = h5py.File(file_path, 'r')

    # Extract data from the file
    u = f[f't'][:]
    density2 = f['density_'][:]
    vec_time = f['time_'][:]
    norm = mpl.colors.Normalize(vmin=0, vmax=0.6)
    # Plot 1: Visualizing u (assuming it's a 2D field)
    plt.imshow(u.T, norm = norm, cmap="gnuplot", origin="lower")

    plt.colorbar(ticks=np.linspace(0, 0.6, 7))
def plot_2x2_snapshots():
    # Define the parameters for the four snapshots
    params = [
        (0.825,0, 'sinusoidal', 0.78,0.357),
        (0.825,0, 'sinusoidal', 0.82,0.357),
        (0.825,0, 'sinusoidal', 0.86,0.357),
        (0.825,0, 'sinusoidal', 0.95,0.357)
    ]

    # Set up a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(wspace=0.2, hspace=0.3)

    # Loop through the parameters and plot each snapshot
    for i, (lambd, pe, flow, mu, eps) in enumerate(params):
        # Call the snapshot function (adjusted to work in a subplot)
        plt.subplot(2, 2, i + 1)
        snapshot1(lambd, pe, flow, mu, eps)

        # Set a title for each subplot
        ax = axes[i // 2, i % 2]
        ax.set_title(f"λ={lambd}, Pe={pe}, μ={mu}, ε={eps}")

    # Show the plot
    plt.show()

# plot_2x2_snapshots()

"""
_________________________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________
________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________
"""
#PLOT SOLUTION FROM FILE AND ZERO FLOW
def plot_solution(lamb, eps):
    # Read data from file
    mu_values, ubar_pos, ubar_neg = np.loadtxt(f'ubar_values{lamb:.4f}.dat', unpack=True)

    # Create the plot
    plt.figure(figsize=(8, 6))
    ubar_pos = np.maximum(0, ubar_pos)
    plt.plot(mu_values, ubar_pos, 'b-', label='Local Homogenous')
    # plt.plot(mu_values, np.zeros(len(mu_values)), '--', color='k')
    plt.xlabel('$\mu$', fontsize=10)
    plt.xlim([0, 1.2])
    plt.ylabel('Ubar', fontsize=10)
    plt.title(r'Equilibrium Solutions, $\lambda =$ ' f'{lamb}')
    plt.grid(True)
    # Add hatched regions

    y_max = np.max(ubar_pos)
    # plt.fill_between(mu_values, 0, y_max, where=(mu_values <= 0.79), color='red', alpha=0.3, hatch='//',
    #                  label='Homogeneous Non Local')
    # plt.fill_between(mu_values, 0, y_max, where=(mu_values >= 1), color='red', alpha=0.3, hatch='//'
    #                  )

    base_folder = f"lambda{lamb}/128_eps{eps:.3f}"
    rho_ = []

    mu_list = []
    for i in range(81):
        m = i * 0.001 + 0.74
        m1 = 0.79

        dm = m-m1
        dm2 = m - 0.82
        if dm != 0 and dm2 != 0:
            mu_list.append(m)
        else:
            pass


    flow = 'sinusoidal'
    pe = 0
    # mu_list.remove(0.800)
    print(mu_list)
    for mu in tqdm(mu_list):
        print(mu)
        # index = np.where(mu_values == round(mu, 3))[0]
        file_path = f'{base_folder}/{flow}_Pe{pe:.1f}_mu{mu:.4f}/dat.h5'  # Change this to your .h5 file path
        f = h5py.File(file_path, 'r')
        density = f[f'density_'][-2000:]
        rho_.append(np.mean(density))

    plt.plot(mu_list, rho_, 'o', mfc='none', color='r', label='Non-Local Model')
#
#
#
#     # # Create an inset axis
#     # ax_inset = inset_axes(plt.gca(), width="40%", height="40%", loc='lower left', borderpad=2)
#     # ax_inset.plot(mu_values, ubar_pos, 'b-')
#     # ax_inset.plot(mu_list, rho_, 'o', mfc='none', color='r')
#     # ax_inset.set_xlim([0.7, 1])
#     # ax_inset.set_ylim([0, max(ubar_pos[mu_values >= 0.7])])
#     # ax_inset.grid(True)
#     # ax_inset.set_title('Zoomed Inset')
#
    plt.legend()

    plt.show()


# plot_solution(0.8,0.357)
"""
_________________________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________
________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________
"""
#PLOT SOLUTION AND ANY FLOW
def plot_solutions(lamb, eps):
    # Read data from file
    mu_values, ubar_pos, ubar_neg = np.loadtxt(f'ubar_values{lamb:.4f}.dat', unpack=True)

    # Create the plot
    plt.figure(figsize=(8, 6))
    ubar_pos = np.maximum(0, ubar_pos)
    plt.plot(mu_values, ubar_pos/ubar_pos, 'b-', label='Local Homogenous')
    # plt.plot(mu_values, np.zeros(len(mu_values)), '--', color='k')
    plt.xlabel('$\mu$', fontsize=10)
    plt.xlim([0, 1.2])
    plt.ylabel('Ubar', fontsize=10)
    plt.title(r'Equilibrium Solutions, $\lambda =$ ' f'{lamb}')
    plt.grid(True)

    base_folder = f"lambda{lamb}/128_eps{eps:.3f}"
    rho_ = []
    rho10 = []
    rho100 = []
    mu_list = []
    for i in range(61):
        m = i * 0.0005 + 0.76
        # m1 = 0.787

        # dm = m - m1
        # dm2 = m - 0.82
        # if dm != 0 and dm2 != 0:
        mu_list.append(m)

    for mu in mu_list:
        flow = 'sinusoidal'
        pe = 0
        index = np.where(mu_values == round(mu, 4))[0]
        file_path = f'{base_folder}/{flow}_Pe{pe:.1f}_mu{mu:.4f}/dat.h5'  # Change this to your .h5 file path
        f = h5py.File(file_path, 'r')
        density = f[f'density_'][-2000:]
        rho_.append(np.mean(density)/ubar_pos[index])

    plt.plot(mu_list, rho_, 'o-', color='r',markersize = 10, label=f'No FLow Pe ={pe}')
    #########################################
    for mu in mu_list:
        flow = 'sinusoidal'
        pe = 200
        index = np.where(mu_values == round(mu, 4))[0]
        file_path = f'{base_folder}/{flow}_Pe{pe:.1f}_mu{mu:.4f}/dat.h5'  # Change this to your .h5 file path
        f = h5py.File(file_path, 'r')
        density = f[f'density_'][-2000:]
        rho100.append(np.mean(density)/ubar_pos[index])

    plt.plot(mu_list, rho100, 'o-', mfc='none', color='m', label=f'{flow} Pe ={pe}')
    ###################################
    for mu in mu_list:
        flow = 'sinusoidal'
        pe = 60
        index = np.where(mu_values == round(mu, 4))[0]
        file_path = f'{base_folder}/{flow}_Pe{pe:.1f}_mu{mu:.4f}/dat.h5'  # Change this to your .h5 file path
        f = h5py.File(file_path, 'r')
        density = f[f'density_'][-2000:]
        rho10.append(np.mean(density)/ubar_pos[index])

    plt.plot(mu_list, rho10, 'o-', mfc='none', color='g',markersize =10, label=f'{flow} Pe ={pe}')
#########################################################
###################################
    rho10s= []
    for mu in mu_list:
        flow = 'sinusoidal'
        pe = 70
        index = np.where(mu_values == round(mu, 4))[0]
        file_path = f'{base_folder}/{flow}_Pe{pe:.1f}_mu{mu:.4f}/dat.h5'  # Change this to your .h5 file path
        f = h5py.File(file_path, 'r')
        density = f[f'density_'][-2000:]
        rho10s.append(np.mean(density)/ubar_pos[index])

    plt.plot(mu_list, rho10s, 'o-', mfc='none', color='k', label=f'{flow} Pe ={pe}')
#########################################################
    plt.legend()
#     plt.xlim([0.75,1.1])
    plt.ylim([-1, 2])
    plt.show()
plot_solutions(0.8,0.357)
# """
# _________________________________________________________________________________________________________________________________________
# _________________________________________________________________________________________________________________________________________
# """
#
# #
# # def snapshot(lambd, pe, flow, mu, eps, ax):
# #     base_folder = f"lambda{lambd}/128_eps{eps:.3f}"
# #     file_path = f'{base_folder}/{flow}_Pe{pe:.1f}_mu{mu:.3f}/dat.h5'  # Change this to your .h5 file path
# #     f = h5py.File(file_path, 'r')
# #     u = f[f't'][:]
# #     bounds = np.array([-0.5, 0.5])
# #     c_shot = "gnuplot"
# #     norm = mpl.colors.Normalize(vmin=0, vmax=1)
# #     # Plot on the provided axis
# #     im = ax.imshow(u.T, cmap=c_shot, norm= norm, origin="lower", extent=np.concatenate((bounds, bounds)))
# #     plt.colorbar(im, ax=ax)
# #     ax.set_title(f'Avg density: {np.mean(u):.3f}\n$\lambda$={lambd}, Pe={pe}, $\mu$={mu}, $\epsilon$={eps}')
# #
# # def plot_snapshots():
# #     # Define the base parameters
# #     lambd = 0.8
# #     pe = 0
# #     flow = 'sinusoidal'
# #     eps = 3
# #     mu_values = [0.7,0.8, 0.85, 0.9, 0.95,1]  # Example values for mu to vary
# #
# #     # Create a 2x2 grid of subplots
# #     fig, axes = plt.subplots(2, 3, figsize=(10, 10))
# #     axes = axes.flatten()  # Flatten the 2x2 array of axes for easy iteration
# #
# #     # Loop through the mu values and plot snapshots
# #     for i, mu in enumerate(mu_values):
# #         snapshot(lambd, pe, flow, mu, eps, axes[i])
# #
# #     # Adjust layout and display
# #     plt.tight_layout()
# #     plt.show()
# #
# # # Call the function to plot the snapshots
# # plot_snapshots()