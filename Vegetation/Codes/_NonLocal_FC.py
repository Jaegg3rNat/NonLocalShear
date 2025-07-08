# -*- coding: utf-8 -*-
# %% Packages
# Import essential libraries for numerical computation, plotting, and performance optimization
import numpy as np  # For array operations and mathematical functions
from matplotlib import colors, cm  # For handling colors and colormaps in plots
from scipy import fftpack  # For Fast Fourier Transform operations
import matplotlib.pyplot as plt  # For creating plots
from tqdm import tqdm  # For progress bars during iterations
import os  # For operating system interface, e.g., file paths
import h5py  # For handling HDF5 file format
import sys  # For accessing command-line arguments
from numba import jit, njit, prange  # For Just-In-Time compilation and parallel loops

"""
________________________________________________________________________________________________________________________
________________________________________________________________________________________________________________________
_____This code use 3 input commands:
        input sys 1: 
                pe:  Float variable | value of velocity flow
        input sys 2: 
                flow_type: String variable | stationary velocity field configurations
                        sinusoidal, rankine, celullar, parabolic, constant.
        input sys 3:
                nx: Int variable | number of points in the lattice.
________________________________________________________________________________________________________________________
________________________________________________________________________________________________________________________                 
"""


# %% Functions
def tophat_kernel(dx, radius):
    # Calculate the radius in grid points corresponding to the competition radius
    r_int = int(radius / dx)  # Number of grid points within the competition radius

    # Initialize the kernel matrix
    num_points_inside = 0
    m = np.zeros((1 + 2 * r_int, 1 + 2 * r_int))  # Kernel matrix of appropriate size
    m_norm = np.pi * radius ** 2  # Normalization constant for the 2D kernel

    # Populate the kernel matrix
    for i in range(-r_int, r_int + 1):
        for j in range(-r_int, r_int + 1):
            if i ** 2 + j ** 2 <= r_int ** 2:  # Inside the competition radius
                m[i + r_int, j + r_int] = 1.0  # / m_norm
                num_points_inside += 1

    # Calculate the area of the discretized kernel
    area_discretized_kernel = num_points_inside * dx ** 2
    m /= area_discretized_kernel
    # Initialize the domain-wide kernel matrix
    m2 = np.zeros((nx, ny))  # Kernel matrix in domain space

    # Place the kernel matrix at the center of the domain
    m2[nx // 2 - r_int:nx // 2 + r_int + 1, ny // 2 - r_int:ny // 2 + r_int + 1] = m

    # Normalize the kernel with grid spacing
    m2 *= dx * dy  # Ensure the kernel accounts for the area of each grid cell
    return m2


def exp_kernel(dx, radius, X, Y):
    # Center of the grid
    x0 = (bounds[0] + bounds[1]) / 2
    y0 = (bounds[0] + bounds[1]) / 2

    # Compute the distance from the center
    distance = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
    pot = np.exp(- distance ** 2 / radius ** 2)
    normalization = np.trapz(np.trapz(pot, x, axis=0), y, axis=0)

    print(np.mean(pot))
    print(np.mean(pot / normalization))
    # plt.imshow(pot ,extent=[bounds[0], bounds[1], bounds[0], bounds[1]], )
    # plt.show()
    return pot * dx * dy / normalization


def rk4_pseudospectral(u, vx, vy, g, D, f_hat, c_hat, dt, n_step, kx, ky):
    """
        Implements the 4th-order Runge-Kutta method using a pseudospectral approach.

        Parameters:
        ----------
        u : ndarray
            Initial condition of the scalar field (2D array).
        vx, vy : ndarray
            Velocity field components in x and y directions.
        g : float or ndarray
            Gravity term or coupling parameter.
        D : float or ndarray
            Diffusion coefficient or matrix.
        m2,m4 : ndarray
            Interaction term or secondary field in Fourier space.
        dt : float
            Time step size.
        n_step : int
            Number of Runge-Kutta steps to integrate.
        kx, ky : ndarray
            Wavenumber vectors in x and y directions.

        Returns:
        -------
        u : ndarray
            Updated scalar field after n_step Runge-Kutta steps.
        """
    # Create mesh grids of wavenumbers for spectral computations
    kx_, ky_ = np.meshgrid(kx, ky, indexing="ij")

    for i in range(n_step):
        # Compute the Fourier transform of the scalar field
        u_hat = fftpack.fft2(u)

        # Perform Runge-Kutta updates in Fourier space
        k1_hat = update_step(u_hat, kx_, ky_, vx, vy, D, f_hat, c_hat, g)
        k2_hat = update_step(u_hat + dt / 2 * k1_hat, kx_, ky_, vx, vy, D, f_hat, c_hat, g)
        k3_hat = update_step(u_hat + dt / 2 * k2_hat, kx_, ky_, vx, vy, D, f_hat, c_hat, g)
        k4_hat = update_step(u_hat + dt * k3_hat, kx_, ky_, vx, vy, D, f_hat, c_hat, g)

        # Combine all RK4 steps to update the solution in Fourier space
        u_hat += dt / 6 * (k1_hat + 2 * (k2_hat + k3_hat) + k4_hat)

        # Transform back to physical space
        u = fftpack.ifft2(u_hat).real

    return u


def update_step(u_hat, kx_, ky_, vx, vy, D, f_hat, c_hat, g):
    """
    Computes a single time step in Fourier space for RK4 integration.

    Parameters:
    ----------
    u_hat : ndarray
        Fourier-transformed scalar field (2D array).
    kx_, ky_ : ndarray
        Wavenumber grids in Fourier space (2D arrays).
    vx, vy : ndarray
        Velocity field components in x and y directions.
    D : float or ndarray
        Diffusion coefficient.
    m2, m4 : ndarray
        Interaction kernel field in Fourier space.
    g : float or ndarray
        Coupling term for interaction between scalar field and velocity (e.g. Flow intensity).

    Returns:
    -------
    u_hat : ndarray
        Updated Fourier-transformed scalar field after applying operators.
    """
    # Transform back to physical space
    u = fftpack.ifft2(u_hat).real

    # Compute Delta
    DeltaF = fftpack.ifftshift(fftpack.ifft2(u_hat * f_hat))
    DeltaC = fftpack.ifftshift(fftpack.ifft2(u_hat * c_hat))

    mf = np.exp(chif * DeltaF)
    mc = np.exp(chic * DeltaC)
    # print(mf)
    # print(mc)

    # Apply diffusion, advection, and nonlinear interaction terms in Fourier space
    diffusion_term = -(4 * np.pi ** 2) * D * (kx_ ** 2 + ky_ ** 2) * u_hat
    advection_term_x = -1j * (2 * np.pi) * kx_ * fftpack.fft2(g * vx * u)
    advection_term_y = -1j * (2 * np.pi) * ky_ * fftpack.fft2(g * vy * u)

    nonlinear_term = fftpack.fft2((mf - mu * mc) * u) - fftpack.fft2(mf * u ** 2)

    # Combine terms with additional constant term (if r is defined elsewhere)
    u_hat_updated = diffusion_term + advection_term_x + advection_term_y + nonlinear_term

    return u_hat_updated


#  VORTEX FIELDS DEFINITIONS
@njit(parallel=True)
def pv_field_domain(x_, y_, pvs, strengths, bounds, periodic_repeats=1):
    """
    Computes the velocity field for point vortices in a periodic domain.

    Parameters:
    ----------
    x_, y_ : ndarray
        Meshgrid coordinates where the velocity field is computed.
    pvs : ndarray
        Coordinates of point vortices.
    strengths : ndarray
        Rotation strength (Γ) of each point vortex.
    bounds : tuple
        (min, max) bounds of the domain.
    periodic_repeats : int, optional
        Number of periodic repetitions for boundary conditions. Default is 1.

    Returns:
    -------
    vx, vy : ndarray
        x and y components of the velocity field.
    """
    L = bounds[1] - bounds[0]
    vx = np.zeros_like(x_)
    vy = np.zeros_like(x_)

    for n in range(len(pvs)):
        for j in prange(x_.shape[0]):
            for k in prange(y_.shape[1]):
                vx_temp = 0
                vy_temp = 0
                for i in range(-periodic_repeats, periodic_repeats + 1):
                    x, y = x_[j, k], y_[j, k]
                    dx = x - pvs[n, 0]
                    dy = y - pvs[n, 1]

                    if dx - i * L != 0 or dy != 0:
                        vx_temp -= np.sin(2 * np.pi * dy / L) / (
                                np.cosh(2 * np.pi * dx / L - 2 * np.pi * i) - np.cos(2 * np.pi * dy / L))
                    if dx != 0 or dy - i * L != 0:
                        vy_temp += np.sin(2 * np.pi * dx / L) / (
                                np.cosh(2 * np.pi * dy / L - 2 * np.pi * i) - np.cos(2 * np.pi * dx / L))

                vx[j, k] += strengths[n] / (2 * L) * vx_temp
                vy[j, k] += strengths[n] / (2 * L) * vy_temp

    return vx, vy


@njit(parallel=True)
def pv_field_domain2(x_, y_, strengths, bounds):
    """
    Computes the velocity field for a cellular vortex field.

    Parameters:
    ----------
    x_, y_ : ndarray
        Meshgrid coordinates where the velocity field is computed.
    strengths : ndarray
        Rotation strength (Γ) of the vortex field.
    bounds : tuple
        (min, max) bounds of the domain.


    Returns:
    -------
    vx, vy : ndarray
        x and y components of the velocity field.
    """
    L = bounds[1] - bounds[0]
    vx = np.zeros_like(x_)
    vy = np.zeros_like(x_)

    for j in prange(x_.shape[0]):
        for k in prange(y_.shape[1]):
            x, y = x_[j, k], y_[j, k]
            vx_temp = -np.sin(np.pi * x / L) * np.cos(np.pi * y / L)
            vy_temp = np.sin(np.pi * y / L) * np.cos(np.pi * x / L)

            vx[j, k] += strengths[0] * vx_temp
            vy[j, k] += strengths[0] * vy_temp

    return vx, vy


@njit(parallel=True)
def pv_field_domain3(x_, y_, strengths, bounds, periodic_repeats=2):
    """
    Computes the velocity field for a Rankine point vortex field in a periodic domain.

    Parameters:
    ----------
    x_, y_ : ndarray
        Meshgrid coordinates where the velocity field is computed.
    strengths : ndarray
        Rotation strength (Γ) of the vortex field.
    bounds : tuple
        (min, max) bounds of the domain.
    periodic_repeats : int, optional
        Number of periodic repetitions for boundary conditions. Default is 2.

    Returns:
    -------
    vx, vy : ndarray
        x and y components of the velocity field.
    """
    L = bounds[1] - bounds[0]
    # q = 0.05 * L  # Cutoff radius based on domain size
    q = 4 * dx

    vx = np.zeros_like(x_)
    vy = np.zeros_like(x_)

    for j in prange(x_.shape[0]):
        for k in prange(y_.shape[1]):
            x, y = x_[j, k], y_[j, k]
            r = np.sqrt(x ** 2 + y ** 2)
            vx_temp, vy_temp = 0, 0

            for i in range(-periodic_repeats, periodic_repeats + 1):
                if r >= q:
                    vx_temp -= np.sin(2 * np.pi * y / L) / (
                            np.cosh(2 * np.pi * x / L - 2 * np.pi * i) - np.cos(2 * np.pi * y / L))
                    vy_temp += np.sin(2 * np.pi * x / L) / (
                            np.cosh(2 * np.pi * y / L - 2 * np.pi * i) - np.cos(2 * np.pi * x / L))
                else:
                    vx_temp -= y * 2 / q
                    vy_temp += x * 2 / q

            vx[j, k] += strengths[0] / (2 * L) * vx_temp
            vy[j, k] += strengths[0] / (2 * L) * vy_temp

    return vx, vy


########################################
########################################
########################################

"""
________________________________________________________________________________________________________
________________________________________________________________________________________________________
_______________________________ SIMULATION PARAMETERS __________________________________________________
________________________________________________________________________________________________________
________________________________________________________________________________________________________
"""

# Seed for reproducibility of initial conditions
seed = 3
np.random.seed(seed)

# Domain bounds and system properties
bounds = np.array([-0.5, 0.5])  # Domain bounds
L = bounds[1] - bounds[0]  # Length of the domain

# Numerical grid properties
nx = 128  # Number of grid points in x
dx = L / nx  # Grid spacing in x
x = np.linspace(*bounds, nx + 1)[:-1]  # Periodic in x (exclude 0 for periodicity)

ny = nx  # Number of grid points in y
dy = L / ny  # Grid spacing in y
y = np.linspace(*bounds, ny + 1)[:-1]  # Periodic in y

comp_rad = 14 * dx   # Competition radius of the kernel
fac_rad = 5 * dx  # Facilitation radius of the kernel
eps = fac_rad / comp_rad

delta = float(sys.argv[2])
chic = 2
chif = delta + chic
mu = float(sys.argv[1])
#
# # Mesh grid of the numerical space (physical domain)
x_, y_ = np.meshgrid(x, y, indexing="ij")

# Print system properties
print("System interval:", bounds)
print("System Length:", L)
print("Number of points:", nx)
print("Delta x:", dx)

# Diffusion and biological parameters
D = 1e-4  # Diffusion coefficient
pe = float(sys.argv[4])  # Peclet number from command-line argument

# Velocity multiplier (gamma)
gamma = [pe * D]  # List of velocity multipliers
# Uncomment the following lines to add more velocity multipliers
# for i in range(1, 5):
#     gamma.append((pe + i) * comp_rad * (D / comp_rad ** 2))

flow_type = sys.argv[3]  # Flow type (e.g., 'rankine', 'sinusoidal', 'cellular')

# Print biological parameters
print("\n########### Biological Parameters ###########")
print("Competition Radius (R_comp):", comp_rad)
print("Facilitation Radius (R_fac):", fac_rad)
print("Diffusion Coefficient (D):", D)
print("Peclet Number (Pe):", pe, "// Velocity (gamma[0]):", gamma[0])
print("Flow type:", flow_type)

"""
________________________________________________________________________________________________________
________________________________________________________________________________________________________
________________________ NON-LOCAL KERNEL DEFINITION __________________________________________
________________________________________________________________________________________________________
________________________________________________________________________________________________________
"""

# m2 = tophat_kernel(dx, comp_rad)  # mc top hat kernel
# m4 = tophat_kernel(dx, fac_rad)  # mf top hat kernel;
m2 = exp_kernel(dx, comp_rad, x_, y_)
m4 = exp_kernel(dx, fac_rad, x_, y_)
#
f_hat = fftpack.fft2(m4)
c_hat = fftpack.fft2(m2)
"""
________________________________________________________________________________________________________
________________________________________________________________________________________________________
____________________________________SIMULATION START____________________________________________________
________________________________________________________________________________________________________
________________________________________________________________________________________________________
"""

# Ensure the correct flow type is provided via command-line arguments
# if len(sys.argv) < 3:
# raise ValueError("Please specify the flow type and other parameters (e.g., 'rankine', 'sinusoidal', 'cellular').")


# Define a directory to store results
main_directory = f"lambda{delta}"
if not os.path.exists(main_directory):
    os.makedirs(main_directory)

# Automatically create subdirectory based on nx value
lattice_size_dir = f"{main_directory}/{nx}_eps{eps:.3f}"
if not os.path.exists(lattice_size_dir):
    os.makedirs(lattice_size_dir)

# Loop over different values for the velocity multiplier gamma
for g in gamma:

    Pe = (g) / (D)

    # Velocity Field Flows based on flow type input
    if flow_type == "sinusoidal":
        w = 2 * np.pi
        vx = np.sin(w * y_)
        vy = np.zeros_like(vx)
    elif flow_type == "rankine":
        # You can select the appropriate function (e.g., pv_field_domain3) for rankine flow
        w = 0
        strengths = 2 * np.array([1])  # np.random.choice([-0.5, 0.5], size=n_vortex)
        vx, vy = pv_field_domain3(x_, y_, strengths, bounds)
    elif flow_type == "cellular":
        # Select the function for the cellular vortex flow
        w = 0
        strengths = 1 * np.array([1])  # np.random.choice([-0.5, 0.5], size=n_vortex)
        vx, vy = pv_field_domain2(x_, y_, strengths, bounds)
    elif flow_type == "pointvortex":
        # You can select the appropriate function (e.g., pv_field_domain3) for rankine flow
        w = 0
        n_vortex = 1
        space_repetitions = 1
        strengths = 2 * np.array([1])  # np.random.choice([-0.5, 0.5], size=n_vortex)
        pvs = np.zeros((1, 2))  # np.random.uniform(-0.01, 0.01, (n_vortex, 1))
        vx, vy = pv_field_domain(x_, y_, pvs, strengths, bounds, space_repetitions)
    elif flow_type == "constant":
        # Select the function for the cellular vortex flow
        w = 0
        vx = np.ones_like(x_)
        vy = np.zeros_like(vx)
    elif flow_type == "parabolic":
        # Select the function for the cellular vortex flow
        w = 0
        vx = (-4 * y_ ** 2 + 4 * y_)
        # vx = (-y_ ** 2 + 1**2)
        vy = np.zeros_like(vx)
    else:
        raise ValueError(f"Unsupported flow type: {flow_type}."
                         f"\n choose between: sinusoidal, rankine, cellular, pointvortex, constant, parabolic")
    # ________________________________________________________________________________________________________
    # # ________________________________________________________________________________________________________
    #
    # plt.plot(0, 0, 'o', color = 'r')
    # plt.plot(x,vy[:,int(nx/2)], '.-')
    #
    # plt.show()
    # plt.close()
    # # ________________________________________________________________________________________________________
    # # ________________________________________________________________________________________________________
    # # # Visualize the flow for simulation control
    # N = np.sqrt(vx ** 2 + vy ** 2)
    # norm = colors.Normalize(vmin=0, vmax=0.5)
    # vx /= N
    # vy /= N
    # start = 0
    # sl = 6
    # # Normalize the density metric for coloring
    #
    # # Select a colormap
    # cmap = cm.Blues
    # color = 'mediumblue'
    # plt.quiver(x_[start::sl, start::sl], y_[start::sl, start::sl], vx[start::sl, start::sl],
    #            vy[start::sl, start::sl], N[start::sl, start::sl], units='inches',
    #            alpha=0.9)
    # plt.show()
    # plt.close()
    # ________________________________________________________________________________________________________
    # ________________________________________________________________________________________________________

    # Create path for saving results based on nx value and flow parameters
    path = f"{lattice_size_dir}/{flow_type}_Pe{Pe:.1f}_mu{mu:.5f}"
    if not os.path.exists(path):
        os.makedirs(path)

    # Create file for saving results
    # h5file = h5py.File(f"{path}/dat.h5", "w")

    """
    ________________________________________________________________________________________________________
    ________________________________________________________________________________________________________
    __________________________ INITIAL CONFIGURATION _______________________________________________________
    ________________________________________________________________________________________________________
    ________________________________________________________________________________________________________
    """


    # # Define initial Gaussian configuration to avoid aliasing
    def fgaussian(kx0, ky0):
        kappa = 2.5
        return np.exp(-(kx0 ** 2 + ky0 ** 2) / kappa)


    k0x = fftpack.fftfreq(nx, 1 / nx)
    k0y = fftpack.fftfreq(ny, 1 / ny)
    k0x_, k0y_ = np.meshgrid(k0x, k0y, indexing="ij")
    # u0 = fftpack.ifftshift(fftpack.fft2(fgaussian(k0x_, k0y_))).real/5
    # ________________________________________________________________________________________________________
    # ________________________________________________________________________________________________________

    from sympy import symbols, solve, exp, Eq


    def equilibrium_solution(delta, mu):
        # Add this code where you need the solution
        x = symbols('x')

        equation = Eq(exp(delta * x) * (1 - x), mu)
        solutions = solve(equation, x)

        # Print real solutions
        real_solutions = [sol.evalf() for sol in solutions if sol.is_real]
        print(f"\nEquilibrium solutions for exp({delta}x)(1-x) = {mu}:")

        # Convert symbolic solutions to float and separate positive/negative
        float_solutions = [float(sol.evalf()) for sol in real_solutions]
        pos_sol = max(float_solutions)
        neg_sol = min(float_solutions)

        # If only one solution exists, use it for both
        if len(float_solutions) == 1:
            pos_sol = neg_sol = float_solutions[0]

        return [neg_sol, pos_sol]


    ubar = equilibrium_solution(delta, mu)[1]
    print('ubar', ubar)
    # Small positive random fluctuations around equilibrium (up to +5% of ubar)
    # fluctuation_amplitude = 0.5
    u0 = ubar + np.random.rand(nx, ny)
    # Random fluctuations around equilibrium (±5% of ubar)
    # fluctuation_amplitude = 0.05
    # u0 = float(ubar)*(1 + fluctuation_amplitude * (2*np.random.rand(nx, ny) - 1))
    u = np.copy(u0)
    # print(u0)
    #
    # Plot Initial Configuration
    # plt.imshow(u.T, cmap="gnuplot", origin="lower", extent=np.concatenate((bounds, bounds)))
    # plt.colorbar(ticks=np.linspace(np.min(u), np.min(u) + 0.9 * (np.max(u) - np.min(u)), 7))
    # plt.show()
    # plt.close()

    # # ________________________________________________________________________________________________________
    # ________________________________________________________________________________________________________

    # Time setup
    dt = 0.01
    dt = min(dt, (dx * dx + dy * dy) / D / 8)
    T = 8000  # simulation duration
    t = np.arange(0, T + dt, dt)
    nt = len(t)

    # Variables to store results
    vec_time = [t[0]]
    total_density = [np.mean(u)]  # Update Average concentration over time window
    density2 = [np.mean(u)]  # Equilibrium concentration (tracked during the simulation)

    error = 10

    # Create the arrays of frequencies that will be used in the simulation
    kx = fftpack.fftfreq(nx, L / nx)
    ky = fftpack.fftfreq(nx, L / ny)
    count = 0
    # Loop over time steps
    for n in tqdm(range(1, nt)):
        # Use this if want to turn on the flow later
        if flow_type == "sinusoidal" and n <= 8000:
            gamma = 0
        else:
            gamma = g
        # if n<= 15000:
        #     gamma = 0
        # elif 15000< n <= 20000:
        #     gamma = 15* D / comp_rad
        # else:
        #     gamma = 0
        # Use if want the flow from begining
        # gamma = g
        # Compute time step
        u = rk4_pseudospectral(u, vx, vy, gamma, D, f_hat, c_hat, dt, n_step=1, kx=kx, ky=ky)
        assert u.any() >= 0, f"Negative density at time {n})"

        # Save total concentration each dt
        vec_time.append(t[n])
        density2.append(np.mean(u))
        if n % 2000 == 0:
            # Create file for saving results
            h5file = h5py.File(f"{path}/dat.h5", "w")
            h5file.create_dataset(f"t", data=u)  # save concentration
            h5file.create_dataset(f"time_", data=vec_time)
            h5file.create_dataset(f"density_", data=density2)
            h5file.close()

            # # Save data set configuration
            # if n % 5000 == 0:
            #     #     #     # Create HDF5 dataset for the current timestep
            #     #     #     h5file.create_dataset(f"t{round(t[n], 3)}", data=u)
            #     #     #
            #     #     ### Plot
            plt.subplots(1, 2, figsize=(10, 5))
            plt.subplots_adjust(wspace=0.05)
            plt.subplot(1, 2, 1)
            plt.imshow(u.T, cmap="gnuplot", origin="lower", extent=np.concatenate((bounds, bounds)))
            plt.colorbar(ticks=np.linspace(np.min(u), np.min(u) + 0.9 * (np.max(u) - np.min(u)), 7))
            #
            plt.xlim([bounds[0], bounds[1]])
            plt.title(f"t = {t[n]:0.3f};")
            plt.subplot(1, 2, 2)
            line = np.array(density2)
            plt.plot(vec_time, line, c="k")
            plt.title(f"A /r L = {np.mean(u)  : .4f};", fontsize=10)
            #
            #     #     # # Choose to show plot live or save
            # plt.show()
            #     #     #
            plt.savefig(f"{path}/fig{count:3d}")
            plt.close()
            # count += 1

        # # USE THIS FOR HEAT MAP EQUILIBIRUM:  Every 50 seconds save the mean and compute the relative error
        # if n % (5000) == 0:
        #     total_density.append(np.mean(density2))
        #     ##     print(np.mean(density))
        #
        #     error = abs(total_density[-2] - total_density[-1]) / total_density[-2]
        #
        # if error < 0.005 and n >= int(nt / 2):
        #     # if error < 0.005 and n == nt -5000:
        #     h5file.create_dataset(f"t{t[n]}", data=u)  # save concentration
        #     break

    # # save results
    # h5file.create_dataset("time", data=vec_time)
    # h5file.create_dataset("tot_density", data=total_density)
    # h5file.create_dataset("density", data=density2)
    # h5file.close()
