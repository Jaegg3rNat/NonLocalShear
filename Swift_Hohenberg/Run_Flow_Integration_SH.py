import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import sys
import os
from scipy import fftpack
from tqdm import tqdm
import h5py

# --- RK4 Integration Methods ---
def rk4_pseudospectral(u, vx, vy, g, D, m2, dt, n_step, kx, ky):
    """
    Performs a 4th order Runge-Kutta integration in pseudospectral space.

    Parameters:
    u (ndarray): The initial field.
    vx (ndarray): Velocity field in the x-direction.
    vy (ndarray): Velocity field in the y-direction.
    g (float): Scaling factor for the velocity fields.
    D (float): Diffusion coefficient.
    m2 (ndarray): Competition Kernel used in the update step.
    dt (float): Time step for integration.
    n_step (int): Number of integration steps.
    kx (ndarray): Wavenumbers in the x-direction.
    ky (ndarray): Wavenumbers in the y-direction.

    Returns:
    ndarray: Updated Real field after integration.
    """
    kx_, ky_ = np.meshgrid(kx, ky, indexing="ij")

    for i in range(n_step):
        u_hat = fftpack.fft2(u)

        # RK4 steps
        k1_hat = update_step(u_hat, kx_, ky_, vx, vy, D, m2, g)
        k2_hat = update_step(u_hat + dt / 2 * k1_hat, kx_, ky_, vx, vy, D, m2, g)
        k3_hat = update_step(u_hat + dt / 2 * k2_hat, kx_, ky_, vx, vy, D, m2, g)
        k4_hat = update_step(u_hat + dt * k3_hat, kx_, ky_, vx, vy, D, m2, g)

        u_hat += dt / 6 * (k1_hat + 2 * (k2_hat + k3_hat) + k4_hat)
        u = fftpack.ifft2(u_hat).real

    return u

def update_step(u_hat, kx_, ky_, vx, vy, D, m2, g):
    """
    Computes a single time step update for the RK4 integration in Fourier space.

    Parameters:
    u_hat (ndarray): Fourier-transformed field.
    kx_ (ndarray): Wavenumbers in the x-direction.
    ky_ (ndarray): Wavenumbers in the y-direction.
    vx (ndarray): Velocity field in the x-direction.
    vy (ndarray): Velocity field in the y-direction.
    D (float): Diffusion coefficient.
    m2 (ndarray): Additional field used in the update step.
    g (float): Scaling factor for the velocity fields.

    Returns:
    ndarray: Updated Fourier-transformed field.
    """
    u = fftpack.ifft2(u_hat).real
    term1 = -(4 * np.pi ** 2) * D * (kx_ ** 2 + ky_ ** 2) * u_hat
    term2 = -1j * (2 * np.pi) * kx_ * fftpack.fft2(g * vx * u)
    term3 = -1j * (2 * np.pi) * ky_ * fftpack.fft2(g * vy * u)
    term4 = r * u_hat - 1 / K * fftpack.fft2(u * fftpack.ifftshift(fftpack.ifft2(u_hat * fftpack.fft2(m2))))

    return term1 + term2 + term3 + term4

def semi_implicit_pseudospectral(u, vx, vy, r, g, D, m2, dt, n_step, kx, ky):
    kx_, ky_ = np.meshgrid(kx, ky, indexing="ij")
    k2 = kx_**2 + ky_**2
    
    linear = r - (1 - 4*np.pi**2*D * k2)**2

    for i in range(n_step):
        u_hat = fftpack.fft2(u)
        
        nonlinear = -1j * 2*np.pi*kx_ * fftpack.fft2(g*vx*u)
        nonlinear += -1j * 2*np.pi*ky_ * fftpack.fft2(g*vy*u)
        nonlinear += fftpack.fft2(u**2-u**3)
        
        u_hat = (u_hat + dt*nonlinear)/(1-dt*linear)
        
        u = fftpack.ifft2(u_hat).real
    
    return u





# --- Velocity Field Definitions ---
# @njit(parallel=True)
def pv_field_domain(x_, y_, pvs, strengths, bounds, periodic_repeats=1):
    """
    Computes the point vortex velocity field within a specified domain.

    Parameters:
    x_ (ndarray): X-coordinates of the meshgrid.
    y_ (ndarray): Y-coordinates of the meshgrid.
    pvs (ndarray): Coordinates of point vortices.
    strengths (ndarray): Strengths of each point vortex.
    bounds (tuple): Bounds of the domain.
    periodic_repeats (int): Number of periodic repetitions for boundary conditions.

    Returns:
    tuple: Velocity fields (vx, vy).
    """
    L = bounds[1] - bounds[0]
    vx = np.zeros_like(x_)
    vy = np.zeros_like(x_)

    for n in range(len(pvs)):
        for j in range(x_.shape[0]):
            for k in range(y_.shape[1]):
                vx_temp, vy_temp = 0, 0
                for i in range(-periodic_repeats, periodic_repeats + 1):
                    x, y = x_[j, k], y_[j, k]
                    dx, dy = x - pvs[n, 0], y - pvs[n, 1]

                    if dx - i * L != 0 or dy != 0:
                        vx_temp -= np.sin(2 * np.pi * dy / L) / (
                                np.cosh(2 * np.pi * dx / L - 2 * np.pi * i) - np.cos(2 * np.pi * dy / L))
                    if dx != 0 or dy - i * L != 0:
                        vy_temp += np.sin(2 * np.pi * dx / L) / (
                                np.cosh(2 * np.pi * dy / L - 2 * np.pi * i) - np.cos(2 * np.pi * dx / L))

                vx[j, k] += strengths[n] / (2 * L) * vx_temp
                vy[j, k] += strengths[n] / (2 * L) * vy_temp

    return vx, vy

# @njit(parallel=True)
def pv_field_domain2(x_, y_, pvs, strengths, bounds, periodic_repeats=2):
    """
    Computes a cellular vortex velocity field within a specified domain.

    Parameters:
    x_ (ndarray): X-coordinates of the meshgrid.
    y_ (ndarray): Y-coordinates of the meshgrid.
    pvs (ndarray): Coordinates of point vortices (not used in this function).
    strengths (ndarray): Strengths of the vortices.
    bounds (tuple): Bounds of the domain.
    periodic_repeats (int): Number of periodic repetitions for boundary conditions.

    Returns:
    tuple: Velocity fields (vx, vy).
    """
    L = bounds[1] - bounds[0]
    vx = np.zeros_like(x_)
    vy = np.zeros_like(x_)

    for j in range(x_.shape[0]):
        for k in range(y_.shape[1]):
            x, y = x_[j, k], y_[j, k]
            vx_temp = -np.sin(np.pi * x / L) * np.cos(np.pi * y / L)
            vy_temp = np.sin(np.pi * y / L) * np.cos(np.pi * x / L)

            vx[j, k] += strengths[0] * vx_temp
            vy[j, k] += strengths[0] * vy_temp

    return vx, vy

# @njit(parallel=True)
def pv_field_domain3(x_, y_, strengths, bounds, periodic_repeats=2):
    """
    Computes a Rankine point vortex velocity field within a specified domain.

    Parameters:
    x_ (ndarray): X-coordinates of the meshgrid.
    y_ (ndarray): Y-coordinates of the meshgrid.
    strengths (ndarray): Strengths of the vortices.
    bounds (tuple): Bounds of the domain.
    periodic_repeats (int): Number of periodic repetitions for boundary conditions.

    Returns:
    tuple: Velocity fields (vx, vy).
    """
    L = bounds[1] - bounds[0]
    q = 0.05 * L
    vx = np.zeros_like(x_)
    vy = np.zeros_like(x_)

    for j in range(x_.shape[0]):
        for k in range(y_.shape[1]):
            vx_temp, vy_temp = 0, 0
            for i in range(-periodic_repeats, periodic_repeats + 1):
                x, y = x_[j, k], y_[j, k]
                r = np.sqrt((x - 0) ** 2 + (y - 0) ** 2)

                if r >= q:
                    vx_temp -= np.sin(2 * np.pi * y / L) / (
                            np.cosh(2 * np.pi * x / L - 2 * np.pi * i) - np.cos(2 * np.pi * y / L))
                    vy_temp += np.sin(2 * np.pi * x / L) / (
                            np.cosh(2 * np.pi * y / L - 2 * np.pi * i) - np.cos(2 * np.pi * x / L))
                else:
                    vx_temp -= y * 2



# --- Simulation Parameters ---
print('#####################################################################')
print('############# SIMULATION PARAMETERS ############################')

# Seed for reproducibility of initial condition
seed = 3
np.random.seed(seed)

# Define system bounds and length
bounds = np.array([-0.5, 0.5])
L = bounds[1] - bounds[0]  # Length of the domain

# Grid parameters for the x-direction
nx = 128
# nx = int(sys.argv[1])
dx = L / nx
x = np.linspace(*bounds, nx + 1)[1:]  # Exclude the first point for periodicity

# Grid parameters for the y-direction
# ny = 128
ny = nx
dy = L / ny
y = np.linspace(*bounds, ny + 1)[1:]  # Exclude the first point for periodicity

# Competition radius for the kernel
comp_rad = 0.2

# Mesh grid of numerical space (physical domain)
x_, y_ = np.meshgrid(x, y, indexing="ij")

# Print system parameters
print('System interval:', bounds)
print('System Length:', L)
print('Number of points (nx, ny):', nx, ny)
print('Delta x:', dx)

# Biological and simulation parameters
D = 1e-4  # Diffusion coefficient
# mu = float(sys.argv[1])  # Adimensional birth rate
mu = 250
r = mu * D / comp_rad ** 2  # Birth rate
r = 0.1
K = 1  # Carrying capacity
# pe = float(sys.argv[2])  # Peclet Number
pe = 10

# List of velocity multipliers to consider based on Peclet Number
gamma = [pe * comp_rad * (D / comp_rad ** 2)]
# for i in range(1, 5):
#     gamma.append((pe + 5 * i) * comp_rad * (D / comp_rad ** 2))

# Print biological parameters
print('\n')
print('########### Biological Parameters ########')
print('Competition Radius (R_comp):', comp_rad)
print('Diffusion Coefficient (D):', D)
print('Adimensional Growth Rate (mu):', mu, '// Growth Rate (r):', r)
print('Peclet Number (Pe):', pe, '// Initial Velocity Multiplier (gamma[0]):', gamma[0])

# --- Non-local Competition Kernel Section ---
# Define number of points in the kernel based on competition radius
r_int = int(comp_rad / dx)

# Initialize competition kernel with normalization
m = np.zeros((1 + 2 * r_int, 1 + 2 * r_int))
m_norm = np.pi * comp_rad ** 2  # Normalization factor for the kernel in 2D

# Populate the kernel with values based on the competition radius
for i in range(-r_int, r_int + 1):
    for j in range(-r_int, r_int + 1):
        if i ** 2 + j ** 2 <= r_int ** 2:
            m[i + r_int, j + r_int] = 1. / m_norm

# for i in range(-r_int, r_int + 1):
#     for j in range(-r_int, r_int + 1):
#         if i ** 2 + j ** 2 <= r_int ** 2:
#             m[i + r_int, j + r_int] = 1

# m_norm = m.sum()*dx*dy
# m /= m_norm

# Create a larger grid for the kernel in the domain
m2 = np.zeros((nx, ny))
m2[nx // 2 - r_int:nx // 2 + r_int + 1, ny // 2 - r_int:ny // 2 + r_int + 1] = m

# Adjust kernel normalization with grid spacing
m2 *= dx * dy


# plt.imshow(m2)
# plt.show()

# # --- Simulation Parameters ---
# print('#####################################################################')
# print('############ SIMULATION PARAMETERS ############################')
# # Seed for reproducibility of initial conditions
# seed = 3
# np.random.seed(seed)
# bounds = np.array([-0.5, 0.5])
# L = bounds[1] - bounds[0]

# nx = 128
# dx = L / nx
# x = np.linspace(*bounds, nx + 1)[1:]  # periodic in x

# ny = 128
# dy = L / ny
# y = np.linspace(*bounds, ny + 1)[1:]  # periodic in y

# comp_rad = 0.2  # Competition radius of Kernel

# # Mesh grid of numerical space (physical)
# x_, y_ = np.meshgrid(x, y, indexing="ij")

# print('System interval:', bounds)
# print('Systems Length:', L)
# print('N points:', nx)
# print('Delta x:', dx)

# D = 1e-4  # Diffusion coefficient
# mu = float(sys.argv[1])  # Adimensional birth rate
# r = mu * D / comp_rad ** 2  # Birth rate
# K = 1  # Carrying capacity
# pe = float(sys.argv[2])  # Peclet Number
# gamma = [pe * comp_rad * (D / comp_rad ** 2)]  # List of velocity multipliers to consider
# for i in range(1, 5):
#     gamma.append((pe + 5 * i) * comp_rad * (D / comp_rad ** 2))

# print('\n###########Biological Parameters########')
# print('R comp:', comp_rad)
# print('Diffusion:', D)
# print('Adimensional Growth:', mu, '// Growth rate:', r)
# print('Peclet:', pe, '// Velocity:', gamma[0])

# # --- Non-Local Competition Kernel Section ---
# r_int = int(comp_rad / dx)  # Define number of points in kernel
# m = np.zeros((1 + 2 * r_int, 1 + 2 * r_int))
# m_norm = np.pi * comp_rad ** 2  # Normalization of Kernel in 2D
# for i in range(-r_int, r_int + 1):
#     for j in range(-r_int, r_int + 1):
#         if i ** 2 + j ** 2 <= r_int ** 2:
#             m[i + r_int, j + r_int] = 1. / m_norm

# m2 = np.zeros((nx, ny))
# m2[nx // 2 - r_int:nx // 2 + r_int + 1, ny // 2 - r_int:ny // 2 + r_int + 1] = m
# m2 *= dx * dy

# --- Simulation Start ---
print('#####################################################################')
print('############ SIMULATION START ####################')

# Dynamically create base folder name based on competition radius
base_folder = f"R_{comp_rad:.1f}"
if base_folder not in os.listdir():
    os.mkdir(base_folder)


# Loop over different values for the velocity multiplying factor gamma
for g in gamma:
    mu = (r * comp_rad ** 2) / D
    Pe = (g / comp_rad) / (D / comp_rad ** 2)

    # --- Define Velocity Field ---
    # Set a variable for the velocity field name
    velocity_field_name = "sinusoidal"  # Default value, change it as per the chosen field

    # Sinusoidal Flow
    if True:  # Set True if using Sinusoidal Flow
        velocity_field_name = "sinusoidal"
        w = 2 * np.pi / L
        vx = np.sin(w * y_)
        vy = np.zeros_like(vx)

    # Parabolic Flow
    # if False:  # Uncomment if using Parabolic Flow
    #     velocity_field_name = "parabolic"
    #     vx = (-4 * y_ ** 2 + 4 * y_)
    #     vy = np.zeros_like(vx)

    # Constant Flow
    # if False:  # Uncomment if using Constant Flow
    #     velocity_field_name = "constant"
    #     vx = np.ones_like(x_)
    #     vy = np.zeros_like(vx)

    # Point-vortex Field
    # if False:  # Uncomment if using Point-vortex Field
    #     velocity_field_name = "point_vortex"
    #     n_vortex = 1
    #     space_repetitions = 1
    #     strengths = 2 * np.array([1])
    #     pvs = np.zeros((1, 2))
    #     vx, vy = pv_field_domain(x_, y_, pvs, strengths, bounds, space_repetitions)

    # Cellular-vortex Field
    # if False:  # Uncomment if using Cellular-vortex Field
    #     velocity_field_name = "cellular_vortex"
    #     n_vortex = 1
    #     space_repetitions = 0
    #     strengths = 1 * np.array([1])
    #     pvs = np.zeros((1, 1))
    #     vx, vy = pv_field_domain2(x_, y_, pvs, strengths, bounds, space_repetitions)

    # Rankine Point-vortex Field
    # if False:  # Uncomment if using Rankine Point-vortex Field
    #     velocity_field_name = "rankine_vortex"
    #     n_vortex = 1
    #     space_repetitions = 2
    #     strengths = 2 * np.array([1])
    #     pvs = np.zeros((1, 2))
    #     vx, vy = pv_field_domain3(x_, y_, strengths, bounds, space_repetitions)

    # --- Create Folder for Results Based on Velocity Field ---
    # Folder path now includes the velocity field name

    # dt = float(sys.argv[2])
    dt = 0.01
    # Condition for reaction diffusion
    # dt = min(dt, (dx * dx + dy * dy) / D / 8)

    path = f"{base_folder}/{velocity_field_name}_mu{mu:.2f}_w{w:.2f}_Pe{Pe:.1f}_nx{nx}_dt{dt}"
    if path.split("/")[1] not in os.listdir(base_folder):
        os.mkdir(path)


    # Create file for results
    print(f'The Velocity field choson was, {velocity_field_name}')
    # h5file = h5py.File(f"{path}/dat.h5", "w")

    # --- Initial Configuration Section ---
    print('####################################################################################')
    print('################### INITIAL CONFIGURATION #########################################')

    # Gaussian Center at origin in the momentum space
    def fgaussian(kx0, ky0):
        kappa = 5
        return np.exp(-(kx0 ** 2 + ky0 ** 2) / kappa)

    # Define initial condition in frequency/Fourier space
    k0x = fftpack.fftfreq(nx, 1 / nx)
    k0y = fftpack.fftfreq(ny, 1 / ny)
    k0x_, k0y_ = np.meshgrid(k0x, k0y, indexing="ij")
    u0 = fftpack.ifftshift(fftpack.fft2(fgaussian(k0x_, k0y_))).real  # Real initial configuration
    # u0 = np.random.normal(1, 0.1, size=x_.shape)  # Alternative initialization
    u0 = np.random.rand(nx, ny)
    u = np.copy(u0)
    plt.imshow(u.T, cmap="gnuplot", origin="lower", extent=np.concatenate((bounds, bounds)))
    plt.show()

    # Define Time Interval
    # dt = float(sys.argv[2])
    # dt = 0.01
    # # Condition for reaction diffusion
    # dt = min(dt, (dx * dx + dy * dy) / D / 8)


    # T = 2000 # Simulation time for creating Animations
    T = 1000  # Simulation duration for Equilibrium Heat Map
    t = np.arange(0, T + dt, dt)
    nt = len(t)

    vec_time = [t[0]]  # Time vector
    density = [np.mean(u)]  # Avg concentration over time
    density2 = [np.mean(u)]  # Principal density data collection
    mean_density = [1]  # Equilibrium density tracking

    error = 10  # Initial error for iterative processes

    # Create arrays of frequencies for the simulation
    kx = fftpack.fftfreq(nx, 1 / nx)
    ky = fftpack.fftfreq(nx, 1 / ny)

    # Simulation loop
    count = 0
    # Final part of the simulation loop
    for n in tqdm(range(1, nt)):
        # Use this if you want to turn on the flow later
        gamma_value = 0 if n <= 2000 else g

        # Compute time step with reaction-diffusion conditions
        # u = rk4_pseudospectral(u, vx, vy, gamma_value, D, m2, dt, n_step=1, kx=kx, ky=ky)
        u = semi_implicit_pseudospectral(u, vx, vy, r, gamma_value, D, m2, dt, n_step=1, kx=kx, ky=ky)
        # assert np.all(u >= 0), f"Negative density at time {n}"

        # Save total concentration each dt
        vec_time.append(t[n])
        density.append(np.mean(u))
        density2.append(np.mean(u))  # Main density tracking array

        # Save data every 10 time steps and plot (uncomment if needed)
        # Use this section for creating animations.
        if n % 500 == 0:
            # h5file.create_dataset(f"t{round(t[n], 3)}", data=u)  # Save concentration
            # Plotting section
            plt.subplots(1, 2,figsize = (25,16))
            plt.subplots_adjust(wspace=0.05)
            plt.subplot(1, 2, 1)
            plt.imshow(u.T, cmap="gnuplot", origin="lower", extent=np.concatenate((bounds, bounds)))
            plt.colorbar(ticks=np.linspace(np.min(u), np.min(u) + 0.9 * (np.max(u) - np.min(u)), 7))
            # Example of plotting additional data over the concentration field
            # plt.plot(g * ((-4 * y ** 2 + 4 * y)), y, ls='--', color='k', alpha=0.5)
            # plt.plot(g * (np.sin(w * y)), y, ls='--', color='k', alpha=0.5)
            plt.xlim([bounds[0], bounds[1]])
            plt.title(f"t = {t[n]:0.3f};")
            plt.subplot(1, 2, 2)
            plt.plot(vec_time, density2, c="g")
            plt.title(f"A /r L = {density2[-1] / r : .3f};")
            # Choose to show plot live or save
            # plt.show()
            # plt.pause(1)
            plt.savefig(f"{path}/fig{count:3d}")
            plt.close()
            count += 1

        # Stopping condition: calculate mean density and relative error every 5000 steps
        if n % nt//20 == 0:
            mean_density.append(np.mean(density))
            # Print mean density for debugging or monitoring
            # print(np.mean(density))
            density = [np.mean(u)]  # Reset density to current mean
            error = abs(mean_density[-2] - mean_density[-1]) / mean_density[-2]  # Compute relative error
            # h5file.create_dataset(f"t{t[n]}", data=u)  # Save the concentration at stopping point

        # Break the simulation if the relative error is below the threshold and simulation is past 20% of total time
        # if error < 0.005 and n >= int(nt / 5):
        #     h5file.create_dataset(f"t{t[n]}", data=u)  # Save the concentration at stopping point
        #     break

    # Save final results
    # h5file.create_dataset("time", data=vec_time)  # Save time vector
    # h5file.create_dataset("conc", data=mean_density)  # Save mean density over time
    # h5file.create_dataset("conc2", data=density2)  # Save main density tracking data
    # h5file.close()

