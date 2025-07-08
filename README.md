
# Shear and Environmental Flow

This project is a numerical integration of partial differential equations (PDEs) using a pseudospectral approach and the Runge-Kutta 4th order (RK4) method. The code is part of the research for the paper [Title in blank].

## Project Structure

- `Vegetation/Codes/_NonLocal_FC.py`: Contains the implementation of the vegetation model.
- `Logistic/Codes/LogisticV1.py`: Contains the implementation of the logistic model.
- `Swift_Hohenberg/Codes/Run_Flow_Integration.py`: Contains the implementation of the Swift-Hohenberg model.

## Requirements

The following packages are necessary to run the code:

- `numpy`: For numerical operations.
- `scipy`: For scientific computations, mainly FFT.
- `sympy`: For symbolic mathematics.
- `matplotlib`: For plotting results.
- `h5py`: For handling HDF5 files.
- `tqdm`: For displaying progress bars.
- `numba`: For JIT compilation to speed up the code.

It is also required to have a Python version >= 3.6.

## Installation

You can install the required packages using `pip`:

```bash
pip install numpy scipy sympy matplotlib h5py tqdm numba
```

## Usage

To run the code, navigate to the directory containing the scripts and execute them using Python:
Each script should be run individually and they have input parameters:
- LogisticV1.py:
  - mu: is the Damkholer number (growth rate)
  - pe: is the Peclet number (advection rate)
  - flowtype: Stationary flow choosen to apply
    - sinusoidal, rankine, parabolic, cellular, point vortex, constant


- _NonLocal_FC.py:
  - mu: decay ratio
  - delta: enhancing constants difference (\chi_f - \chi_c)
  - flowtype:
    - sinusoidal, rankine, parabolic, cellular, point vortex, constant
  - pe: is the Peclet number (advection rate)
    
```bash
python Vegetation/Codes/_NonLocal_FC.py mu delta flowtype pe
python Logistic/Codes/LogisticV1.py mu pe flowtype
```

## Description

The base output of the code is a figure showing the Spatial Pattern Distribution.
This figure is refreshed every 2000 time steps.
If you want to save the data, you should uncomment the final lines of the code and choose if you want to save the field distribution or only the Density time series.


### _NonLocal_FC.py

This script performs the numerical integration of Eqs. 2.16 of the paper, using a pseudospectral approach and RK4. It includes the following steps:
Is the unnormalized version of the vegetation model. 

Therefore the `mu` parameter is defined from (0,1). `mu > 1` will is not defined.

`delta` is the \lambda parameter defined in the paper.

`pe` in this model is not the Peclet number, but the advection rate.

- Initialization of parameters and variables.
- Definition of the equilibrium solution using symbolic mathematics.
- Time-stepping loop to update the solution using the RK4 method.
- Saving results and plotting the configuration at specified intervals.


### LogisticV1.py

This script implements the logistic model for the given problem. It includes:

This script runs the unnormalized version of the nonlocal logistic model. However it uses as input the normalize parameters defined in the paper.

`mu` is the Damkholer number (normalize growth rate) defined in the paper.
`pe` is the Peclet number (normlazid advection rate).

- Initialization of parameters and variables.
- Definition of the logistic equation.
- Numerical integration using the RK4 method.
- Saving results and plotting the configuration at specified intervals.

## License

This project is licensed under the MIT License.
