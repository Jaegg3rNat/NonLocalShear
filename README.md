
# Shear and Environmental Flow

This project is a numerical integration of partial differential equations (PDEs) using a pseudospectral approach and the Runge-Kutta 4th order (RK4) method. The code is part of the research for the paper
**_Flow spatial structure determines pattern instabilities in nonlocal models of population dynamics._**
Published in *Communication Physics*.




[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16080825.svg)](https://doi.org/10.5281/zenodo.16080825)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![ArXiv](https://img.shields.io/badge/ArXiv-2409.04268-b31b1b)](https://arxiv.org/abs/2409.04268)
[![Communication Physics](https://img.shields.io/badge/Communication%20Physics-Nature-red?style=for-the-badge&logo=nature&logoColor=white)](https://www.nature.com/articles/s42005-025-02246-3)



[![Python Version](https://img.shields.io/badge/Python-3.6%2B-FFD43B.svg)](https://www.python.org/)



## Autorship

#### Authors:
- Nathan O. Silvano (maintainer)
- Joao Valeriano
- Emilio Hernandez-Garcia
- Cristobal Lopez
- Ricardo Martinez-Garcia


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

It is also required to have a Python version 3.6+.

## Installation

You can install the required packages using `pip`:

```bash
pip install numpy scipy sympy matplotlib h5py tqdm numba
```

## Usage

To run the code, navigate to the directory containing the scripts and execute them using Python:
Each script should be run individually and they have input parameters:
- LogisticV1.py:
  - `mu`: is the Damkholer number (growth rate)
  -  `pe`: is the Peclet number (advection rate)
  - `flowtype`: Stationary flow choosen to apply
    - sinusoidal, rankine, parabolic, cellular, point vortex, constant


- _NonLocal_FC.py:
  - `mu`: decay ratio
  - `delta`: enhancing constants difference ($\chi_f - \chi_c$,  in the paper is defined as $\lambda$)
  - `flowtype`:
    - sinusoidal, rankine, parabolic, cellular, point vortex, constant
  - `pe`: is the Peclet number (advection rate)
    
```bash
python Vegetation/Codes/_NonLocal_FC.py mu delta flowtype pe
python Logistic/Codes/LogisticV1.py mu pe flowtype
```

## Description

### Generating Figures

The base output of the code without any modification is a figure showing the Spatial Pattern Distribution.
This figure is refreshed every 2000 time steps.

The example bellow is the typical output image of the standard code:

<img src="Vegetation/Figures/fig  0.png" alt="Example Output" width="600"/>

If you want to save the data to perform any further analysis,
you should uncomment the final lines of the code and choose if you want to save the field distribution
or only the Density time series. Adjust correctly after how many time step you want to save the data

The left side shows the spatial distribution, while the right side shows the Integral of Density /L^2 x Time plot


This is the basis output for both Vegetation and Logistic models.

In case you want to save the data, you can uncomment the last lines of the code and choose whether to save the field distribution or only the density time series.

### _NonLocal_FC.py

This script performs the numerical integration of Eqs. 2.16 of the paper, using a pseudospectral approach and RK4.
Is the unnormalized version of the vegetation model. 

Therefore the `mu` parameter is defined from (0,1). `mu > 1` is not defined.

`delta` is the $\lambda$ parameter defined in the paper.

`pe` in this model is not the Peclet number, but the advection rate (or flow intensisty $v_0$).

- Initialization of parameters and variables.
- Definition of the equilibrium solution using symbolic mathematics.
- Time-stepping loop to update the solution using the RK4 method.
- Saving results and plotting the configuration at specified intervals.


### LogisticV1.py

This script implements the logistic model for the given problem. It includes:

This script runs the unnormalized version of the nonlocal logistic model. However it uses as input the normalize parameters defined in the paper.

`mu` is the Damkholer number (normalized growth rate) defined in the paper.
`pe` is the Peclet number (normalized advection rate).

That means:

  input : `mu` is  $$\mu = \frac{\alpha R^2}{D}$$
  input : `pe` is  $$Pe = \frac{v_0 R^2}{D}$$


- Initialization of parameters and variables.
- Definition of the logistic equation.
- Numerical integration using the RK4 method.
- Saving results and plotting the configuration at specified intervals.

## Acknowledgments

This work was partially funded by the Center of Advanced Systems Understanding (CASUS), which is financed by Germany’s Federal Ministry of Education and Research (BMBF) and by the Saxon Ministry for Science, Culture and Tourism (SMWK) with tax funds on the basis of the budget approved by the Saxon State Parliament. N.O.S was partially supported by Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES) - Finance Code 001 and by the CAPES-Print program through a sandwich doctoral fellowship at CASUS. RMG and JV were partially supported by FAPESP through a BIOTA Jovem Pesquisador Grant 2019/05523-8 (RMG and JV), ICTP-SAIFR grant 2021/14335-0 (RMG), and a Master's fellowship 2020/14169-0 (JV). RMG acknowledges support from Instituto Serrapilheira (Serra-1911-31200). C.L. and E.H-G were supported by grants LAMARCA PID2021-123352OB-C32 funded by MCIN/AEI/10.13039/501100011033323 and FEDER "Una manera de hacer Europa"; and TED2021-131836B-I00 funded by MICIU/AEI/\,\newline
10.13039/501100011033 and by the European Union "NextGenerationEU/PRTR". 
E.H-G. and N.O.S. also acknowledge the Maria de Maeztu program for Units of Excellence, CEX2021-001164-M funded by MCIN/AEI/10.13039/501100011033. C.L. was partially supported by the Scultetus Center Visiting Scientist Program at CASUS.

## License

This project is licensed under the MIT License.

Copyright (c) 2025 Nathan O. Silvano

## Citation
 If you use this code, please cite the original paper as:
 
```
 @misc{silvano_2025_16080825,
  author       = {Silvano, Nathan and
                  Valeriano, Joao and
                  Hernández-García, Emilio and
                  Lopez, Cristobal and
                  Martinez-Garcia, Ricardo},
  title        = {Flow spatial structure determines pattern
                   instabilities in nonlocal models of population
                   dynamics},
  month        = jul,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.16080825},
  url          = {https://doi.org/10.5281/zenodo.16080825},
}
```
