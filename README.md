# SBI-sirens
This is the repository of the technical part of my thesis "Gravitational Wave Cosmology: From Standard to Biased Sirens", conducted under the supervision of Prof. Dr. Samaya Nissanke, at the University of Amsterdam, GRAPPA track.

## Objective
Are conventional standard sirens methods quick, accurate and precise enough for the abundance of observational data we expect in the next decades?

Considering this central question, it becomes clear that the development—and, crucially, the rigorous validation—of alternative methods for cosmological inference is both timely and necessary. This
finally leads to the techincal objectives of my Master’s thesis:

*Enhance the dark sirens galaxy catalog methods with simulation-based inference.*

## Contents
This repository consists of the following files and folders (note that the full description of the code and sbi framework is included in my thesis): 

- `SimulatorPackages.py`: Classes for the physical simulator.
- `GWFast_code.ipynb`: The physical simulator is based on [`GWFast`](https://github.com/CosmoStatGW/gwfast/tree/master). This notebook contains my debugging attempts and tests when creating `SimulatorPackages.py`.
- `SBI_notebook.ipynb`: Notebook to build the [`swyft`](https://swyft.readthedocs.io/en/stable/install.html) simulator based on the physical simulator, to simulate and save samples, as well as build several networkds as inference models to train and test.
- `SBI_notebook_ColabVersion.ipynb`: Same with `SBI_notebook.ipynb`, but for google colaboratory. Note that I run different scenarios in these two notebooks (as described in my thesis).
- `Thesis Figures`: The figures presented in my thesis.
- `SBI_galaxies.csv`,`gal_catalog.csv`,`micecat_451.csv`: Galaxy data required for simulating samples.

## Data and Results
The simulated data, as well as the training and testing results are uploaded in my [google drive](https://drive.google.com/drive/folders/12U4gAg5Eoe3ffXoD9_guNWYzWU6406gs?usp=sharing).
