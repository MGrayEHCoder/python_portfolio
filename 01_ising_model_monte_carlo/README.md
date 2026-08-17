# Ising Model Monte Carlo Simulation

## Overview

This project implements a Monte Carlo simulation of the two-dimensional Ising model using Python. The simulation models a square lattice of interacting spins and examines how the total energy of the system changes as a function of temperature.

The simulation uses a 32 × 32 spin lattice. Each spin is initialized randomly to either +1 or -1, and the system is then evolved through repeated spin updates based on the energy change associated with flipping an individual spin.

## Methods

The simulation includes:

* Random initialization of the spin lattice
* Temperature-dependent spin updates
* Periodic boundary conditions
* Calculation of the total system energy
* Monte Carlo sampling over multiple time steps
* Data collection over a range of temperatures
* Export of simulation results to CSV

The simulation evaluates temperatures from 1.5 to 3.3 in increments of 0.2 and records the calculated energy as the simulation progresses.

## Python Libraries

The project uses:

* NumPy
* Matplotlib
* Pandas

See `requirements.txt` for the required Python packages.

## Running the Simulation

Clone or download this repository and navigate to the project directory.

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Run the simulation with:

```bash
python ising_model.py
```

The simulation generates a CSV file containing the energy data:

```text
IsingModelData.csv
```

## Output

The primary output is a CSV file containing the calculated system energy at different temperatures and simulation steps.

The original project also includes functionality for visualizing the evolution of the spin lattice and plotting total energy versus time.

## Project Structure

```text
01-ising-model-monte-carlo/
├── ising_model.py
├── requirements.txt
└── README.md
```

## Future Improvements

Potential improvements to the simulation include:

* Adding magnetization calculations
* Measuring additional thermodynamic quantities
* Improving the organization of simulation parameters
* Adding automated visualization of simulation results
* Comparing simulation results across different lattice sizes
* Investigating behavior near the critical temperature
