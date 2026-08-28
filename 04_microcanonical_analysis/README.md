# Microcanonical Analysis of Polymer Structure Transitions

## Overview
This project analyzes polymer simulation data using a microcanonical approach to identify structural phase transitions.

The analysis builds on simulation data used in the polymer structure classification neural-network project. Rather than classifying polymer configurations with a neural network, this project examines the thermodynamic behavior of the system directly and identifies phase transitions from features in the microcanonical entropy.

The resulting analysis produces phase diagrams showing the locations of detected structural transitions as a function of energy and bending strength.

---

## Scientific Approach
The analysis uses histogram reweighting to obtain the microcanonical entropy as a function of energy.

The general workflow is:

```text
Simulation Data
      ↓
Histogram Reweighting
      ↓
Microcanonical Entropy S(E)
      ↓
Savitzky-Golay Derivatives
      ↓
Transition Detection
      ↓
Higher-Order Repeat Filtering
      ↓
Structure/Phase Regions
      ↓
Phase Diagram
```
The derivatives of the microcanonical entropy are used to identify changes in the thermodynamic behavior of the polymer system. 

Multiple derivative orders are examined in order to identify different order of phase transitions.

## Microcanonical Analysis
The core analysis begins by reconstructing the microcanonical entropy from simulation data.

For each bending strength, the program:
1. Loads the parallel-tempering simulation data.
2. Performs histogram reweighting.
3. Calculates the microcanonical entropy as a function of energy.
4. Calculates derivatives of the entropy using Savizky-Golay filtering.
5. Searches for zero crossings in the derivatives.
6. Identifies candidate phase-transition energies.
7. Removes repeated detections associated with higher-order derivatives.

The higher-order repeat filtering is important because the same physical transition can appear in multiple derivatives. The analysis therefore compares detected transition energies and removes detections that occur sufficiently close to previously identified transitions.

## Determining the Derivative Window
The Savizky-Golay filter requires a window length for calculating the derivaties.

The `findWindowLength()` functions searches through possible window lengths and determines the number of detected transitions for each one.

A stable platequ in the number of detected transitions is used to select and appropriate window length.

This helps reduce sensitivity to the particular smoothing window chosen for the derivative calculation.

## Phase Diagram
The phase diagram combines the thermodynamic transition analysis with previously classified polymer structures.

For each polymer structure, the program calculates a convex hull in energy/bending-strength space. These regions are then plotted together with the transitions locations obtained from the mocrocanonical analysis.

The resluting plot provides a visual comparison between:
* Structrural regions
* Energy
* Bending strength
* Detected phase transitions

## Phase Transitions Markers
Different marker shapes are used to distinguish the order of the detected phase transition.
```text
o  →  1st Order Phase Transition
s  →  2nd Order Phase Transition
d  →  3rd Order Phase Transition
^  →  4th Order Phase Transition
```
A legend is generated automatically when the phase diagram is created.

## Data Organization
The simulation data is expected to be organized approximately as follows:

```text
data/
└── 40/
    ├── 24-06-28_40_Constbend 0.5/
    │   └── data/
    ├── 24-06-28_40_Constbend 0.625/
    │   └── data/
    ├── 24-06-28_40_Constbend 0.75/
    │   └── data/
    ├── 24-06-28_40_Constbend 0.875/
    │   └── data/
    └── 24-06-28_40_Constbend 1.0/
        └── data/
```

Structure-label data used to generate the phase regions is also stored in the `data/` directory.

The simulation data is not included in this repository.

The data used for this analysis originated from a Monte Carlo simulation that was not generated as a part of this project, so the raw simulation data is kept separate from the GitHub repository.

## Output
Generated figures are stored in the `output/` directory.

Examples include:
```text
PhaseDiagram_L_40_ConstBend_0.5.png
PhaseDiagram_L_40_ConstBend_0.625.png
PhaseDiagram_L_40_ConstBend_0.75.png
PhaseDiagram_L_40_ConstBend_0.875.png
PhaseDiagram_L_40_ConstBend_1.0.png
```
The program can also generate microcanonical diagnostic plots containing:
* Microcanonical entropy
* First derivative
* Second derivative
* Third derivative
* Fourth derivative

These plots are useful for examining how the tranistion locations were determined.

## Running the Analysis

### 1. Install the required packages
From the project directory, run:
```Bash
py -m pip install -r requirements.txt
```

### 2. Verify the data directory
Make sure the required simulation data is located in the expected `data/` directory structure.

Because the raw data is not included in teh repository, it must be supplied separately.

### 3. Run the analysis
Run:
```Bash
py microcanonical_analysis.py
```
the current `main` function runs the phase-diagram analysis for"
```
Polymer length: 40
```
The resulting figure are saved to the `output/` directory.

## Main Functions
`histReweight()`
Performs histogram reweighting and calculates the microcanonical entropy as a function of energy.

`findWindowLength()`
Determines an appropriate Savizky-Golay filter window by examing the stability of detected transition counts.

`PlotPhaseEnergies()`
Identifies phase-transition energies from derivatives of the microcanonical entropy and plots them on the phase-region diagram.

This function also removes higher-order repeated detections so that the same transition is not counted multiple times.

`MakePhaseRegions()`
Loads previously classified polymer structures and generates convex-hull regions in energy/bending-strength space.

`MakePhaseDiagram()`
Generates diagnostic plots of the microcanonical entropy and its first four derivatives.

`make_folder()`
Constructs the paths to the simulation datasets for each strength.

## Parameters
The current analysis examines polymer systems with:
* Polymer length: `40`
* Bending strengths:
    * `0.5`
    * `0.625`
    * `0.75`
    * `0.875`
    * `1.0`

The phase-region analysis also supports different potential widths through the `width` parameter.

## Project Structure
```text
04_microcanonical_analysis/
│
├── data/
│   └── [simulation data - not tracked by Git]
│
├── output/
│   └── [generated figures - not tracked by Git]
│
├── microcanonical_analysis.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Relationship to the Polymer Neural Network Project
This project uses the same underlying polymer simulation dataset as the polymer structure classification neural network project.

The two projects approach the data from different perspectives:

| Project | Approach | Goal |
|---|---|---|
| Polymer Structure Classification | Neural Network | Classify polymer structures |
| Microcanonical Analysis | Statistical/thermodynamic analysis | Identify structural phase transitions |

The neural-network project focuses on automated classification, while this project investigates the physical transitions between structures.

Together, they provide complementary views of the polymer system.

---

## Notes
The analysis contains several parameters associated with numerical differentiation and transition detection, including the derivative cutoff and Savitzky-Golay window selection.

These parameters should be treated as part of the analysis methodology rather than arbitrary machine-learning hyperparameters.

The raw simulation and generated output files are exclued from version control.