# Polymer Structure Classification Using Neural Networks

## Overview
This project investigates the use of neural networks to classify polymer structures generated from Monte Carlo simulation data.

Two different neural network approaches are developed and compared:

1. A **phase-space/property-based neural network**
2. A **coordinate-based neural network**

The goal is to investigate whether polymer structural classification can be performed directly from molecular coordinate information and how that compares with classification using derived phase-space and physical properties.

---

## Project Structure
```
03_polymer_structure_classification/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── polymer_phase_space_data_formatting.py
├── polymer_coordinate_data_formatting.py
│
├── polymer_phase_space_NN_training.py
├── polymer_coordinate_NN_training.py
│
├── polymer_neural_network_test.py
│
├── data/
│   └── Local simulation data (not included)
│
└── output/
    └── Trained models (not included)
```

## Neural Network Approaches 

### 1. Phase-Space / Property-Based Network 
The first neural network uses physical and statistical properties derived from the polymer simulation. This derivation does take some processing on the part of the researcher.

Coordinate information is removed during preprocessing, leaving the network to classify polymer structures using the remaining features. 

The network architecture is :
```
Input
  ↓
40 neurons — Swish
  ↓
50 neurons — ReLU
  ↓
25 neurons — ELU
  ↓
Output — Softmax
```
The number of output neurons is determined automatically from the number of structure classes present in the dataset.

### 2. Coordinate-Based Network
The second neural network uses the polymer's coordinate information directly.

The XYZ coordinates of the polymer aer retained while simulation and derived propery information is removed. This coordinate data comes directly from the simulations.

Before training, each polymer is translated so that its minimum coordinate values are shifted to the origin and the resulting features are normalized.

The network architecture is:
```
Input
  ↓
Input-sized layer — ReLU
  ↓
Input-sized layer — ReLU
  ↓
Input-sized layer — ReLU
  ↓
60 neurons — Swish
  ↓
30 neurons — ELU
  ↓
15 neurons — GELU
  ↓
Output — Softmax
```

## Data Preprocessing 
The preprocessing programs perform several operations before the data is supplied to the neural networks.

### 1. Phase-Space Data
The phase-space preprocessing program:
* Loads the original simulation data
* Removes polymer coordinate information
* Removes simulation metadata and derived quantities that are not used as network inputs
* Separates the structure classifications labels
* Randomizes the dataset
* Automatically maps structure classes to consecutive zero-based labels
* Splits the data into training and testing datasets
* Normalizes the input features using the training dataset
* Saves the processed datasets locally

### 2. Coordinate Data
The coordinate preprocessing program:
* Loads the original simulation data
* Removes simulation metadata and derived quantities
* Separates the structure classifications labels
* Randomizes the dataset
* Automatically maps structure classes to consecutive zero-based labels
* Translates each polymer relative to its coordinate minimum
* Splits the data into training and testing datasets
* Normalizes the input features using the training dataset
* Saves the processed datasets locally

## Class Label Handling
The original simulation data does not necessarily contain every possible structure class in every dataset.

For example, one dataset may contain:
```
0 1 2 3 4 5 7
```
while another may contain:
```
0 1 2 3 4 5 6 7
```
The preprocessing programs automatically detect the classes present and map htem to consecutie zero-based labels.

For example:
```
Original: 0 1 2 3 4 5 7
Mapped:   0 1 2 3 4 5 6
```
This allows the neural networks to work with datasets containing different combinations of structure classes without manually changing the code.

## Training and Testing
The dataset is divided into separate training and testing sets.

The training data is used to train the neural networks, while the test data is held out for evaluating their performance.

Normalization parameters are calculated using the traning data only. The same parameters are then applied to the test data to avoid allwing information from the test to influence the training process.

The test program evaluates both neural networks and reports their classification accuracy and loss.

## Running the Project

### 1. Install Dependencies
From the project directory:
```Bash
py -m pip install -r requirements.txt
```

### 2. Prepare Data
```Bash
py preprocessing/data_formatting_phase.py
py preprocessing/data_formatting_coordinate.py
```

### 3. Train the Networks
```Bash
py polymer_coordinates_NN_training.py
py polymer_phase_space_NN_training.py
```

### 4. Test the Networks
```Bash
py polymer_NN_test.py
```
The trained models and processed datasets are generaated locally and are intentionally excluded from the GitHub repository.

## Data
The original polymer simulation data used by this project was generated using a Monte Carlo simulation that was not created as part of this project.

For that reason, the raw and processed datasets are not included in this repository.

The preprocessing code neessary to transform the dat for the neural networks is included.

## Technologies
* Python
* NumPy
* Pandas
* TensorFlow / Keras
* Neural Networks
* Supervised machine learing

## Purpose
This project was developed as an exploration of applying machine learning techniques to computational polymer physics.

The primary focus is comparing whether polymer structural information can be learned from:
* Derived physical/phase-space properties
* Direct monomer coordinate information

The comparison provides insight into how effective neural networks are for this sort of problem, and how much a researcher must work to make them effective.

## Acknowledgements
This project was done under the mentorship of Dr. Matthew Williams, Assistant Professor of Physics at Murray State University.