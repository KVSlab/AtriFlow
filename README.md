# AtriFlow - Atrial Flow Rates Estimation

**AtriFlow** is a Python package designed for estimating atrial flow rates in patient-specific geometries, specifically
the left atrium. It provides tools for modeling atrial flow dynamics using computational fluid dynamics (CFD) techniques
and optimization methods. The package supports both atrial fibrillation (AF) and sinus rhythm (SR) conditions.

## Features

- Perform optimization of flow rates for patient-specific geometries
- Compute flow rates for atrial fibrillation (AF) and sinus rhythm (SR)
- Visualize and analyze atrial flow using various plotting tools
- Configurable flow rate calculation using geometrical and hemodynamic data

## Requirements

- Python 3.10 or later
- Dependencies:
    - `matplotlib>=3.0`
    - `numpy>=1.19.0`
    - `pandas>=1.0`
    - `scipy>=1.5.0`
    - `seaborn>=0.11.0`

## Installation

To install `AtriFlow`, clone the repository:

```bash
git clone https://github.com/KVSlab/AtriFlow.git
```

Then navigate inside the `AtriFlow` folder and install its dependencies in a Python environment:

```bash
python -m pip install .
```

Successful execution will install `AtriFlow` to your computer.

### Optional Development Dependencies

If you want to run tests and perform code formatting, install the optional dependencies:

```bash
pip install .[test]
```

## Usage

### Optimize Flow Rates

To optimize the flow rates for AF or SR conditions:

```bash
atriflow-optimize-af
```

```bash
atriflow-optimize-sr
```

### Compute and visualize flow Rates

To compute and visualize atrial fibrillation (AF) or sinus rhythm (SR) flow rates, you can use the following commands:

```bash
atriflow-af
```

```bash
atriflow-sr
```

This will store flow rate files in the `data/flow_rates/flow_rates_[CONDITION]` folder for conditions `af` and `sr`.
Filenames will indicate the model used, e.g. `flow_rate_[model-name]_Q-A.txt` for the **Q-A**-model

## Author

Henrik A. Kjeldsberg [Email](mailto:henrik.kjeldsberg@live.no)
