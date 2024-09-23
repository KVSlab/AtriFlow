# AtriFlow - Atrial Flow Rates Estimation

**AtriFlow** is a Python package designed for estimating atrial flow rates in patient-specific geometries, specifically
the left atrium. It provides tools for modeling atrial flow dynamics using computational fluid dynamics (CFD) techniques
and optimization methods. The package supports both atrial fibrillation (AF) and sinus rhythm (SR) conditions.

## Features

- Perform optimization of flow rates for patient-specific geometries
- Compute flow rates for atrial fibrillation (AF) and sinus rhythm (SR)
- Visualize and analyze atrial flow using various plotting tools
- Configurable flow rate calculation using geometrical and hemodynamic data

## Installation

### Requirements

- Python 3.10 or later
- Dependencies:
    - `matplotlib>=3.0`
    - `numpy>=1.19.0`
    - `pandas>=1.0`
    - `scipy>=1.5.0`
    - `seaborn>=0.11.0`

To install the package and its dependencies, run:

```bash
pip install .
```

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

This will store flow rate files in the 'data/flow_rates' folder.

## Author

Henrik A. Kjeldsberg [Email](mailto:henrik.kjeldsberg@live.no)