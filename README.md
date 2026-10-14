# Multi-Scale Physics-Informed Neural Networks (MS-PINNs)

A research-oriented implementation of **Multi-Scale Physics-Informed Neural Networks (MS-PINNs)** for solving differential equations, with experiments on ordinary differential equations (ODEs) and the nonlinear Burgers' equation.

## Overview

Physics-Informed Neural Networks (PINNs) incorporate governing differential equations into neural network training by combining data-driven objectives with physics-based constraints.

This project explores a multi-scale approach to PINNs and provides a codebase for running experiments, evaluating model behavior, and comparing implementations.

## Objectives

- Implement baseline Physics-Informed Neural Networks.
- Explore a multi-scale PINN architecture.
- Solve representative differential equations.
- Organize training, loss computation, and evaluation into modular components.
- Provide a foundation for further research and experimentation.

## Experiments

The repository includes experiments for the following problems:

| Problem | Description |
|---|---|
| Decay equation | An ordinary differential equation experiment |
| Oscillator | An experiment involving oscillatory dynamics |
| Burgers' equation | A nonlinear partial differential equation benchmark |

## Project Structure

```text
Multi-Scale-PINN/
├── src/
│   ├── equations/
│   │   └── burgers.py
│   ├── evaluation/
│   │   └── metrics.py
│   ├── losses/
│   │   └── pinn_loss.py
│   ├── models/
│   │   ├── pinn.py
│   │   └── final_msf_pinn.py
│   └── training/
│       └── trainer.py
├── experiments/
│   ├── burgers/
│   │   └── run_burgers.py
│   ├── decay/
│   │   └── run_decay.py
│   └── oscillator/
│       └── run_oscillator.py
└── README.md
```

*This tree highlights the main components; additional files may be present in the repository.*

## Methodology

The project separates key parts of the PINN workflow into dedicated modules:

- **Models:** Neural network implementations.
- **Loss functions:** Physics-informed training objectives.
- **Training:** Model optimization and training workflow.
- **Equations:** Problem-specific equation definitions.
- **Evaluation:** Metrics for assessing model predictions.
- **Experiments:** Individual scripts for the selected differential equations.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Mahdi-hz83/Multi-Scale-PINN.git
cd Multi-Scale-PINN
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

Install the dependencies required by the project. If a `requirements.txt` file is available, run:

```bash
pip install -r requirements.txt
```

### 4. Run an experiment

The repository contains experiment scripts for the decay equation, oscillator, and Burgers' equation. For example:

```bash
python experiments/decay/run_decay.py
```

Check the corresponding script for its configuration and execution requirements.

## Technologies

- Python
- Physics-Informed Neural Networks
- Numerical methods for differential equations

*Add specific framework versions and additional libraries once verified against the project dependencies.*

## Future Work

Potential directions for further development include:

- Quantitative comparisons between baseline PINNs and MS-PINNs.
- Error analysis and convergence studies.
- Sensitivity analysis across network configurations and random seeds.
- Improved visualization of predictions and residual errors.

## Author

**Mahdi Hosseinzadeh**

GitHub: [@Mahdi-hz83](https://github.com/Mahdi-hz83)

## License

No license has been specified yet. Add a license if you intend to permit reuse or redistribution of this code.
