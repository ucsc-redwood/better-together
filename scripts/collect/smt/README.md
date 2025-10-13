# SMT Package for Schedule Optimization

This package provides modular components for solving schedule optimization problems using Satisfiability Modulo Theories (SMT).

## Package Structure

```
smt/
├── __init__.py           # Package initialization
├── baselines.py          # Baseline data and configuration management
├── data_loader.py        # CSV data loading and processing utilities
├── constraints.py        # SMT constraints and optimization setup
├── solver.py            # Main SMT solver orchestration
├── solution_analyzer.py  # Solution analysis and output formatting
└── README.md            # This documentation
```

## Module Descriptions

### `baselines.py`
- Contains baseline timing data for different device/app/backend combinations
- Provides functions to retrieve baseline information
- Manages application-specific stage counts

### `data_loader.py`
- Handles CSV file loading and data preprocessing
- Computes average timings across multiple runs
- Determines GPU backend (CUDA vs Vulkan) based on data availability
- Provides data structure definitions for the optimization problem

### `constraints.py`
- Defines SMT decision variables and constraints
- Implements assignment constraints (each stage assigned to exactly one core type)
- Implements chunk time constraints (minimizing gap between max and min chunk times)
- Implements contiguity constraints (cores appear in continuous blocks)
- Provides solution blocking functionality for finding multiple solutions

### `solver.py`
- Main orchestration module for the optimization process
- Coordinates constraint setup and solution finding
- Manages multiple solution generation
- Provides high-level interface for the optimization problem

### `solution_analyzer.py`
- Analyzes and formats optimization solutions
- Generates detailed solution representations with metrics
- Creates unique solution identifiers
- Handles JSON output formatting
- Calculates load balancing metrics

## Usage

The main script `02_gen_schedule_merged.py` demonstrates how to use this package:

```python
from smt.baselines import get_baseline_for_config
from smt.data_loader import load_csv_and_compute_averages
from smt.solver import solve_optimization_problem
from smt.solution_analyzer import dump_solutions_as_json

# Get baseline data
baseline_data = get_baseline_for_config(device, app, backend)

# Load and process CSV data
stage_timings, use_cuda = load_csv_and_compute_averages(csv_path, app)

# Solve the optimization problem
solutions = solve_optimization_problem(
    stage_timings, baseline_data, num_solutions, app
)

# Output results
dump_solutions_as_json(solutions, baseline_data, "pretty", output_file)
```

## Benefits of Modular Design

1. **Separation of Concerns**: Each module has a specific responsibility
2. **Maintainability**: Easier to modify individual components without affecting others
3. **Testability**: Each module can be tested independently
4. **Reusability**: Components can be reused in different contexts
5. **Readability**: Smaller, focused modules are easier to understand

## Dependencies

- `z3`: SMT solver library
- `pandas`: Data processing
- `numpy`: Numerical computations
- Standard library: `json`, `os`, `sys`, `argparse`, `hashlib`

## Future Extensions

The modular design makes it easy to:
- Add new constraint types
- Implement different optimization objectives
- Support additional data formats
- Add new analysis metrics
- Create alternative solvers 