# Schedule Analysis Results Package

This package contains modular components for parsing and analyzing schedule log files. The original monolithic script has been broken down into focused, reusable modules.

## Module Structure

### Core Modules

- **`log_parser.py`** - Handles log file discovery and parsing
- **`statistics.py`** - Calculates and displays statistics  
- **`model_comparison.py`** - Handles model predictions and comparisons
- **`visualization.py`** - Creates charts and plots
- **`parse_schedules_main.py`** - Main orchestrator

## Usage

### Command Line Interface

```bash
python optimizer/orchestrate/04_parse_schedules.py /path/to/log/files [options]
```

### Options

- `--verbose, -v` - Print detailed statistics for each log file
- `--model, -m` - Path to JSON file containing model predictions
- `--output, -o` - Output directory for visualization files
- `--time-window, -t` - Time window for analysis (format: start-end, default: 0.0-1.0)
- `--max-schedules, -n` - Maximum number of schedules to analyze

## Benefits

1. **Maintainability** - Each module has a single responsibility
2. **Reusability** - Modules can be imported and used independently
3. **Testability** - Individual functions can be tested in isolation
4. **Readability** - Smaller files are easier to understand and modify
5. **Extensibility** - New functionality can be added as new modules 