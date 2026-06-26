# Subprojects Setup Guide

## Overview
The `subprojects/` directory has been created to house experimental work, prototypes, and personal analyses that should not be tracked in the main phenofusion repository.

## What's Been Set Up

### 1. `.gitignore` File
Created a comprehensive `.gitignore` that excludes:
- The entire `subprojects/` directory
- Python bytecode and virtual environments
- Jupyter notebooks (*.ipynb)
- Large data files (*.pkl, *.parquet)
- Model checkpoints (*.pth)
- Output directories
- Logs and SLURM output files

### 2. Subprojects Directory Structure
```
subprojects/
├── README.md                    # Overview of subprojects directory
└── example_subproject/          # Template subproject
    ├── README.md                # Subproject documentation
    ├── scripts/                 # Python analysis scripts
    │   └── example_analysis.py
    ├── notebooks/               # Jupyter notebooks
    ├── data/                    # Local data files
    ├── outputs/                 # Results and figures
    └── bashscripts/             # SLURM job scripts
        └── run_analysis.sh
```

## Creating a New Subproject

1. **Create a new directory**:
   ```bash
   cd /burg-archive/home/al4385/phenofusion/subprojects
   mkdir my_new_subproject
   cd my_new_subproject
   ```

2. **Copy the template structure** (optional):
   ```bash
   cp -r ../example_subproject/* .
   ```

3. **Update the README** to describe your subproject

4. **Add your code** - everything in `subprojects/` is automatically ignored by git

## Using Main Phenofusion Code

Your subproject can import from the main phenofusion package:

```python
import sys
from pathlib import Path

# Add phenofusion to path
phenofusion_root = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(phenofusion_root))

# Import from phenofusion
from phenofusion.dataio import TFT_process
from phenofusion.analysis import MetricsCalculator
```

## Running Subproject Code

### Locally:
```bash
cd subprojects/my_subproject
python scripts/my_analysis.py
```

### On SLURM:
```bash
sbatch subprojects/my_subproject/bashscripts/run_analysis.sh
```

## Benefits

- **Experiment freely** without affecting the main codebase
- **No git clutter** - your work-in-progress code stays local
- **Reuse phenofusion utilities** - import from the main package
- **Independent workflows** - each subproject is self-contained
- **Easy cleanup** - delete subprojects when done

## Git Verification

To verify a directory is ignored:
```bash
git status --ignored subprojects/
```

You should see: `Ignored files: subprojects/`

## Notes

- The `subprojects/` directory itself is NOT tracked, so your work stays completely local
- You can still manually track specific files if needed with: `git add -f subprojects/file.py`
- The example_subproject is just a template - feel free to delete or modify it
- Each subproject should have its own README explaining its purpose
