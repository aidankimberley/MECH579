# MECH597 - Shared Virtual Environment

This folder contains a shared Python virtual environment for all MECH597 assignments and projects.

## 🚀 Quick Start

### Activate the virtual environment:
```bash
cd /Users/aidan1/Documents/McGill/MECH597
./activate_venv.sh
# OR
source venv/bin/activate
```

### Install required packages:
```bash
pip install -r requirements.txt
```

### Run Python files from any assignment:
```bash
# From the MECH597 root directory (after activation)
python Assignment_3/root_finder.py
python Assignment_3/SQP.py
python Assignment1/Assignment1.py
python "Assignment 2"/brequet_range_optimizer.py
```

### Deactivate when done:
```bash
deactivate
```

## 📁 Project Structure
```
MECH597/
├── venv/                    # Shared virtual environment
├── requirements.txt         # Package dependencies
├── activate_venv.sh        # Convenience activation script
├── README.md               # This file
├── Assignment1/            # Assignment 1 files
├── Assignment 2/           # Assignment 2 files
└── Assignment_3/           # Assignment 3 files
```

## 📦 Included Packages
- **Core Scientific Computing**: NumPy, SciPy, Matplotlib, Pandas
- **Advanced Computing**: JAX (for high-performance computing)
- **Symbolic Math**: SymPy
- **Machine Learning**: Scikit-learn
- **Optimization**: CVXPY
- **Development**: Jupyter, IPython

## 🔧 Usage Examples

### Running Assignment Files:
```bash
# Activate environment
source venv/bin/activate

# Run any Python file from any assignment
python Assignment_3/root_finder.py
python Assignment_3/SQP.py
python Assignment1/quick_calcs.py
```

### Adding New Packages:
```bash
# Activate environment first
source venv/bin/activate

# Install new package
pip install package_name

# Update requirements.txt
pip freeze > requirements.txt
```

### Working with Jupyter Notebooks:
```bash
# Activate environment
source venv/bin/activate

# Start Jupyter
jupyter notebook
```

## 💡 Benefits of Shared Environment
- **Consistency**: Same package versions across all assignments
- **Efficiency**: No need to recreate environments for each assignment
- **Organization**: Centralized package management
- **Collaboration**: Easy to share exact environment setup

## 🛠️ Troubleshooting

### If packages are missing:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### If you need to recreate the environment:
```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Check what's installed:
```bash
source venv/bin/activate
pip list
```
