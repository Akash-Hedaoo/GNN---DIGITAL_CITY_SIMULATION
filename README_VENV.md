# Using the Virtual Environment

This project uses a virtual environment (`.venv`) with all dependencies pre-installed.

## Quick Start

### Option 1: Activate the venv (Recommended)
```powershell
# In PowerShell
. .\activate_venv.ps1

# Then run scripts normally
python step1_acquire_larger_network.py
python step2_feature_engineering.py
# etc...
```

### Option 2: Use helper scripts
```powershell
# Run any script directly with venv
.\run_with_venv.ps1 step1_acquire_larger_network.py
```

Or in Command Prompt:
```cmd
run_with_venv.bat step1_acquire_larger_network.py
```

### Option 3: Direct Python call
```powershell
.venv\Scripts\python.exe step1_acquire_larger_network.py
```

## Running the Full Pipeline

Execute the steps in order:

```powershell
# Activate venv first
. .\activate_venv.ps1

# Then run the pipeline
python step1_acquire_larger_network.py
python step2_feature_engineering.py
python step3_generate_training_data.py
python step4_train_model.py
python step5_whatif_analysis.py
python step6_visualize_results.py
```

## Verify Installation

Check if dependencies are installed:
```powershell
.venv\Scripts\python.exe -c "import torch; import torch_geometric; print('✅ All dependencies installed!')"
```

