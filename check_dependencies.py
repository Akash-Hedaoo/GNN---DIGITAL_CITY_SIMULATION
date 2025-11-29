"""
Dependency Checker for Digital Twin City Simulation
=====================================================
Run this script to verify all dependencies are installed correctly.
"""

import sys

def check_dependencies():
    """Check if all required dependencies are installed"""
    
    print("🔍 Checking dependencies...\n")
    
    dependencies = {
        'torch': 'PyTorch',
        'torch_geometric': 'PyTorch Geometric',
        'networkx': 'NetworkX',
        'numpy': 'NumPy',
        'plotly': 'Plotly',
        'streamlit': 'Streamlit',
        'pandas': 'Pandas'
    }
    
    all_good = True
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            version = __import__(module).__version__ if hasattr(__import__(module), '__version__') else 'installed'
            print(f"✅ {name:20s} - {version}")
        except ImportError:
            print(f"❌ {name:20s} - NOT INSTALLED")
            all_good = False
    
    print("\n" + "="*60)
    
    if all_good:
        print("✅ All dependencies are installed!")
        print("\nYou can now run: streamlit run streamlit_gui.py")
    else:
        print("❌ Some dependencies are missing!")
        print("\nTo install missing dependencies, run:")
        print("  pip install torch torch-geometric networkx numpy plotly streamlit pandas")
    
    print("="*60)
    
    return all_good


if __name__ == "__main__":
    check_dependencies()
