import sys
import os
import streamlit as st
print(f"--- DEBUG START ---")
print(f"Python executable: {sys.executable}")
try:
    import torch
    print(f"Torch version: {torch.__version__}")
    print(f"Torch file: {torch.__file__}")
    st.write(f"Torch version: {torch.__version__}")
except ImportError as e:
    print(f"Torch import failed: {e}")
    st.write(f"Torch import failed: {e}")

print(f"Sys path: {sys.path}")
print(f"--- DEBUG END ---")
