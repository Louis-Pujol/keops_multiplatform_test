import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA by unsetting CUDA_PATH')
args = parser.parse_args()

if args.no_cuda:
    os.environ.pop('CUDA_PATH', None)  # Safely remove CUDA_PATH if it exists
    print("CUDA_PATH environment variable unset. CUDA will not be used.")

import numpy as np
import pykeops

pykeops.test_numpy_bindings()
pykeops.test_torch_bindings()
