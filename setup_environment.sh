#!/bin/bash
set -e

conda env create -f environment.yml
conda activate mftcoder
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD/DDFA:$PYTHONPATH"
