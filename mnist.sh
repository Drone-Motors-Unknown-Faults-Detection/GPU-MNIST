#!/bin/bash

set -e

clear

echo "[INFO] Active Python3 Virtual Environments"
source venv/bin/activate

clear
echo "[INFO] Start to run MNIST program"
python3 src/mnist.py

echo "[INFO] Done"
