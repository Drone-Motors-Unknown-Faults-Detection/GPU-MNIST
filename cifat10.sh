#!/bin/bash

set -e

clear

echo "[INFO] Active Python3 Virtual Environments"
source venv/bin/activate

clear
echo "[INFO] Start to run Cifar10 program"
python3 src/cifar10.py

echo "[INFO] Done"
