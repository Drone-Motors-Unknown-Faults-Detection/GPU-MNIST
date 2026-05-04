#!/bin/bash

set -e

clear

echo "[INFO] Active Python3 Virtual Environments"
source venv/bin/activate

clear
echo "[INFO] Start to run resnet program"
python3 src/resnet.py

echo "[INFO] Done"
