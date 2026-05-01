#!/bin/bash

set -e

clear

echo "[INFO] Build Virtual Python3 Env."
python3 -m venv venv
echo "[INFO] Venv build completed"

echo "[INFO] Activate the python3 venv"
source venv/bin/activate

echo "[INFO] Install python3 required package"
pip install -r requirements.txt
echo "[INFO] Install completed"

deactivat
