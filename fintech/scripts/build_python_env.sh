#!/bin/bash

virtualenv ~/my/venv -p /usr/bin/python3.6
source venv/bin/activate
pip install tensorflow==1.14.0
cd ~/my/stable_baselines/
pip install -e .[test]
pip install pandas
pip install sklearn
pip install empyrical
pip install ta==0.4.7
pip install openpyxl
deactivate
echo "export PATH=\"\$PATH:/home/efi/my/venv/bin\"" >> ~/.bashrc