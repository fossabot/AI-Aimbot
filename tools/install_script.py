#install needed packages and set up environment

import subprocess
import sys
import os

cwd = os.getcwd()

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

with open(cwd+'/tools/requirements.txt') as f:
    package_list = f.read().splitlines()

for package in package_list:
    install(package)