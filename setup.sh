#!/bin/bash
# setup.sh
# Installs missing system libraries for OpenCV

apt-get update
apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6
echo "System libraries for OpenCV installed."