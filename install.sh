#!/bin/sh

sudo apt-get update
sudo apt-get upgrade

echo "install tensorflow"
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp27-none-linux_x86_64.whl
sudo pip install --upgrade $TF_BINARY_URL

echo "install keras"
sudo pip install keras

echo "install scikit-learn"
sudo pip install scikit-learn

echo "install pandas"
sudo pip install pandas

echo "install h5py"
sudo apt-get install libhdf5-dev
sudo pip install h5py
