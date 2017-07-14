# Installation Guide

# Install Anaconda (replace the particular URL for your platform)
wget https://repo.continuum.io/archive/Anaconda2-4.4.0-Linux-x86_64.sh
bash Anaconda2-4.4.0-Linux-x86_64.sh
# Make sure you let the installed add conda in your system path
source ~/.bashrc

# Install OpenCV 3 (would resolve most dependenices)
conda install -c conda-forge opencv=3.2.0

# Install easydict
conda install -c auto easydict=1.4

# Install libgcc
conda install libgcc

# Install protobuf
pip install protobuf

sudo apt-get install git

git clone https://github.com/aivsol/autonomous-driving.git

cd autonomous-driving/

git submodule update --init /--recursive

cd core/py-faster-rcnn

# Build Caffe
# Follow the instructions at:
# https://github.com/rbgirshick/py-faster-rcnn

# Make sure you have downloaded the pre-traned models for the demo
# 1. Download faster_rcnn_models using the step 5 of the README.md of
#    https://github.com/rbgirshick/py-faster-rcnn and place it at
#    $REPO_ROOT/api/resources/weights/faster_rcnn_models/
# 2. Download GTSDB weights using the step 5 of the README.md of
#    https://github.com/sridhar912/tsr-py-faster-rcnn and place it at 
#    $REPO_ROOT/api/resources/weights/GTSDB/

# Change directory to the root of autonomous-driving repository
export FLASK_APP=main.py
flask run
