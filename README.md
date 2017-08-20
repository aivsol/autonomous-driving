# Installation Guide

#### Install general requisite tools
```
sudo apt-get install git

sudo apt-get install vim

sudo apt-get install cmake
```

#### Install Anaconda (replace the URL with the one that is suitable for your platform)
```
wget https://repo.continuum.io/archive/Anaconda2-4.4.0-Linux-x86_64.sh
bash Anaconda2-4.4.0-Linux-x86_64.sh
```

#### Install required conda packages
```
conda install -c conda-forge opencv=3.2.0

conda install -c auto easydict=1.4

conda install libgcc

conda install -c menpo ffmpeg

conda install nomkl
```

#### Install protobuf
````
pip install protobuf
````


#### Caffe pre-reqs
````
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler

sudo apt-get install --no-install-recommends libboost-all-dev

sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev

sudo apt-get install libatlas-base-dev

sudo apt-get install libopenblas-base
````

#### Build CAFFE (you would need to tweak some files if you want to build caffe for CPU only mode)
````
cd core/sub-modules/py-faster-rcnn
Follow the instructions at: https://github.com/rbgirshick/py-faster-rcnn
````
#### Install Tensorflow
````
Follow the instructions at: https://www.tensorflow.org/install/install_linux
````
#### Build aivsol-TFFRCNN
````
cd core/sub-modules/aivsol-TFFRCNN/lib/
# If you are building for CPU, set gpu_mode=False in config.py file in lib
make
````

#### Place model weights
````
# CAFFE GTSDB Weights
Download from https://drive.google.com/file/d/0B0CHhxRP_jmIRlVKR250d0pMNEE/view

Place the download file at: autonomous-driving/api/resources/faster-rcnn/weights/GTSDB/

# CAFFE pascal_voc Weights:
Obtain model weights by following point 5 under “Installation (sufficient for demo)”  at https://github.com/rbgirshick/py-faster-rcnn

Place the download file at: autonomous-driving/api/resources/faster-rcnn/weights/pascal_voc/

# TF GTSDB Weights:
Download from https://drive.google.com/drive/folders/0Bw4dQNbrw0jdMldWTS1GcldxY2s

Place the download file at: autonomous-driving/api/resources/tffaster-rcnn/weights/GTSDB

# TF pascal_voc Weights:
Download from https://drive.google.com/file/d/0B_xFdh9onPagVmt5VHlCU25vUEE/view?usp=sharing

Place the download file at: autonomous-driving/api/resources/tffaster-rcnn/weights/pascal_voc
````

#### Run the application
````
export FLASK_APP=main.py
flask run
````