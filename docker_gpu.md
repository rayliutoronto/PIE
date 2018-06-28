## TensorFlow GPU in Docker on Ubuntu 16.04
### install docker-ce
```
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
apt-cache policy docker-ce
sudo apt-get install -y docker-ce
sudo systemctl status docker
```
### add current user in docker group
```
sudo usermod -aG docker ${USER}
su - ${USER}
```
### install NVIDIA Graphics Driver
```
curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo dpkg -i ./cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda_drivers
sudo reboot
```
### preparation of installing nvidia-docker2: Add the package repositories
```
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
```
### Install nvidia-docker2 and reload the Docker daemon configuration
```
sudo apt-get install nvidia-docker2 -y
sudo pkill -SIGHUP dockerd
```
### Test nvidia-smi with the latest official CUDA image
```
docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
```
###  launches the latest TensorFlow GPU binary image in a Docker container
```
nvidia-docker run -it -p 8888:8888 tensorflow/tensorflow:1.8.0-gpu-py3 bash
```
### in docker
```
apt-get update
apt-get install git wget unzip -y
cd ~

wget https://s3.amazonaws.com/open-source-william-falcon/cudnn-9.0-linux-x64-v7.1.tgz  
tar -xzvf cudnn-9.0-linux-x64-v7.1.tgz  
cp cuda/include/cudnn.h /usr/local/cuda/include
cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

#Add these lines to end of ~/.bashrc:
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda

nvcc  --version
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2

git clone https://github.com/FlowAnywhere/PIE.git
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
mkdir PIE/data/word_vectors
mv glove*.txt PIE/data/word_vectors
cd PIE
export PYTHONPATH=$PWD
cd tests
python test_train.py
```