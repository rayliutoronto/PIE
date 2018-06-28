## TensorFlow GPU in Docker on Ubuntu 16.04
### install docer-ce
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
sudo apt-get install cuda_drivers
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
sudo apt-get install nvidia-docker2
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
export PYTHONPATH=$PWD
git clone https://github.com/FlowAnywhere/PIE.git
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
mkdir PIE/data/word_vectors
mv glove*.txt PIE/data/word_vectors
cd PIE/tests
python test_train.py
```