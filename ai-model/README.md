# RAISE Wind Turbine AI Model

https://gitlab.com/AlbertPM/raise-toy-models/-/wikis/home

## Installation

### Conda Environment

conda create --name raise python=3.11

* PyTorch CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

* PyTorch Lightning
conda install lightning -c conda-forge

* Ray Tune
conda install -c conda-forge "ray-tune"

* Matplotlib
conda install -c conda-forge matplotlib

* scikit-learn
conda install -c conda-forge scikit-learn

* onnx
conda install -c conda-forge onnx