# MATIG
=====
we will update README.md on next Monday(2025.2.17)
=====
Official code of "Masked Autoencoders for Point Cloud with Text and Image Guidance"(MATIG).

## Installation
We provide two enviroments for running the MATIG inference or (and) training locally
1. Conda 
```
conda create -n MATIG python=3.9
conda activate MATIG 
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
# pip install -U git+https://github.com/NVIDIA/MinkowskiEngine # optional
# conda install -c dglteam/label/cu113 dgl
pip install omegaconf torch_redstone einops tqdm open3d 
# pip install huggingface_hub # if you download the training data
```


2. docker
We also provide a Docker environment to facilitate easy setup and consistent development across different platforms. [baidu disk](https://pan.baidu.com/s/1a_WEaRfEI25FzwyxcyCLUQ) (47m7). You can assess the running container with the following command:
```shell
docker run --gpus all --shm-size=64g -v Your_matig_path:/points -it pitml_exp:v1.1 /bin/bash
```
To use the Chamfer Loss functionality, you need to compile it first. Follow the steps below to build the Chamfer Loss module.
```shell
cd Point-MAE/extensions/chamfer_dist
python setup.py install
```

## inference
To perform model inference, follow these steps:
1. Download the pre-trained model MATIG from [baidu disk](https://pan.baidu.com/s/1s4VZgOe0_mK_oopufkXSeQ)(p9gg) or [google drive](https://drive.google.com/drive/folders/19Xt45384v1vX6PExgD5cVwAPzSefiAUL)
2. Use the following command to perform inference with the downloaded model:
```shell
python src/main.py --config src/configs/MATIG_LVIS.yaml
```
or
```shell
python src/main.py --config src/configs/MATIG_noLVIS.yaml
```
## train
```shell
python src/main.py --train --config src/configs/MATIG_LVIS.yaml --trial_name MATIG_LVIS model.name=MATIG model.use_dense=True training.lr=0.0005 training.lr_decay_rate=0.967
```

## Citation

