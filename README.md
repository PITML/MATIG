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
we also provide a docker enviroment.



## Citation

