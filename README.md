# VAE
Variational AutoEncoder on DTU dataset
## Installation

#### Tested on Ubuntu 20.04 + Pytorch 1.10.1 + Pytorch Lignting 1.3.5

Install environment:
```
conda create -n casmvsnerf python=3.8
conda activate casmvsnerf
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install pytorch-lightning==1.3.5 imageio pillow scikit-image opencv-python configargparse lpips kornia warmup_scheduler matplotlib test-tube imageio-ffmpeg
pip install inplace-abn==1.1.0
```


## Training
Please see each subsection for training on different datasets. Available training datasets:

* [DTU](#dtu)

### DTU dataset

#### Data download

Download the preprocessed [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view)
and [Depth_raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip) from original [MVSNet repo](https://github.com/YoYo000/MVSNet)
and unzip.