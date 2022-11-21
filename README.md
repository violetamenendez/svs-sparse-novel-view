# SVS: Adversarial refinement for sparse novel view synthesis
PyTorch Lightning implementation of paper "SVS: Adversarial refinement for sparse novel view synthesis", BMVC 2022.

> SVS: Adversarial refinement for sparse novel view synthesis
> [Violeta Menéndez González](https://github.com/violetamenendez), [Andrew Gilbert](https://www.andrewjohngilbert.co.uk/), [Graeme Phillipson](https://www.bbc.co.uk/rd/people/graeme-phillipson), [Stephen Jolly](https://www.bbc.co.uk/rd/people/s-jolly), [Simon Hadfield](http://personal.ee.surrey.ac.uk/Personal/S.Hadfield/biography.html)
> BMVC 2022
>

#### [paper](https://arxiv.org/abs/2211.07301)

![Architecture](images/architecture.png)

## Installation

#### Tested on Ubuntu 20.04 + Python 3.8 + Pytorch 1.10.1 + Pytorch Lightning 1.5.8

```
conda env create -f environment.yml
```

Alternatively, you can install the conda environment manually:
```
conda create -n svs python=3.8 pip
conda activate svs
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
python -m pip install pytorch-lightning==1.5.8
conda install pillow scipy
python -m pip install inplace_abn kornia
python -m pip install configargparse imageio opencv-python lpips coloredlogs
```

## Datasets

### 1. Training datasets

#### (a) [**DTU**](https://roboimagedata.compute.dtu.dk/?page_id=36)
Download the preprocessed [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view)
and [Depth_raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip) from original [MVSNet repo](https://github.com/YoYo000/MVSNet)
and unzip. We provide an example in `dtu/`.

#### (b) [**LLFF**](https://bmild.github.io/llff/) released scenes
Download and process [real_iconic_noface.zip](https://drive.google.com/drive/folders/1M-_Fdn4ajDa0CS8-iqejv0fQQeuonpKF) (6.6G) using the following commands:
```angular2
# download
gdown https://drive.google.com/uc?id=1ThgjloNt58ZdnEuiCeRf9tATJ-HI0b01
unzip real_iconic_noface.zip
```

### 2. Test datasets
Download `nerf_llff_data.zip` from [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) and add it to your LLFF folder

## Training
```
python train.py --expname $exp_name --num_epochs $num_epochs --use_viewdirs --dataset_name $dataset --datadir $dataset --save_dir $save_dir --batch_size $batch_size --configdir $config_dir --patch_size $patch_size --precision $precision --gan_loss $gan_loss --gan_type $gan_type --lambda_rec $lambda_rec --with_distortion_loss --lambda_distortion $l_dist --with_depth_smoothness --lambda_depth_smooth $l_ds --pts_embedder --lambda_adv $l_adv --with_perceptual_loss --lambda_perc $l_perc --lrate $lrate --lrate_disc $lrate_disc
```

## Testing
```
python test.py --expname $exp_name --num_epochs 1 --use_viewdirs --save_dir $save_dir --dataset_name $dataset --datadir $data_dir --configdir $config_dir --ckpt $ckpt --pts_embedder
```

## Citation
```
@inproceedings{menendez2022svs,
  author    = {Menéndez González, Violeta and Gilbert, Andrew and Phillipson, Graeme and Jolly, Stephen and Hadfield, Simon},
  title     = {SVS: Adversarial Refinement for Sparse Novel View Synthesis},
  booktitle = {BMVC},
  year      = {2022}
}

```