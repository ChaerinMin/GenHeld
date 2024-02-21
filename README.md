```bash
git clone --recursive https://github.com/ChaerinMin/GenHeld.git
```

# Installation

This code was implemented under `Ubuntu 22.04`, `NVIDIA RTX 4060 Ti`, `cuda 11.8`, and `gcc 10.5.0`.

1. Create a conda environment
    ```bash
    conda create -n genheld python=3.8.18
    conda activate genheld
    pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
    ```

2. Install packages 

    ```bash
    conda install -c fvcore -c iopath -c conda-forge fvcore iopath
    pip install -r requirements.txt
    pip install -U openmim
    mim install mmcv-full==1.7.2
    ```

3. V-HACD

   Install [v-hacd](https://github.com/kmammou/v-hacd) and change `/home/cmin5/v-hacd/app/build/TestVHACD` to your own path at [configs/metric/metric.yaml](configs/metric/metric.yaml).

4. Blender

    Visit [Blender](https://www.blender.org/download/),  download, and unzip. We used version 4.0.2. If your file system is like below, 

    ```bash
    <home directory>
    |-- blender
    |   |-- lib
    |   |-- license
    |   |-- blender
    |   |-- blender.desktop
    ```

     add `<home directory>/blender` to `$PATH`.
    

# Datasets

## Required data

Download all files and folders from https://drive.google.com/drive/folders/1klKIZ7CsX0gnNH0c741VU88YeI5mOgJZ?usp=sharing and place them under `assets/` The file structure should look like below.
```bash
GenHeld
|-- assets
|   |-- contact_zones.pkl
|   |-- lama
|   |   |-- big-lama
|   |   |   |-- ...
|   |   |-- default.yaml
|   |-- sam
|   |   | -- sam_vit_h_4b8939.pth
|   |-- ...
|-- main.py
|-- ...
```

For reference, find [MANO](https://mano.is.tue.mpg.de/), [NIMBLE](https://github.com/reyuwei/NIMBLE_model?tab=readme-ov-file), [SMPLX](https://smpl-x.is.tue.mpg.de/), [SAM](https://github.com/facebookresearch/segment-anything), [LAMA](https://github.com/advimman/lama), [EgoHOS](https://github.com/owenzlz/EgoHOS) and [HiFiHR](https://github.com/viridityzhu/HiFiHR/tree/main?tab=readme-ov-file). We adopted the contact_zones.pkl from [Obman](https://github.com/hassony2/obman_train/tree/master), and preprocessed SMPLX arms from [HARP](https://github.com/korrawe/harp). 
    
## Hands

### [FreiHAND](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html)

Run the following code to download FreiHAND.

```bash
cd <your storage>
mkdir FreiHAND
cd FreiHAND
wget https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip  # 3.7GB
unzip FreiHAND_pub_v2.zip  # 4.6GB
rm FreiHAND_pub_v2.zip
cd <GenHeld>
ln -s <abs path to FreiHAND> data/
```

Then, your file system should look like below.

```bash
GenHeld
|-- data
|   |-- FreiHAND
|   |   |-- evaluation
|   |   |   |-- ...
|   |   |-- ...
|-- main.py
|-- ...
```

## Objects

### [YCB](https://www.ycbbenchmarks.com/object-models/)

Run the following code to download YCB.

```bash
python data/download_ycb.py --download_dir <your storage>
ln -s <abs path to YCB> data/
```

Then, your file system should look like below.

```bash
GenHeld
|-- data
|   |-- FreiHAND
|   |   |-- evaluation
|   |   |-- ...
|   |-- YCB
|   |   |-- 001_chips_can
|   |   |-- ...
|-- main.py
|-- ...
```