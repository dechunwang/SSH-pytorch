# A pytorch implementation of SSH (Single shot headless) face detector

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1728203.svg)](https://doi.org/10.5281/zenodo.1728203)

## Introduction
This project is a pytorch implementation of a Single shot headless face detector. Original paper can be find here [**SSH face detector ICCV 2017 paper**](https://arxiv.org/abs/1708.03979)

official implementation is in Caffe [**SSH/mahyarnajibi**](https://github.com/mahyarnajibi/SSH)

During my implementation of pytorch version, I referred the Faster-RCNN pytorch implementation [jwyang/faster-rcnn.pytorch]().

Compare to other faster-rcnn pytorch implementation, jwyang's implementation revised all layers from numpy to pytorch. I used anchor_target_layer , generate_anchor_box, propsal_layer and bbox layer for his implementation.

### What i included in this repository:
- [x] Support python3
- [x] Support pytorch-1.0
- [x] Support distributed training (gradient average)
- [ ] Support multi-gpu(Multi-GPU can be done by using mulitiple distributed training on a single machine. The reason it is not support right now is because of the input size is not a fixed size.)
- [x] Match mAP with original paper (current pretrain model have lower mAP (0.88)in wider face dataset that official implementation (0.907). ~~Maybe more training is needed. Great thanks if anyone can help me find out what went wrong)~~
i fixed a problem in anchor_target_layer. Now i can get( 0.905, 0.890, 0.809) vs SSH:(0.919,0.907,0.809)


### Installation
1. Clone this repository
2. Create conda env by using pytorch.yml provide in this repository
3. Run ```make``` in the ```model``` directory:
```
cd model
make
```

### Pretrained model
download pretrained model by [this link](https://drive.google.com/file/d/19bmuol6CbSqL3pj9SBzUL6UhrxC3XYbC/view?usp=sharing)
place file under check_point/  folder

### Training a model
For training on the *WIDER* dataset, you need to download the [WIDER face training images](https://drive.google.com/file/d/0B6eKvaijfFUDQUUwd21EckhUbWs/view?usp=sharing) and the [face annotations](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip) from the [dataset website](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/). These files should be copied into ```data/datasets/wider/``` (you can create symbolic links if you prefer to store the actual data somewhere else).

Before starting to train  you should have a directory structure as follows:
 ```
data
   |--datasets
         |--wider
             |--WIDER_train/
             |--wider_face_split/
   |--imagenet_models
         |--VGG16.caffemodel
```

For training with the default parameters, you can call the ```train.py```. As an example:

```
python3 train.py
```
For a list of all possible options run
```python3 train.py --help```.

### Training on distributed system (Or multi GPU)
On each node, you can run ```train_dist.py```
For example
```
python3 train_dist.py --master_ip xxx.xxx.xxx.xxx --master_port xxxxx --rank 0 --world_size 2
```
--master_ip is your master node ip address and also --master_port. You need to make sure that each node would able to communicate with  master node via TCP/IP. Rank 0 means master node, rank range from[0,world_size). World_size define total number of node you want to use.

You can treat a node with multi GPU as a multi node with single gpu. by doing this, we can utilize multi GPU on a single machine. For example
```
python train_dist.py --gpu='0' --rank=0 --world_size=2 --master_ip='127.0.0.1' --master_port=29500
python train_dist.py --gpu='1' --rank=1 --world_size=2 --master_ip='127.0.0.1' --master_port=29500
```
### Test a trained model
For test a image with trained model, you can call the ```test.py```. As an example:
```
python3 test.py
```
For a list of all possible options run
```python3 train.py --help```.
### Evaluating a trained model
The evaluation on the *WIDER* dataset is based on the official *WIDER* evaluation tool which requires *MATLAB*.
you need to download the [validation images](https://drive.google.com/file/d/0B6eKvaijfFUDd3dIRmpvSk8tLUk/view?usp=sharing) and
the [annotations](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip) (if not downloaded for training) from the
*WIDER* [dataset website](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/). These files should be copied into the ```data/datasets/wider``` directory as follows:
 ```
data
   |--datasets
         |--wider
             |--WIDER_val/
             |--wider_face_split/
```
For evaluating with the default parameters, you can call the ```eval.py```. As an example:

```
python3 eval.py
```
For a list of all possible options run
```python3 eval.py --help```.

### Citation
```
@misc{dechunwang_2018_1728203,
  author       = {dechunwang},
  title        = {dechunwang/SSH-pytorch: First release},
  month        = nov,
  year         = 2018,
  doi          = {10.5281/zenodo.1728202},
  url          = {https://doi.org/10.5281/zenodo.1728202}
}
```
