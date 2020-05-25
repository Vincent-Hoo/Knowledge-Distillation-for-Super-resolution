# Knowledge Distillation for Super-Resolution
### Introduction

This repository is the official implementation of the paper **"FAKD: Feature-Affinity Based Knowledge Distillation for Efficient Image Super-Resolution"** from **ICIP 2020**. In this work, we propose a novel and efficient SR model, name Feature Affinity-based Knowledge Distillation (FAKD), by transferring the structural  knowledge of a heavy teacher model to a lightweight student model. To transfer the structural knowledge effectively, FAKD aims to distill the second-order statistical information from feature maps and trains a lightweight student network with low computational and memory cost. Experimental results demonstrate the efficacy of our method and superiority over other knowledge distillation based methods in terms of both quantitative and visual metrics.

### Main Results

Here is the quantitative results (PSNR and SSIM) of RCAN and SAN with and w/o FAKD.



**Note:**

- RCAN is from the paper [Image Super-Resolution Using Very Deep Residual Channel Attention Networks](www.arxiv.org/abs/1807.02758).

- SAN is from the paper [Second-order Attention Network for Single Image Super-resolution](http://openaccess.thecvf.com/content_CVPR_2019/html/Dai_Second-Order_Attention_Network_for_Single_Image_Super-Resolution_CVPR_2019_paper.html).

### Quick Start

#### Dependencies

- python 3.6.9
- pytorch 1.1.0
- skimage 0.15.0
- numpy 1.16.4
- imageio 2.6.1
- matplotlib
- tqdm

#### Data Preparation

We use [DIV2K]() dataset to train our model.